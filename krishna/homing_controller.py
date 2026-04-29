#!/usr/bin/env python3

import math
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from can_runtime import BusRuntime


@dataclass(frozen=True)
class HomingMotorConfig:
    motor_id: int
    target_deg: float
    search_direction_deg: float
    trigger_current_A: float
    nudge_back_deg: float


@dataclass
class ControlTuning:
    active_kp: float
    active_kd: float
    hold_kp: float
    hold_kd: float
    homed_hold_kp: float
    homed_hold_kd: float
    loop_hz: float

    min_move_time_s: float
    seconds_per_deg: float

    trigger_confirm_s: float
    trigger_velocity_raw_max: float = 5.0

    feedback_timeout_s: float = 0.15
    bus_silence_timeout_s: float = 0.40

    startup_timeout_s: float = 5.0
    startup_poll_s: float = 0.01
    zero_feedback_timeout_s: float = 1.5
    zero_position_tolerance_deg: float = 5.0

    max_search_time_s: float = 10.0
    max_search_travel_deg: float = 180.0

    status_period_s: float = 0.5

    safe_idle_kp: float = 48.0
    safe_idle_kd: float = 0.8

    nudge_position_tolerance_deg: float = 2.0
    nudge_settle_time_s: float = 0.12

    search_max_vel_deg_s: float = 80.0
    search_max_acc_deg_s2: float = 120.0
    continuous_search_vel_deg_s: float = 6.0

    nudge_max_vel_deg_s: float = 60.0
    nudge_max_acc_deg_s2: float = 168.0

    advance_target_tolerance_deg: float = 2.0
    advance_settle_time_s: float = 0.20

    stall_progress_epsilon_deg: float = 0.25
    stall_timeout_s: float = 0.50

    wait_log_period_s: float = 1.0

    contact_hold_offset_deg: float = 0.5
    contact_release_move_deg: float = 3.0
    contact_stall_vel_max: float = 2.0


    def __post_init__(self):
        self.loop_dt = 1.0 / self.loop_hz


@dataclass
class MotionProfileState:
    cmd_deg: float
    vel_deg_s: float = 0.0


class BusHomingController:
    def __init__(
        self,
        runtime: BusRuntime,
        phase1_config: List[HomingMotorConfig],
        phase2_config: List[HomingMotorConfig],
        tuning: ControlTuning,
        print_lock: Optional[threading.Lock] = None,
    ):
        self.runtime = runtime
        self.phase1_config = list(phase1_config)
        self.phase2_config = list(phase2_config)
        self.tuning = tuning
        self.print_lock = print_lock or threading.Lock()

        self.cfg_by_id: Dict[int, HomingMotorConfig] = {}
        for cfg in self.phase1_config + self.phase2_config:
            self.cfg_by_id[cfg.motor_id] = cfg

    def _log(self, msg: str):
        with self.print_lock:
            print(msg, flush=True)

    def verify_periodic_feedback_and_capture_boot_holds(self):
        ok = self.runtime.wait_for_periodic_feedback(
            timeout_s=self.tuning.startup_timeout_s,
            poll_period_s=self.tuning.startup_poll_s,
            feedback_timeout_s=self.tuning.feedback_timeout_s,
            bus_silence_timeout_s=self.tuning.bus_silence_timeout_s,
        )
        if not ok:
            raise RuntimeError(self.runtime.get_fault_summary())
        self.runtime.capture_boot_hold_positions()

    def send_idle_hold_once(self):
        targets = self.runtime.get_idle_hold_targets()
        self.runtime.command_positions_deg(
            targets,
            kp=self.tuning.safe_idle_kp,
            kd=self.tuning.safe_idle_kd,
            torque=0.0,
        )

    def hold_all_once(self):
        targets = self.runtime.get_idle_hold_targets()
        states = self.runtime.get_all_state_copies()
        for mid in self.runtime.motor_ids:
            st = states[mid]
            kp = self.tuning.homed_hold_kp if st.homed else self.tuning.hold_kp
            kd = self.tuning.homed_hold_kd if st.homed else self.tuning.hold_kd
            self.runtime.send_position_deg(mid, targets[mid], kp=kp, kd=kd, torque=0.0)

    def run_all_phases(self, stop_event: Optional[threading.Event] = None):
        if stop_event is None:
            stop_event = threading.Event()

        self.verify_periodic_feedback_and_capture_boot_holds()

        for _ in range(int(self.tuning.loop_hz)):
            self.send_idle_hold_once()
            time.sleep(self.tuning.loop_dt)

        if self.phase1_config:
            self._run_phase(
                sequence_ids=[cfg.motor_id for cfg in self.phase1_config],
                hold_at_nudge_ids=set(),
                phase_name="PHASE1",
                stop_event=stop_event,
            )

        if self.phase2_config:
            self._run_phase(
                sequence_ids=[cfg.motor_id for cfg in self.phase2_config],
                hold_at_nudge_ids=set(),
                phase_name="PHASE2",
                stop_event=stop_event,
            )

    def _status_line(self, sequence_ids: List[int], active_id: int, active_symbol: str) -> str:
        states = self.runtime.get_all_state_copies()
        parts = []
        for mid in sequence_ids:
            st = states[mid]
            if mid == active_id:
                symbol = active_symbol
            elif st.homed:
                symbol = "✅"
            else:
                symbol = "⏸️"
            parts.append(f"{symbol}M{mid}:{st.position_deg:.1f}°|{st.current_A:+.2f}A")
        return "  [" + self.runtime.channel + "] " + "  ".join(parts)

    def _refresh_or_raise(self):
        self.runtime.refresh_watchdogs(
            feedback_timeout_s=self.tuning.feedback_timeout_s,
            bus_silence_timeout_s=self.tuning.bus_silence_timeout_s,
            fault_on_missing_feedback=True,
        )
        if self.runtime.faulted:
            raise RuntimeError(self.runtime.get_fault_summary())

    def _wait_for_zero_feedback(self, motor_id: int, pre_zero_pos_deg: float):
        st = self.runtime.get_state_copy(motor_id)
        last_count = st.feedback_count
        last_new_feedback_time = time.time()
        last_log = time.time()

        while True:
            self._refresh_or_raise()
            st = self.runtime.get_state_copy(motor_id)
            now = time.time()

            if st.feedback_count > last_count:
                last_count = st.feedback_count
                last_new_feedback_time = now

                if abs(st.position_deg) <= self.tuning.zero_position_tolerance_deg:
                    return

                if abs(pre_zero_pos_deg) <= 5.0 and abs(st.position_deg) <= (self.tuning.zero_position_tolerance_deg + 2.0):
                    return

            if (now - last_new_feedback_time) > self.tuning.feedback_timeout_s:
                raise RuntimeError(
                    f"[{self.runtime.channel}] M{motor_id}: zero feedback stale for "
                    f"{now - last_new_feedback_time:.3f}s"
                )

            if (now - last_log) >= self.tuning.wait_log_period_s:
                last_log = now
                self._log(
                    f"  [{self.runtime.channel}] M{motor_id} zero wait: "
                    f"pos={st.position_deg:+.1f}° age={st.feedback_age*1000:.1f}ms "
                    f"count={st.feedback_count}"
                )

            time.sleep(self.tuning.loop_dt)

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _profile_step(
        self,
        profile: MotionProfileState,
        goal_deg: float,
        vmax_deg_s: float,
        amax_deg_s2: float,
    ) -> MotionProfileState:
        dt = self.tuning.loop_dt
        err = goal_deg - profile.cmd_deg

        if abs(err) < 1e-6 and abs(profile.vel_deg_s) < 1e-6:
            return MotionProfileState(goal_deg, 0.0)

        direction = 1.0 if err >= 0.0 else -1.0
        brake_speed = math.sqrt(max(0.0, 2.0 * amax_deg_s2 * abs(err)))
        desired_vel = direction * min(vmax_deg_s, brake_speed)

        dv_max = amax_deg_s2 * dt
        new_vel = profile.vel_deg_s + self._clamp(desired_vel - profile.vel_deg_s, -dv_max, dv_max)
        new_cmd = profile.cmd_deg + new_vel * dt

        if (goal_deg - profile.cmd_deg) * (goal_deg - new_cmd) <= 0.0:
            new_cmd = goal_deg
            new_vel = 0.0

        return MotionProfileState(new_cmd, new_vel)

    def _send_mixed_targets(self, active_id: int, active_cmd_deg: float):
        hold_targets = self.runtime.get_idle_hold_targets()
        states = self.runtime.get_all_state_copies()

        for mid in self.runtime.motor_ids:
            if mid == active_id:
                self.runtime.send_position_deg(
                    mid,
                    active_cmd_deg,
                    kp=self.tuning.active_kp,
                    kd=self.tuning.active_kd,
                    torque=0.0,
                )
            else:
                st = states[mid]
                kp = self.tuning.homed_hold_kp if st.homed else self.tuning.hold_kp
                kd = self.tuning.homed_hold_kd if st.homed else self.tuning.hold_kd
                self.runtime.send_position_deg(
                    mid,
                    hold_targets[mid],
                    kp=kp,
                    kd=kd,
                    torque=0.0,
                )

    def _run_nudge(
        self,
        cfg: HomingMotorConfig,
        sequence_ids: List[int],
        phase_name: str,
        stop_event: threading.Event,
    ):
        _ = phase_name
        motor_id = cfg.motor_id
        nudge_target_deg = cfg.nudge_back_deg

        st0 = self.runtime.get_state_copy(motor_id)
        profile = MotionProfileState(cmd_deg=st0.position_deg, vel_deg_s=0.0)

        self._log(f"  ↩️ [{self.runtime.channel}] M{motor_id} nudge: 0.0° -> {nudge_target_deg:+.1f}°")

        start = time.time()
        last_status = 0.0
        at_target_since = None
        last_wait_log = time.time()

        while True:
            if stop_event.is_set():
                raise RuntimeError(f"[{self.runtime.channel}] stop requested")

            self._refresh_or_raise()

            profile = self._profile_step(
                profile,
                goal_deg=nudge_target_deg,
                vmax_deg_s=self.tuning.nudge_max_vel_deg_s,
                amax_deg_s2=self.tuning.nudge_max_acc_deg_s2,
            )
            self._send_mixed_targets(motor_id, profile.cmd_deg)

            now = time.time()
            st = self.runtime.get_state_copy(motor_id)

            if (now - last_status) >= self.tuning.status_period_s:
                last_status = now
                self._log(self._status_line(sequence_ids, motor_id, "↩️"))

            pos_err = abs(st.position_deg - nudge_target_deg)
            vel_ok = abs(st.velocity_raw) <= self.tuning.trigger_velocity_raw_max
            time_ok = (now - start) >= self.tuning.min_move_time_s

            if pos_err <= self.tuning.nudge_position_tolerance_deg and vel_ok:
                if at_target_since is None:
                    at_target_since = now
                elif time_ok and (now - at_target_since) >= self.tuning.nudge_settle_time_s:
                    self.runtime.mark_homed(motor_id, nudge_target_deg)
                    self._log(f"  ↩️ [{self.runtime.channel}] M{motor_id} nudge done -> holding at {nudge_target_deg:+.1f}°")
                    return
            else:
                at_target_since = None

            if (now - last_wait_log) >= self.tuning.wait_log_period_s:
                last_wait_log = now
                self._log(
                    f"  [{self.runtime.channel}] M{motor_id} nudge waiting: "
                    f"pos={st.position_deg:+.1f}° target={nudge_target_deg:+.1f}° "
                    f"err={pos_err:.2f}° cur={st.current_A:+.2f}A "
                    f"fcur={st.filtered_current_A:+.2f}A vel={st.velocity_raw:+.2f}"
                )

            time.sleep(self.tuning.loop_dt)
    
    def _run_phase(
        self,
        sequence_ids: List[int],
        hold_at_nudge_ids: Set[int],
        phase_name: str,
        stop_event: threading.Event,
    ):
        _ = hold_at_nudge_ids

        for motor_id in sequence_ids:
            cfg = self.cfg_by_id[motor_id]
            st0 = self.runtime.get_state_copy(motor_id)

            phase_start_deg = st0.position_deg
            active_target_deg = phase_start_deg + cfg.target_deg
            profile = MotionProfileState(cmd_deg=phase_start_deg, vel_deg_s=0.0)

            self._log(
                f"\n[{self.runtime.channel}] {phase_name} ▶ M{motor_id} "
                f"rel {cfg.target_deg:+.1f}° from {phase_start_deg:+.1f}° -> {active_target_deg:+.1f}°"
            )

            last_status = 0.0
            last_wait_log = time.time()
            target_settled_since = None
            current_above_since = None
            stall_since = None

            continuous_search_started = False
            contact_latched = False
            latched_contact_pos_deg = None
            latched_target_deg = None

            search_sign = 0.0
            if cfg.search_direction_deg > 0.0:
                search_sign = 1.0
            elif cfg.search_direction_deg < 0.0:
                search_sign = -1.0

            while True:
                if stop_event.is_set():
                    raise RuntimeError(f"[{self.runtime.channel}] stop requested")

                self._refresh_or_raise()
                now = time.time()

                profile = self._profile_step(
                    profile,
                    goal_deg=active_target_deg,
                    vmax_deg_s=self.tuning.search_max_vel_deg_s,
                    amax_deg_s2=self.tuning.search_max_acc_deg_s2,
                )
                self._send_mixed_targets(motor_id, profile.cmd_deg)

                st = self.runtime.get_state_copy(motor_id)

                if (now - last_status) >= self.tuning.status_period_s:
                    last_status = now
                    self._log(self._status_line(sequence_ids, motor_id, "🚀"))

                travel_deg = abs(st.position_deg - phase_start_deg)
                if travel_deg > self.tuning.max_search_travel_deg:
                    raise RuntimeError(
                        f"[{self.runtime.channel}] M{motor_id}: search travel exceeded "
                        f"{self.tuning.max_search_travel_deg:.1f}°"
                    )

                current_abs = abs(st.filtered_current_A)
                current_ok = current_abs >= cfg.trigger_current_A
                vel_abs = abs(st.velocity_raw)
                velocity_ok = vel_abs <= self.tuning.trigger_velocity_raw_max
                stalled_ok = vel_abs <= self.tuning.contact_stall_vel_max

                if not contact_latched and current_ok:
                    contact_latched = True
                    latched_contact_pos_deg = st.position_deg
                    latched_target_deg = st.position_deg + search_sign * self.tuning.contact_hold_offset_deg
                    active_target_deg = latched_target_deg
                    profile = MotionProfileState(cmd_deg=st.position_deg, vel_deg_s=0.0)
                    current_above_since = now
                    stall_since = now if stalled_ok else None
                    self._log(
                        f"  [{self.runtime.channel}] M{motor_id} contact latched: "
                        f"freeze target at {active_target_deg:+.1f}° "
                        f"(fcur={st.filtered_current_A:+.2f}A vel={st.velocity_raw:+.2f})"
                    )

                elif contact_latched:
                    active_target_deg = latched_target_deg

                    if current_ok:
                        if current_above_since is None:
                            current_above_since = now
                    else:
                        current_above_since = None

                    if stalled_ok:
                        if stall_since is None:
                            stall_since = now
                    else:
                        stall_since = None

                    if current_above_since is not None and (now - current_above_since) >= self.tuning.trigger_confirm_s:
                        self._log(
                            f"  ✅ [{self.runtime.channel}] M{motor_id}: "
                            f"{abs(st.filtered_current_A):.2f}A >= {cfg.trigger_current_A:.2f}A "
                            f"-> zero at {st.position_deg:+.1f}°"
                        )
                        pre_zero_pos_deg = st.position_deg
                        self.runtime.zero_motor(motor_id, permanent=False)
                        self._wait_for_zero_feedback(motor_id, pre_zero_pos_deg)
                        self._run_nudge(cfg, sequence_ids, phase_name, stop_event)
                        break

                    if stall_since is not None and (now - stall_since) >= self.tuning.trigger_confirm_s:
                        self._log(
                            f"  ✅ [{self.runtime.channel}] M{motor_id}: "
                            f"stalled at hard stop (vel={st.velocity_raw:+.2f}) "
                            f"-> zero at {st.position_deg:+.1f}°"
                        )
                        pre_zero_pos_deg = st.position_deg
                        self.runtime.zero_motor(motor_id, permanent=False)
                        self._wait_for_zero_feedback(motor_id, pre_zero_pos_deg)
                        self._run_nudge(cfg, sequence_ids, phase_name, stop_event)
                        break

                    if (
                        latched_contact_pos_deg is not None and
                        abs(st.position_deg - latched_contact_pos_deg) >= self.tuning.contact_release_move_deg and
                        not current_ok
                    ):
                        contact_latched = False
                        latched_contact_pos_deg = None
                        latched_target_deg = None
                        current_above_since = None
                        stall_since = None

                if not continuous_search_started:
                    near_target = abs(st.position_deg - active_target_deg) <= self.tuning.advance_target_tolerance_deg
                    if near_target and velocity_ok:
                        if target_settled_since is None:
                            target_settled_since = now
                        elif (now - target_settled_since) >= self.tuning.advance_settle_time_s:
                            continuous_search_started = True
                            target_settled_since = None
                    else:
                        target_settled_since = None
                else:
                    if not contact_latched:
                        active_target_deg += (
                            search_sign
                            * self.tuning.continuous_search_vel_deg_s
                            * self.tuning.loop_dt
                        )

                if (now - last_wait_log) >= self.tuning.wait_log_period_s:
                    last_wait_log = now
                    self._log(
                        f"  [{self.runtime.channel}] M{motor_id} search waiting: "
                        f"pos={st.position_deg:+.1f}° target={active_target_deg:+.1f}° "
                        f"travel={travel_deg:.1f}° cur={st.current_A:+.2f}A "
                        f"fcur={st.filtered_current_A:+.2f}A vel={st.velocity_raw:+.2f} "
                        f"latched={contact_latched}"
                    )

                time.sleep(self.tuning.loop_dt)
