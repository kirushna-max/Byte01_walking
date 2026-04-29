#!/usr/bin/env python3

import copy
import math
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional

import can

from ak60_v3_control import AK60V3Motor


class FaultCode(Enum):
    NONE = auto()
    NO_FEEDBACK = auto()
    SEND_FAIL = auto()
    BUS_ERROR = auto()
    MOTOR_ERROR = auto()
    STARTUP_TIMEOUT = auto()
    ZERO_TIMEOUT = auto()
    TRIGGER_TIMEOUT = auto()
    TRAVEL_LIMIT = auto()
    THREAD_DIED = auto()
    BARRIER_BROKEN = auto()


@dataclass
class MotorState:
    motor_id: int

    position_rad: float = 0.0
    position_deg: float = 0.0
    velocity_raw: float = 0.0
    current_A: float = 0.0
    torque: float = 0.0
    temperature_C: int = 0
    error_code: int = 0

    filtered_current_A: float = 0.0

    last_feedback_time: float = 0.0
    feedback_age: float = float("inf")
    feedback_count: int = 0
    missed_deadline_count: int = 0

    state_valid: bool = False
    startup_ready: bool = False
    homed: bool = False

    boot_hold_deg: float = 0.0
    boot_hold_captured: bool = False

    final_hold_deg: float = 0.0
    last_command_time: float = 0.0

    faulted: bool = False
    fault_code: FaultCode = FaultCode.NONE
    fault_msg: str = ""


class MotorFeedbackListener(can.Listener):
    def __init__(self, states: Dict[int, MotorState], lock: threading.RLock, rx_info: dict):
        self.states = states
        self.lock = lock
        self.rx_info = rx_info

    def on_message_received(self, msg):
        if not AK60V3Motor.is_feedback_frame(msg):
            return

        now = time.time()
        motor_id = AK60V3Motor.rx_motor_id(msg)

        with self.lock:
            self.rx_info["last_rx_time"] = now
            self.rx_info["rx_count"] += 1

            if motor_id not in self.states:
                return

            decoded = AK60V3Motor.decode_feedback_message(msg)
            if decoded is None:
                return

            st = self.states[motor_id]
            st.position_rad = decoded["position_rad"]
            st.position_deg = decoded["position_deg"]
            st.velocity_raw = decoded["velocity_raw"]
            st.current_A = decoded["current_A"]
            st.torque = decoded["torque"]
            st.temperature_C = decoded["temperature_C"]
            st.error_code = decoded["error_code"]

            alpha = 0.25
            if st.feedback_count == 0:
                st.filtered_current_A = st.current_A
            else:
                st.filtered_current_A = (1.0 - alpha) * st.filtered_current_A + alpha * st.current_A

            st.last_feedback_time = now
            st.feedback_age = 0.0
            st.feedback_count += 1
            st.state_valid = True
            st.startup_ready = True

            if not st.boot_hold_captured:
                st.boot_hold_deg = st.position_deg
                st.boot_hold_captured = True

            if st.error_code != 0:
                st.faulted = True
                st.fault_code = FaultCode.MOTOR_ERROR
                st.fault_msg = f"motor error code {st.error_code}"

    def on_error(self, exc):
        with self.lock:
            self.rx_info["listener_failed"] = True
            self.rx_info["listener_error"] = exc

    def stop(self):
        return


class BusRuntime:
    def __init__(self, channel: str, motor_ids: List[int], bitrate: int = 1_000_000):
        self.channel = channel
        self.motor_ids = sorted(set(int(mid) for mid in motor_ids))
        self.lock = threading.RLock()

        self.faulted = False
        self.fault_msg = ""

        self.states: Dict[int, MotorState] = {
            mid: MotorState(motor_id=mid) for mid in self.motor_ids
        }
        self.motors: Dict[int, AK60V3Motor] = {}

        self.rx_info = {
            "last_rx_time": 0.0,
            "rx_count": 0,
            "listener_failed": False,
            "listener_error": None,
        }

        self.bus = can.ThreadSafeBus(
            channel=channel,
            interface="socketcan",
            bitrate=bitrate,
            ignore_config=True,
        )

        filters = [
            {
                "can_id": (AK60V3Motor.FEEDBACK_MODE_ID << 8) | mid,
                "can_mask": 0x1FFFFFFF,
                "extended": True,
            }
            for mid in self.motor_ids
        ]
        try:
            self.bus.set_filters(filters)
        except Exception:
            pass

        for mid in self.motor_ids:
            self.motors[mid] = AK60V3Motor(
                motor_id=mid,
                can_interface=channel,
                bitrate=bitrate,
                bus=self.bus,
            )

        self.listener = MotorFeedbackListener(self.states, self.lock, self.rx_info)
        self.notifier = can.Notifier(self.bus, [self.listener], timeout=0.01)

    def stop(self):
        try:
            if self.notifier is not None:
                self.notifier.stop(timeout=1.0)
        finally:
            self.bus.shutdown()

    def latch_bus_fault(self, msg: str):
        with self.lock:
            self.faulted = True
            self.fault_msg = f"[{self.channel}] {msg}"

    def latch_motor_fault(self, motor_id: int, code: FaultCode, msg: str):
        with self.lock:
            st = self.states[motor_id]
            st.faulted = True
            st.state_valid = False
            st.fault_code = code
            st.fault_msg = msg
            self.faulted = True
            self.fault_msg = f"[{self.channel}] M{motor_id}: {msg}"

    def get_state_copy(self, motor_id: int) -> MotorState:
        with self.lock:
            return copy.deepcopy(self.states[motor_id])

    def get_all_state_copies(self) -> Dict[int, MotorState]:
        with self.lock:
            return copy.deepcopy(self.states)

    def capture_boot_hold_positions(self):
        with self.lock:
            for st in self.states.values():
                if st.startup_ready and not st.boot_hold_captured:
                    st.boot_hold_deg = st.position_deg
                    st.boot_hold_captured = True

    def set_final_hold_deg(self, motor_id: int, hold_deg: float):
        with self.lock:
            self.states[motor_id].final_hold_deg = hold_deg

    def mark_homed(self, motor_id: int, hold_deg: float):
        with self.lock:
            st = self.states[motor_id]
            st.homed = True
            st.final_hold_deg = hold_deg

    def zero_motor(self, motor_id: int, permanent: bool = False):
        try:
            self.motors[motor_id].set_zero_position(permanent=permanent)
        except Exception as exc:
            self.latch_motor_fault(motor_id, FaultCode.SEND_FAIL, f"zero failed: {exc}")
            raise

    def disable_motor(self, motor_id: int):
        try:
            self.motors[motor_id].disable()
        except Exception:
            pass

    def disable_all(self):
        for mid in self.motor_ids:
            self.disable_motor(mid)

    def send_position_deg(self, motor_id: int, position_deg: float, kp: float, kd: float, torque: float = 0.0):
        try:
            self.motors[motor_id].send_mit_command(
                position=math.radians(position_deg),
                velocity=0.0,
                kp=kp,
                kd=kd,
                torque=torque,
            )
            with self.lock:
                self.states[motor_id].last_command_time = time.time()
        except Exception as exc:
            self.latch_motor_fault(motor_id, FaultCode.SEND_FAIL, f"MIT send failed: {exc}")
            raise

    def command_positions_deg(self, targets_deg: Dict[int, float], kp: float, kd: float, torque: float = 0.0):
        for mid in self.motor_ids:
            target_deg = targets_deg[mid]
            self.send_position_deg(mid, target_deg, kp=kp, kd=kd, torque=torque)

    def all_startup_ready(self, ids: Optional[List[int]] = None) -> bool:
        ids = self.motor_ids if ids is None else ids
        with self.lock:
            return all(self.states[mid].startup_ready for mid in ids)

    def get_idle_hold_targets(self) -> Dict[int, float]:
        with self.lock:
            out = {}
            for mid in self.motor_ids:
                st = self.states[mid]
                out[mid] = st.final_hold_deg if st.homed else st.boot_hold_deg
            return out

    def get_fault_summary(self) -> str:
        with self.lock:
            if self.fault_msg:
                return self.fault_msg
            for mid in self.motor_ids:
                st = self.states[mid]
                if st.faulted:
                    return f"[{self.channel}] M{mid}: {st.fault_msg}"
            return f"[{self.channel}] unknown fault"

    def refresh_watchdogs(
        self,
        feedback_timeout_s: float,
        bus_silence_timeout_s: float,
        fault_on_missing_feedback: bool = True,
    ):
        now = time.time()

        with self.lock:
            if self.rx_info["listener_failed"]:
                self.faulted = True
                self.fault_msg = f"[{self.channel}] listener failed: {self.rx_info['listener_error']}"

            last_rx = self.rx_info["last_rx_time"]
            if last_rx > 0.0 and (now - last_rx) > bus_silence_timeout_s:
                self.faulted = True
                self.fault_msg = f"[{self.channel}] no RX for {now - last_rx:.3f}s"

            try:
                bus_state = getattr(self.bus, "state", None)
                if bus_state is not None:
                    state_text = str(bus_state).upper()
                    if "BUS_OFF" in state_text:
                        self.faulted = True
                        self.fault_msg = f"[{self.channel}] bus state {bus_state}"
            except Exception:
                pass

            for mid in self.motor_ids:
                st = self.states[mid]

                if st.last_feedback_time > 0.0:
                    st.feedback_age = now - st.last_feedback_time
                else:
                    st.feedback_age = float("inf")

                if st.error_code != 0:
                    st.faulted = True
                    st.fault_code = FaultCode.MOTOR_ERROR
                    st.fault_msg = f"motor error code {st.error_code}"
                    self.faulted = True
                    self.fault_msg = f"[{self.channel}] M{mid}: {st.fault_msg}"

                if st.startup_ready and st.feedback_age > feedback_timeout_s:
                    st.missed_deadline_count += 1
                    if fault_on_missing_feedback:
                        st.faulted = True
                        st.fault_code = FaultCode.NO_FEEDBACK
                        st.fault_msg = f"stale feedback age {st.feedback_age:.3f}s"
                        self.faulted = True
                        self.fault_msg = f"[{self.channel}] M{mid}: {st.fault_msg}"

    def wait_for_periodic_feedback(
        self,
        timeout_s: float,
        poll_period_s: float,
        feedback_timeout_s: float,
        bus_silence_timeout_s: float,
    ) -> bool:
        print(f"[{self.channel}] waiting for periodic feedback from motors {self.motor_ids} ...")
        start = time.time()
        announced = set()

        while (time.time() - start) < timeout_s:
            self.refresh_watchdogs(
                feedback_timeout_s=feedback_timeout_s,
                bus_silence_timeout_s=bus_silence_timeout_s,
                fault_on_missing_feedback=False,
            )
            if self.faulted:
                return False

            with self.lock:
                for mid in self.motor_ids:
                    st = self.states[mid]
                    if st.startup_ready and mid not in announced:
                        print(f"  [{self.channel}] M{mid} ready at {st.position_deg:+.1f}°")
                        announced.add(mid)

                ready = all(self.states[mid].startup_ready for mid in self.motor_ids)

            if ready:
                self.capture_boot_hold_positions()
                print(f"[{self.channel}] startup feedback verified.")
                return True

            time.sleep(poll_period_s)

        missing = []
        with self.lock:
            for mid in self.motor_ids:
                if not self.states[mid].startup_ready:
                    missing.append(mid)

        self.faulted = True
        self.fault_msg = f"[{self.channel}] startup timeout waiting for motors {missing}"
        return False
