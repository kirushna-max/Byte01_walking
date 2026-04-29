#!/usr/bin/env python3
"""run_robot.py — Monolithic Kutta quadruped control script.

Task 1 – Motor homing via CAN bus (BusRuntime + BusHomingController).
Task 2 – Policy joint targets bridged from sim → real hardware via socket.
Task 3 – MuJoCo simulation launched simultaneously (keyboard WASD/arrows).

Run from the rl_mjlab/ directory:
    python run_robot.py --checkpoint logs/rsl_rl/<exp>/<run>/model_XXX.pt

Dependencies (all local or already installed):
    Local files  : can_runtime.py, homing_controller.py, ak60_v3_control.py
    pip packages : python-can, torch, mujoco, pygame (for optional PS5 fallback)
    mjlab        : installed in .pyenv (mjlab package)
"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────────
import argparse
import glob
import json
import math
import multiprocessing as mp
import pickle
import queue
import socket
import sys
import threading
import time
from dataclasses import asdict
from multiprocessing.queues import Queue as MpQueue
from multiprocessing.synchronize import Event as MpEvent
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── CAN / motor control (local files in rl_mjlab/) ────────────────────────────
from can_runtime import BusRuntime
from homing_controller import BusHomingController, ControlTuning, HomingMotorConfig

# ── mjlab simulation stack ────────────────────────────────────────────────────
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

TASK_ID = "Kutta-Flat"

# ── Socket (bridge: sim → hardware) ──────────────────────────────────────────
LIVE_SOCKET_HOST = "127.0.0.1"
LIVE_SOCKET_PORT = 50000
LIVE_QUEUE_MAXSIZE = 1

# ── Motor config JSON ─────────────────────────────────────────────────────────
MOTOR_CONFIG_PATH = "motor_config.json"

# ── Leg / motor ordering ──────────────────────────────────────────────────────
# Socket payload key order must match the socket_listener_process expectation.
# Mapping: fl = motors 1-3 (revolute1/2/3 = FL hip/thigh/knee)
#          bl = motors 4-6 (revolute4/5/6 = RL hip/thigh/knee)
#          fr = motors 7-9 (revolute7/8/9 = FR hip/thigh/knee)
#          br = motors 10-12 (revolute10/11/12 = RR hip/thigh/knee)
LEG_ORDER: Tuple[str, ...] = ("fl", "bl", "fr", "br")
LegPayload = Dict[str, List[float]]
MotorCommandConfig = Dict[int, Dict[str, Any]]

# ── Live-control safety cap ────────────────────────────────────────────────────
MAX_LIVE_DEG_PER_S = 50.0

# ── Homing flag ───────────────────────────────────────────────────────────────
CALIBRATION_REQUIRED = True   # Set False to skip homing (sim-only testing)

# ── Current / temperature logging ─────────────────────────────────────────────
CURRENT_LOG_PATH = "/home/byte/ak60_motor_control/multiprocess_code/all_4_legs/cur_test/motor_currents.csv"
CURRENT_LOG_HZ   = 5.0

TEMP_LOG_PATH    = "/home/byte/ak60_motor_control/multiprocess_code/all_4_legs/cur_test/motor_temps.csv"
TEMP_LOG_HZ      = 5.0

# ── CAN bus / homing configuration ───────────────────────────────────────────
CAN_CONFIG: Dict[str, Dict[str, List[HomingMotorConfig]]] = {
    "can0": {
        "phase1": [
            HomingMotorConfig(5, -60.0, -1.0, 4.0, 65.0),
            HomingMotorConfig(6,  10.0,  1.0, 5.0, -38.0),
            HomingMotorConfig(4,  60.0,  1.0, 4.0, -60.0),
        ],
        "phase2": [
            HomingMotorConfig(2, -60.0, -1.0, 4.0, 65.0),
            HomingMotorConfig(3,  10.0,  1.0, 4.5, -38.0),
            HomingMotorConfig(1,  60.0,  1.0, 4.0, -60.0),
        ],
    },
    "can1": {
        "phase1": [
            HomingMotorConfig(8,   60.0,  1.0, 4.0, -65.0),
            HomingMotorConfig(9,  -10.0, -1.0, 5.0,  38.0),
            HomingMotorConfig(7,  -60.0, -1.0, 4.0,  60.0),
        ],
        "phase2": [
            HomingMotorConfig(11,  60.0,  1.0, 4.0, -65.0),
            HomingMotorConfig(12, -10.0, -1.0, 5.0,  38.0),
            HomingMotorConfig(10, -60.0, -1.0, 4.0,  60.0),
        ],
    },
}

# ── Keyboard velocity magnitudes (m/s and rad/s) ──────────────────────────────
KBD_LIN_VEL_X = 1.0   # W / S
KBD_LIN_VEL_Y = 0.6   # arrow left / right (strafe)
KBD_ANG_VEL_Z = 1.0   # A / D

# ── MuJoCo key codes (ASCII) ──────────────────────────────────────────────────
KEY_W     = ord('w')
KEY_A     = ord('a')
KEY_S     = ord('s')
KEY_D     = ord('d')
KEY_SPACE = ord(' ')
KEY_ENTER = 13
KEY_MINUS = ord('-')
KEY_EQUAL = ord('=')
# Arrow keys via GLFW keycodes (not ASCII)
GLFW_KEY_UP    = 265
GLFW_KEY_DOWN  = 264
GLFW_KEY_LEFT  = 263
GLFW_KEY_RIGHT = 262

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CAN / HOMING HELPERS  (verbatim from provided homing script)
# ═══════════════════════════════════════════════════════════════════════════════

def collect_motor_ids(bus_cfg: Dict[str, List[HomingMotorConfig]]) -> List[int]:
    ids = []
    for phase_name in ("phase1", "phase2"):
        ids.extend(cfg.motor_id for cfg in bus_cfg[phase_name])
    return sorted(set(ids))


def hold_until_both_phase1_done(
    controller: BusHomingController,
    my_done_event: threading.Event,
    other_done_event: threading.Event,
    stop_event: threading.Event,
) -> bool:
    my_done_event.set()
    while not stop_event.is_set():
        if other_done_event.is_set():
            return True
        controller.hold_all_once()
        time.sleep(controller.tuning.loop_dt)
    return False


def hold_until_takeover(
    controller: BusHomingController,
    takeover_event: threading.Event,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set() and not takeover_event.is_set():
        controller.hold_all_once()
        time.sleep(controller.tuning.loop_dt)


def controller_thread_entry(
    controller: BusHomingController,
    my_phase1_done: threading.Event,
    other_phase1_done: threading.Event,
    my_phase2_done: threading.Event,
    takeover_event: threading.Event,
    stop_event: threading.Event,
    errors: List[str],
) -> None:
    channel = controller.runtime.channel
    try:
        if controller.phase1_config:
            controller._run_phase(
                sequence_ids=[cfg.motor_id for cfg in controller.phase1_config],
                hold_at_nudge_ids=set(),
                phase_name="PHASE1",
                stop_event=stop_event,
            )

        if not hold_until_both_phase1_done(
            controller=controller,
            my_done_event=my_phase1_done,
            other_done_event=other_phase1_done,
            stop_event=stop_event,
        ):
            return

        if stop_event.is_set():
            return

        if controller.phase2_config:
            controller._run_phase(
                sequence_ids=[cfg.motor_id for cfg in controller.phase2_config],
                hold_at_nudge_ids=set(),
                phase_name="PHASE2",
                stop_event=stop_event,
            )

        my_phase2_done.set()
        hold_until_takeover(
            controller=controller,
            takeover_event=takeover_event,
            stop_event=stop_event,
        )

    except Exception as exc:
        errors.append(f"{channel}: {exc}")
        stop_event.set()
        my_phase1_done.set()
        other_phase1_done.set()
        my_phase2_done.set()
        takeover_event.set()


# ── Motor config JSON ─────────────────────────────────────────────────────────

def load_motor_command_config(path: str) -> MotorCommandConfig:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    motors = raw.get("motors")
    if not isinstance(motors, dict):
        raise ValueError("motor_config.json must contain a top-level 'motors' object")
    config: MotorCommandConfig = {}
    for motor_id in range(1, 13):
        entry = motors.get(str(motor_id))
        if not isinstance(entry, dict):
            raise ValueError(f"motor_config.json missing config for motor {motor_id}")
        config[motor_id] = {
            "kp":          float(entry["kp"]),
            "kd":          float(entry["kd"]),
            "flipped":     bool(entry["flipped"]),
            "gear_ratio":  float(entry["gear_ratio"]),
        }
    return config


# ── CAN-bus logger threads ────────────────────────────────────────────────────

def runtime_for_motor_id(motor_id: int, rt0: BusRuntime, rt1: BusRuntime) -> BusRuntime:
    if 1 <= motor_id <= 6:
        return rt0
    if 7 <= motor_id <= 12:
        return rt1
    raise ValueError(f"invalid motor_id {motor_id}")


def current_logger_thread_entry(
    rt0: BusRuntime,
    rt1: BusRuntime,
    stop_event: threading.Event,
    log_path: str = CURRENT_LOG_PATH,
    log_hz: float = CURRENT_LOG_HZ,
) -> None:
    log_dt    = 1.0 / log_hz
    motor_ids = list(range(1, 13))
    try:
        with open(log_path, "a", buffering=1, encoding="utf-8") as fh:
            fh.seek(0, 2)
            if fh.tell() == 0:
                fh.write("timestamp_s," + ",".join(f"m{i}_a" for i in motor_ids) + "\n")
            print(f"[CurrentLogger] {log_hz:.0f} Hz → {log_path}", flush=True)
            while not stop_event.is_set():
                t_start = time.monotonic()
                ts      = time.time()
                currents: List[float] = []
                for mid in motor_ids:
                    try:
                        currents.append(round(runtime_for_motor_id(mid, rt0, rt1).get_state_copy(mid).current_A, 4))
                    except Exception:
                        currents.append(float("nan"))
                fh.write(f"{ts:.4f}," + ",".join(str(c) for c in currents) + "\n")
                elapsed = time.monotonic() - t_start
                if log_dt - elapsed > 0:
                    time.sleep(log_dt - elapsed)
    except Exception as exc:
        print(f"[CurrentLogger] Fatal: {exc}", flush=True)


def temp_logger_thread_entry(
    rt0: BusRuntime,
    rt1: BusRuntime,
    stop_event: threading.Event,
    log_path: str = TEMP_LOG_PATH,
    log_hz: float = TEMP_LOG_HZ,
) -> None:
    log_dt    = 1.0 / log_hz
    motor_ids = list(range(1, 13))
    try:
        with open(log_path, "a", buffering=1, encoding="utf-8") as fh:
            fh.seek(0, 2)
            if fh.tell() == 0:
                fh.write("timestamp_s," + ",".join(f"m{i}_c" for i in motor_ids) + "\n")
            print(f"[TempLogger] {log_hz:.0f} Hz → {log_path}", flush=True)
            while not stop_event.is_set():
                t_start = time.monotonic()
                ts      = time.time()
                temps: List[float] = []
                for mid in motor_ids:
                    try:
                        temps.append(round(runtime_for_motor_id(mid, rt0, rt1).get_state_copy(mid).temperature_C, 4))
                    except Exception:
                        temps.append(float("nan"))
                fh.write(f"{ts:.4f}," + ",".join(str(t) for t in temps) + "\n")
                elapsed = time.monotonic() - t_start
                if log_dt - elapsed > 0:
                    time.sleep(log_dt - elapsed)
    except Exception as exc:
        print(f"[TempLogger] Fatal: {exc}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SOCKET LISTENER PROCESS  (receives packets, runs in separate process)
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_leg_payload(payload: Any) -> LegPayload:
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict with keys: fl, bl, fr, br")
    normalized: LegPayload = {}
    for leg in LEG_ORDER:
        if leg not in payload:
            raise ValueError(f"missing leg '{leg}' in payload")
        values = payload[leg]
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError(f"payload['{leg}'] must be a list/tuple of 3 floats")
        normalized[leg] = [float(v) for v in values]
    return normalized


def socket_listener_process(
    dest_queue: MpQueue,
    stop_event: MpEvent,
    port: int = LIVE_SOCKET_PORT,
    host: str = LIVE_SOCKET_HOST,
) -> None:
    """Separate process: accept TCP connections, deserialise pickle payloads,
    keep only the freshest packet in dest_queue."""
    print(f"[Socket PID {mp.current_process().pid}] Listening on {host}:{port}", flush=True)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.settimeout(0.2)
    try:
        server.bind((host, port))
        server.listen(5)
        while not stop_event.is_set():
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue
            except OSError:
                if stop_event.is_set():
                    break
                raise
            with conn:
                conn.settimeout(0.2)
                data = bytearray()
                while not stop_event.is_set():
                    try:
                        chunk = conn.recv(4096)
                    except socket.timeout:
                        continue
                    if not chunk:
                        break
                    data.extend(chunk)
            if not data:
                continue
            try:
                packet = _validate_leg_payload(pickle.loads(bytes(data)))
            except Exception as exc:
                print(f"[Socket] Drop invalid packet from {addr}: {exc}", flush=True)
                continue
            # evict stale packet
            while True:
                try:
                    dest_queue.get_nowait()
                except Exception:
                    break
            try:
                dest_queue.put_nowait(packet)
            except Exception:
                pass
    except Exception as exc:
        print(f"[Socket] Server error: {exc}", flush=True)
        raise
    finally:
        server.close()
        print(f"[Socket PID {mp.current_process().pid}] Shutdown", flush=True)


# ── Sender helper called from the main sim loop ───────────────────────────────

def push_sim_targets_to_queue(live_queue: MpQueue, leg_payload: LegPayload) -> None:
    """Non-blocking: discard old packet and push latest sim targets."""
    while True:
        try:
            live_queue.get_nowait()
        except Exception:
            break
    try:
        live_queue.put_nowait(leg_payload)
    except Exception:
        pass


def sim_joint_targets_to_leg_payload(
    joint_targets_deg: List[float],
) -> LegPayload:
    """Convert the 12-element joint target list (revolute1…12, degrees) to
    the fl/bl/fr/br socket payload expected by the hardware."""
    if len(joint_targets_deg) != 12:
        raise ValueError(f"Expected 12 joint targets, got {len(joint_targets_deg)}")
    # revolute1-3  → fl (motors 1-3)
    # revolute4-6  → bl (motors 4-6)
    # revolute7-9  → fr (motors 7-9)
    # revolute10-12→ br (motors 10-12)
    return {
        "fl": joint_targets_deg[0:3],
        "bl": joint_targets_deg[3:6],
        "fr": joint_targets_deg[6:9],
        "br": joint_targets_deg[9:12],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LIVE CONTROL (post-homing, sim drives hardware)
# ═══════════════════════════════════════════════════════════════════════════════

def flatten_leg_payload(payload: LegPayload) -> List[float]:
    flat: List[float] = []
    for leg in LEG_ORDER:
        flat.extend(payload[leg])
    return flat


def transform_live_angles(
    raw_angles_deg: List[float],
    motor_config: MotorCommandConfig,
) -> Dict[int, float]:
    if len(raw_angles_deg) != 12:
        raise ValueError(f"expected 12 angles, got {len(raw_angles_deg)}")
    transformed: Dict[int, float] = {}
    for motor_id, angle_deg in enumerate(raw_angles_deg, start=1):
        cfg = motor_config[motor_id]
        if bool(cfg["flipped"]):
            angle_deg = -angle_deg
        angle_deg *= float(cfg["gear_ratio"])
        transformed[motor_id] = angle_deg
    return transformed


def send_live_targets(
    rt0: BusRuntime,
    rt1: BusRuntime,
    targets_deg: Dict[int, float],
    motor_config: MotorCommandConfig,
) -> None:
    for motor_id in range(1, 13):
        cfg     = motor_config[motor_id]
        runtime = runtime_for_motor_id(motor_id, rt0, rt1)
        runtime.send_position_deg(
            motor_id,
            targets_deg[motor_id],
            kp=float(cfg["kp"]),
            kd=float(cfg["kd"]),
            torque=0.0,
        )


def drain_latest_packet(live_queue: MpQueue) -> Optional[LegPayload]:
    latest = None
    while True:
        try:
            latest = live_queue.get_nowait()
        except queue.Empty:
            break
    return latest


def limit_target_step(
    current_cmd_deg: float,
    requested_deg: float,
    max_deg_per_s: float,
    dt: float,
) -> float:
    max_step = max_deg_per_s * dt
    delta    = requested_deg - current_cmd_deg
    if delta >  max_step:
        return current_cmd_deg + max_step
    if delta < -max_step:
        return current_cmd_deg - max_step
    return requested_deg


def initialize_live_command_state(rt0: BusRuntime, rt1: BusRuntime) -> Dict[int, float]:
    live_cmd_deg: Dict[int, float] = {}
    for motor_id in range(1, 13):
        runtime = runtime_for_motor_id(motor_id, rt0, rt1)
        live_cmd_deg[motor_id] = runtime.get_state_copy(motor_id).position_deg
    return live_cmd_deg


def run_hardware_live_loop(
    ctrl0: BusHomingController,
    ctrl1: BusHomingController,
    rt0: BusRuntime,
    rt1: BusRuntime,
    tuning: ControlTuning,
    stop_event: threading.Event,
    live_queue: MpQueue,
    motor_config: MotorCommandConfig,
    test_leg: Optional[str] = None,
) -> None:
    """Runs in a background thread — reads sim targets from live_queue and
    sends them to real motors with a rate-limit and watchdog check."""
    first_seen = False
    last_targets_deg: Optional[Dict[int, float]] = None
    live_cmd_deg = initialize_live_command_state(rt0, rt1)

    print("\n[HW Loop] Waiting for first sim packet …", flush=True)

    while not stop_event.is_set():
        rt0.refresh_watchdogs(
            feedback_timeout_s=tuning.feedback_timeout_s,
            bus_silence_timeout_s=tuning.bus_silence_timeout_s,
            fault_on_missing_feedback=True,
        )
        rt1.refresh_watchdogs(
            feedback_timeout_s=tuning.feedback_timeout_s,
            bus_silence_timeout_s=tuning.bus_silence_timeout_s,
            fault_on_missing_feedback=True,
        )
        if rt0.faulted:
            raise RuntimeError(rt0.get_fault_summary())
        if rt1.faulted:
            raise RuntimeError(rt1.get_fault_summary())

        leg_packet = drain_latest_packet(live_queue)
        if leg_packet is not None:
            flat = flatten_leg_payload(leg_packet)
            last_targets_deg = transform_live_angles(flat, motor_config)
            if not first_seen:
                print("[HW Loop] ✅ First sim packet received — hardware tracking sim.", flush=True)
                first_seen = True

        if not first_seen:
            ctrl0.hold_all_once()
            ctrl1.hold_all_once()
        else:
            for motor_id in range(1, 13):
                update_target = True
                if test_leg is not None:
                    leg_idx = LEG_ORDER.index(test_leg)
                    if motor_id not in range(leg_idx * 3 + 1, leg_idx * 3 + 4):
                        update_target = False

                if update_target:
                    live_cmd_deg[motor_id] = limit_target_step(
                        current_cmd_deg=live_cmd_deg[motor_id],
                        requested_deg=last_targets_deg[motor_id],
                        max_deg_per_s=MAX_LIVE_DEG_PER_S * motor_config[motor_id]["gear_ratio"],
                        dt=tuning.loop_dt,
                    )
            send_live_targets(rt0, rt1, live_cmd_deg, motor_config)

        time.sleep(tuning.loop_dt)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — KEYBOARD-CONTROLLED MUJOCO VIEWER
# ═══════════════════════════════════════════════════════════════════════════════

class _KeyboardVelocityState:
    """Thread-safe store for the current keyboard velocity command."""

    def __init__(self) -> None:
        self._lin_x = 0.0
        self._lin_y = 0.0
        self._ang_z = 0.0
        self._lock  = threading.Lock()

    def set(self, lin_x: float, lin_y: float, ang_z: float) -> None:
        with self._lock:
            self._lin_x = lin_x
            self._lin_y = lin_y
            self._ang_z = ang_z

    @property
    def cmd(self) -> Tuple[float, float, float]:
        with self._lock:
            return self._lin_x, self._lin_y, self._ang_z


class KeyboardMujocoViewer(NativeMujocoViewer):
    """NativeMujocoViewer extended with:
    * WASD + arrow keyboard velocity injection into the twist command term.
    * Interception of policy action tensor → push to live_queue for hardware.

    Key bindings
    ------------
    W / S           – forward / backward  (lin_vel_x)
    A / D           – yaw left / right    (ang_vel_z)
    Arrow Left/Right– strafe              (lin_vel_y)
    Arrow Up/Down   – unused (reserved for speed ±)
    Space           – toggle pause
    Enter           – reset environment
    - / =           – speed down / up
    """

    def __init__(
        self,
        env,
        policy,
        live_queue: Optional[MpQueue] = None,
        **kwargs,
    ) -> None:
        self._kbd  = _KeyboardVelocityState()
        # NativeMujocoViewer accepts key_callback kwarg
        super().__init__(env, policy, key_callback=self._on_key, **kwargs)
        self._cmd_term   = None   # lazily resolved
        self._live_queue = live_queue

    # ── lazy resolver for the 'twist' command term ────────────────────────────

    def _get_cmd_term(self):
        if self._cmd_term is None:
            raw_env: ManagerBasedRlEnv = self.env.unwrapped
            try:
                self._cmd_term = raw_env.command_manager.get_term("twist")
            except Exception:
                self._cmd_term = None
        return self._cmd_term

    # ── key callback (runs on MuJoCo viewer thread → safe side-effects only) ─

    def _on_key(self, key: int) -> None:
        # Accumulate key states into a velocity vector.
        # NativeMujocoViewer already handles Space/Enter/±/etc.
        # We only need to intercept W/A/S/D and arrow keys.
        lin_x = self._kbd.cmd[0]
        lin_y = self._kbd.cmd[1]
        ang_z = self._kbd.cmd[2]

        if key == KEY_W:
            lin_x =  KBD_LIN_VEL_X
        elif key == KEY_S:
            lin_x = -KBD_LIN_VEL_X
        elif key == KEY_A:
            ang_z =  KBD_ANG_VEL_Z
        elif key == KEY_D:
            ang_z = -KBD_ANG_VEL_Z
        elif key == GLFW_KEY_LEFT:
            lin_y =  KBD_LIN_VEL_Y
        elif key == GLFW_KEY_RIGHT:
            lin_y = -KBD_LIN_VEL_Y
        elif key in (GLFW_KEY_UP, GLFW_KEY_DOWN):
            pass  # handled by NativeMujocoViewer (speed ±)
        else:
            # Any other key clears movement commands (acts as a "release all")
            lin_x = 0.0
            lin_y = 0.0
            ang_z = 0.0

        self._kbd.set(lin_x, lin_y, ang_z)

    # ── step hook: inject velocity & capture action tensor ───────────────────

    def step_simulation(self) -> None:
        if self._is_paused:
            return

        # 1. Inject keyboard velocity into the twist command tensor.
        lin_x, lin_y, ang_z = self._kbd.cmd
        cmd_term = self._get_cmd_term()
        if cmd_term is not None:
            device = self.env.unwrapped.device
            cmd_term.vel_command_b[:, 0] = lin_x
            cmd_term.vel_command_b[:, 1] = lin_y
            cmd_term.vel_command_b[:, 2] = ang_z

        # 2. Run the normal simulation step.
        super().step_simulation()

        # 3. Intercept the processed joint targets from the action manager.
        if self._live_queue is not None:
            try:
                self._forward_joint_targets()
            except Exception as exc:
                print(f"[Viewer] joint target forward failed: {exc}", flush=True)

    def _forward_joint_targets(self) -> None:
        """Read env0's processed joint position targets (radians) from the
        action manager, convert to degrees, and push to live_queue."""
        raw_env = self.env.unwrapped
        # action_manager.action is the raw policy output (pre-scaling).
        # The actual joint targets are computed inside JointPositionActionCfg
        # as:  target_rad = default_pos_rad + raw_action * scale
        # We recover the target by reading the actuator processed_actions or
        # the env's action tensor + scale/offset.
        # The most robust approach: read env.action (the unscaled policy output)
        # then recompute target = default + action * scale.
        # scale=0.25, default=0.0 (all joints zeroed in INIT_STATE).
        try:
            action_tensor = raw_env.action_manager.action  # [num_envs, 12]
        except AttributeError:
            return

        ACTION_SCALE   = 0.25   # from JointPositionActionCfg(scale=0.25)
        DEFAULT_POS_RAD = 0.0   # all joints default to 0.0

        # env0 only (hardware tracks env0)
        raw_action_12 = action_tensor[0].detach().cpu().tolist()   # list of 12
        targets_rad   = [DEFAULT_POS_RAD + a * ACTION_SCALE for a in raw_action_12]
        targets_deg   = [math.degrees(r) for r in targets_rad]

        leg_payload = sim_joint_targets_to_leg_payload(targets_deg)
        push_sim_targets_to_queue(self._live_queue, leg_payload)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CHECKPOINT SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def pick_checkpoint(log_root: Path, experiment_name: str) -> Path:
    pattern    = str(log_root / experiment_name / "**" / "*.pt")
    candidates = sorted(glob.glob(pattern, recursive=True))
    if not candidates:
        print(f"\n[ERROR] No .pt checkpoints found under: {log_root / experiment_name}")
        print("        Train a model first, or pass --checkpoint /path/to/model.pt")
        sys.exit(1)
    print("\n── Available checkpoints ─────────────────────────────────────────────")
    for i, c in enumerate(candidates):
        rel = Path(c).relative_to(log_root)
        print(f"  [{i}] {rel}")
    print("──────────────────────────────────────────────────────────────────────")
    while True:
        raw = input("Select checkpoint number (default 0): ").strip()
        if raw == "":
            idx = 0
        elif raw.isdigit() and 0 <= int(raw) < len(candidates):
            idx = int(raw)
        else:
            print(f"  Enter a number 0–{len(candidates)-1}.")
            continue
        chosen = Path(candidates[idx])
        print(f"[INFO] Selected: {chosen}")
        return chosen


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — ARGUMENT PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Kutta quadruped: home motors, run sim, drive hardware.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", "-c", metavar="PATH", default=None,
        help="Path to .pt checkpoint. Omit to pick interactively.",
    )
    p.add_argument(
        "--no-terminations", action="store_true",
        help="Disable episode terminations in the sim.",
    )
    p.add_argument(
        "--num-envs", "-n", type=int, default=1,
        help="Number of parallel environments (1 recommended for hardware sync).",
    )
    p.add_argument(
        "--device", default=None,
        help="Compute device, e.g. 'cuda:0' or 'cpu'. Auto-detected if omitted.",
    )
    p.add_argument(
        "--no-hardware", action="store_true",
        help="Skip CAN / homing — run sim only (no hardware required).",
    )
    p.add_argument(
        "--test-leg", choices=["fl", "bl", "fr", "br"], default=None,
        help="Only send live sim commands to this specific leg's motors. Other legs hold position.",
    )
    p.add_argument(
        "--homing", type=lambda x: x.lower() not in ("false", "0", "no"),
        default=True, metavar="BOOL",
        help="Run motor homing before live control. Pass 'false' to skip (e.g. --homing false).",
    )
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── parse CLI ─────────────────────────────────────────────────────────────
    args = parse_args()
    hardware_enabled = not args.no_hardware
    calibration_required = CALIBRATION_REQUIRED and args.homing

    # ── startup banner ────────────────────────────────────────────────────────
    time.sleep(1)
    print("=" * 70)
    print("🤖 Kutta Quadruped — Unified Control Script")
    print(f"   Hardware : {'ENABLED' if hardware_enabled else 'DISABLED (--no-hardware)'}")
    if hardware_enabled:
        if calibration_required:
            print("   Homing   : Phase1 + Phase2 on can0 + can1 simultaneously")
        else:
            print("   Homing   : SKIPPED (--homing false)")
    print("   Sim      : Kutta-Flat, keyboard WASD + arrows")
    print("   Keyboard : W/S=forward/back  A/D=yaw  ←/→=strafe")
    print("              Space=pause  Enter=reset  -/==speed")
    print("=" * 70)

    # ── shared state ──────────────────────────────────────────────────────────
    stop_event        = threading.Event()
    errors: List[str] = []

    # hardware process handles
    rt0 = rt1 = None
    ctrl0 = ctrl1 = None
    live_queue       : Optional[MpQueue]   = None
    live_socket_stop : Optional[MpEvent]   = None
    live_socket_proc                        = None
    hw_thread        : Optional[threading.Thread] = None

    try:
        # ══════════════════════════════════════════
        # TASK 1 — Motor homing
        # ══════════════════════════════════════════
        if hardware_enabled:
            can0_ids = collect_motor_ids(CAN_CONFIG["can0"])
            can1_ids = collect_motor_ids(CAN_CONFIG["can1"])

            rt0 = BusRuntime("can0", can0_ids, bitrate=1_000_000)
            rt1 = BusRuntime("can1", can1_ids, bitrate=1_000_000)

            tuning = ControlTuning(
                active_kp=114.0,
                active_kd=1.2,
                hold_kp=54.0,
                hold_kd=0.9,
                homed_hold_kp=84.0,
                homed_hold_kd=1.2,
                loop_hz=100.0,
                min_move_time_s=1.2,
                seconds_per_deg=0.03,
                trigger_confirm_s=0.15,
                trigger_velocity_raw_max=4.0,
                feedback_timeout_s=0.15,
                bus_silence_timeout_s=0.40,
                startup_timeout_s=5.0,
                startup_poll_s=0.01,
                zero_feedback_timeout_s=1.5,
                zero_position_tolerance_deg=5.0,
                max_search_time_s=10.0,
                max_search_travel_deg=220.0,
                status_period_s=0.5,
                safe_idle_kp=48.0,
                safe_idle_kd=0.8,
                nudge_position_tolerance_deg=2.0,
                nudge_settle_time_s=0.12,
                search_max_vel_deg_s=100.0,
                search_max_acc_deg_s2=140.0,
                continuous_search_vel_deg_s=40.0,
                nudge_max_vel_deg_s=60.0,
                nudge_max_acc_deg_s2=168.0,
                advance_target_tolerance_deg=2.0,
                advance_settle_time_s=0.20,
                stall_progress_epsilon_deg=0.25,
                stall_timeout_s=0.50,
                wait_log_period_s=1.0,
                contact_hold_offset_deg=0.5,
                contact_release_move_deg=3.0,
                contact_stall_vel_max=2.0,
            )

            print_lock = threading.Lock()

            ctrl0 = BusHomingController(
                runtime=rt0,
                phase1_config=CAN_CONFIG["can0"]["phase1"],
                phase2_config=CAN_CONFIG["can0"]["phase2"],
                tuning=tuning,
                print_lock=print_lock,
            )
            ctrl1 = BusHomingController(
                runtime=rt1,
                phase1_config=CAN_CONFIG["can1"]["phase1"],
                phase2_config=CAN_CONFIG["can1"]["phase2"],
                tuning=tuning,
                print_lock=print_lock,
            )

            print("\n📡 Waiting for periodic motor feedback …")
            ctrl0.verify_periodic_feedback_and_capture_boot_holds()
            ctrl1.verify_periodic_feedback_and_capture_boot_holds()
            print("✅ All motors streaming feedback.\n")

            print("🧷 Sending low-gain idle hold for 1 s …")
            for _ in range(int(tuning.loop_hz)):
                ctrl0.send_idle_hold_once()
                ctrl1.send_idle_hold_once()
                time.sleep(tuning.loop_dt)

            final_hold_takeover = threading.Event()

            if calibration_required:
                can0_p1 = threading.Event()
                can1_p1 = threading.Event()
                can0_p2 = threading.Event()
                can1_p2 = threading.Event()

                print("🚀 Starting homing phase 1 on both CAN buses …\n")
                t0 = threading.Thread(
                    target=controller_thread_entry,
                    args=(ctrl0, can0_p1, can1_p1, can0_p2,
                          final_hold_takeover, stop_event, errors),
                    daemon=True,
                )
                t1 = threading.Thread(
                    target=controller_thread_entry,
                    args=(ctrl1, can1_p1, can0_p1, can1_p2,
                          final_hold_takeover, stop_event, errors),
                    daemon=True,
                )
                t0.start()
                t1.start()

                while not stop_event.is_set():
                    if errors:
                        raise RuntimeError(" | ".join(errors))
                    if can0_p2.is_set() and can1_p2.is_set():
                        break
                    time.sleep(tuning.loop_dt)

                if errors:
                    raise RuntimeError(" | ".join(errors))

                for _ in range(5):
                    ctrl0.hold_all_once()
                    ctrl1.hold_all_once()
                    time.sleep(tuning.loop_dt)

                final_hold_takeover.set()
                t0.join()
                t1.join()

                if errors:
                    raise RuntimeError(" | ".join(errors))
                print("✅ Homing complete.\n")

            else:
                print("⏩ Skipping homing (CALIBRATION_REQUIRED=False).")
                final_hold_takeover.set()

            # ── Load motor command config ──────────────────────────────────
            print(f"\n📄 Loading motor config from {MOTOR_CONFIG_PATH} …")
            motor_config = load_motor_command_config(MOTOR_CONFIG_PATH)
            print("✅ Motor config loaded.")

            # ── Current / temp loggers ────────────────────────────────────
            threading.Thread(
                target=current_logger_thread_entry,
                args=(rt0, rt1, stop_event),
                daemon=True, name="CurrentLogger",
            ).start()
            threading.Thread(
                target=temp_logger_thread_entry,
                args=(rt0, rt1, stop_event),
                daemon=True, name="TempLogger",
            ).start()

            # ── Socket listener process ────────────────────────────────────
            live_queue       = mp.Queue(maxsize=LIVE_QUEUE_MAXSIZE)
            live_socket_stop = mp.Event()
            live_socket_proc = mp.Process(
                target=socket_listener_process,
                args=(live_queue, live_socket_stop, LIVE_SOCKET_PORT, LIVE_SOCKET_HOST),
                daemon=True,
            )
            live_socket_proc.start()
            print(f"📡 Socket listener process PID {live_socket_proc.pid} started.")

            # ── Hardware live-control thread ───────────────────────────────
            hw_thread = threading.Thread(
                target=run_hardware_live_loop,
                args=(ctrl0, ctrl1, rt0, rt1, tuning, stop_event,
                      live_queue, motor_config),
                kwargs={"test_leg": args.test_leg},
                daemon=True,
                name="HWLiveLoop",
            )
            hw_thread.start()

        # ══════════════════════════════════════════
        # TASK 3 — MuJoCo simulation (blocks until window closed)
        # ══════════════════════════════════════════

        # Task registration
        import mjlab.tasks   # noqa: F401
        import src.tasks     # noqa: F401

        configure_torch_backends()
        device: str = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        env_cfg   = load_env_cfg(TASK_ID, play=True)
        agent_cfg = load_rl_cfg(TASK_ID)

        if args.no_terminations:
            env_cfg.terminations = {}
            print("[INFO] Episode terminations disabled.")

        env_cfg.scene.num_envs = args.num_envs

        # Checkpoint resolution
        if args.checkpoint is not None:
            resume_path = Path(args.checkpoint)
            if not resume_path.exists():
                print(f"[ERROR] Checkpoint not found: {resume_path}")
                sys.exit(1)
            print(f"[INFO] Checkpoint: {resume_path}")
        else:
            log_root    = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
            resume_path = pick_checkpoint(log_root, agent_cfg.experiment_name)

        # Build env + policy
        env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
        runner     = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(
            str(resume_path),
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
        policy = runner.get_inference_policy(device=device)

        print(f"\n[INFO] Launching Kutta-Flat | device={device} | envs={args.num_envs}")
        print("[INFO] Close the MuJoCo window to exit.\n")

        viewer = KeyboardMujocoViewer(env, policy, live_queue=live_queue)
        try:
            viewer.run()
        finally:
            env.close()

    except KeyboardInterrupt:
        print("\n🛑 Ctrl+C — stopping …")
        stop_event.set()
    except Exception as exc:
        print(f"\n❌ Runtime error: {exc}")
        stop_event.set()
    finally:
        # ── Tear down socket process ──────────────────────────────────────
        if live_socket_stop is not None:
            live_socket_stop.set()
        if live_socket_proc is not None:
            live_socket_proc.join(timeout=1.0)
            if live_socket_proc.is_alive():
                live_socket_proc.terminate()
                live_socket_proc.join(timeout=1.0)

        stop_event.set()

        # ── Disable motors ────────────────────────────────────────────────
        print("\n🔌 Disabling motors and shutting down CAN …")
        for rt in (rt0, rt1):
            if rt is not None:
                try:
                    rt.disable_all()
                except Exception:
                    pass
        for rt in (rt0, rt1):
            if rt is not None:
                try:
                    rt.stop()
                except Exception:
                    pass
        print("✅ Done.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
