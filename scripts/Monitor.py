"""Monitor.py — Real-time monitor for RL policy inputs / outputs.

How it works
------------
Play.py is the *producer*: it writes each policy step's observation vector
and action vector into a tiny shared-memory block (via multiprocessing.shared_memory).
This script is the *consumer*: run it alongside Play.py.

Production-side hook
--------------------
Import and call ``attach_monitor_hook(policy, env)`` in play.py immediately
after the policy is created (before the viewer loop).  The hook wraps the
policy callable so that every call also pushes data into shared memory.

Alternatively, run ``python Monitor.py --demo`` to see a self-contained
animated demo with random data (no Play.py required).

Display rate : 1 Hz  (terminal print)
Log rate     : 50 Hz (CSV log, matches policy step rate)
"""

from __future__ import annotations

import argparse
import csv
import os
import struct
import sys
import time
from datetime import datetime
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Shared-memory layout constants  (must match producer side)
# ---------------------------------------------------------------------------
SHM_NAME = "rl_monitor_shm"

# Header: [sequence(uint64), obs_dim(int32), act_dim(int32), timestamp_ns(int64)]
HEADER_FMT = "=QiiQ"          # little-endian: u64, i32, i32, u64
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 24 bytes

# Maximum vector sizes (pre-allocated; actual dims come from header)
MAX_OBS = 512
MAX_ACT = 64
FLOAT32_BYTES = 4

SHM_SIZE = HEADER_SIZE + (MAX_OBS + MAX_ACT) * FLOAT32_BYTES


# ---------------------------------------------------------------------------
# Producer-side API  (imported by play.py)
# ---------------------------------------------------------------------------

class _MonitorSHM:
    """Thin wrapper around shared memory for *writing* policy I/O."""

    def __init__(self) -> None:
        try:
            self._shm = SharedMemory(name=SHM_NAME, create=True, size=SHM_SIZE)
            self._created = True
        except FileExistsError:
            self._shm = SharedMemory(name=SHM_NAME, create=False)
            self._created = False
        self._buf = self._shm.buf
        self._seq: int = 0

    def write(self, obs: np.ndarray, act: np.ndarray) -> None:
        obs_flat = obs.flatten().astype(np.float32)
        act_flat = act.flatten().astype(np.float32)
        obs_dim = min(len(obs_flat), MAX_OBS)
        act_dim = min(len(act_flat), MAX_ACT)
        ts_ns = time.time_ns()
        self._seq += 1

        header = struct.pack(HEADER_FMT, self._seq, obs_dim, act_dim, ts_ns)
        self._buf[:HEADER_SIZE] = header

        obs_start = HEADER_SIZE
        obs_end = obs_start + obs_dim * FLOAT32_BYTES
        self._buf[obs_start:obs_end] = obs_flat[:obs_dim].tobytes()

        act_start = obs_start + MAX_OBS * FLOAT32_BYTES
        act_end = act_start + act_dim * FLOAT32_BYTES
        self._buf[act_start:act_end] = act_flat[:act_dim].tobytes()

    def close(self) -> None:
        self._buf.release()
        self._shm.close()
        if self._created:
            self._shm.unlink()


_shm_writer: _MonitorSHM | None = None


def attach_monitor_hook(policy, env):
    """Wrap *policy* so every call also pushes obs + action to shared memory.

    Call this in play.py after the policy is created, e.g.::

        from scripts.Monitor import attach_monitor_hook
        policy = attach_monitor_hook(policy, env)

    Returns the wrapped policy (a drop-in replacement).
    """
    global _shm_writer
    _shm_writer = _MonitorSHM()
    print(f"[Monitor] Shared memory '{SHM_NAME}' opened — start Monitor.py to observe.")

    original_call = policy.__call__ if hasattr(policy, "__call__") else policy

    class _WrappedPolicy:
        def __call__(self, obs):
            action = original_call(obs)
            try:
                # obs may be a dict {"actor": tensor} or a plain tensor
                if isinstance(obs, dict):
                    obs_arr = obs.get("actor", next(iter(obs.values())))
                else:
                    obs_arr = obs
                obs_np = obs_arr[0].detach().cpu().numpy() if hasattr(obs_arr, "detach") else np.asarray(obs_arr[0])
                act_np = action[0].detach().cpu().numpy() if hasattr(action, "detach") else np.asarray(action[0])
                _shm_writer.write(obs_np, act_np)  # type: ignore[union-attr]
            except Exception:
                pass  # Never crash the sim loop
            return action

    wrapped = _WrappedPolicy()
    # Copy over any attributes the original may have had
    return wrapped


# ---------------------------------------------------------------------------
# Consumer-side: display + logging
# ---------------------------------------------------------------------------

def _read_shm(shm: SharedMemory):
    """Return (seq, obs_dim, act_dim, ts_ns, obs_vec, act_vec) or None."""
    buf = bytes(shm.buf[:HEADER_SIZE])
    seq, obs_dim, act_dim, ts_ns = struct.unpack(HEADER_FMT, buf)
    if seq == 0 or obs_dim <= 0:
        return None
    obs_bytes = bytes(shm.buf[HEADER_SIZE: HEADER_SIZE + obs_dim * FLOAT32_BYTES])
    obs_vec = np.frombuffer(obs_bytes, dtype=np.float32).copy()
    act_start = HEADER_SIZE + MAX_OBS * FLOAT32_BYTES
    act_bytes = bytes(shm.buf[act_start: act_start + act_dim * FLOAT32_BYTES])
    act_vec = np.frombuffer(act_bytes, dtype=np.float32).copy()
    return seq, obs_dim, act_dim, ts_ns, obs_vec, act_vec


def _format_vec(label: str, vec: np.ndarray, cols: int = 8) -> str:
    lines = [f"  {label} [{len(vec)}]:"]
    for i in range(0, len(vec), cols):
        chunk = vec[i:i + cols]
        nums = "  ".join(f"{v:+8.4f}" for v in chunk)
        lines.append(f"    [{i:>3}]  {nums}")
    return "\n".join(lines)


def _clear() -> None:
    os.system("clear" if os.name == "posix" else "cls")


def run_monitor(log_dir: Path | None = None, demo: bool = False) -> None:
    # ---- open shared memory ------------------------------------------------
    shm: SharedMemory | None = None
    if not demo:
        print(f"[Monitor] Waiting for Play.py to create shared memory '{SHM_NAME}' …")
        while shm is None:
            try:
                shm = SharedMemory(name=SHM_NAME, create=False)
            except FileNotFoundError:
                time.sleep(0.5)
        print("[Monitor] Connected.")

    # ---- log file ----------------------------------------------------------
    if log_dir is None:
        log_dir = Path("logs") / "monitor"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"policy_io_{ts_str}.csv"
    log_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(log_file)
    print(f"[Monitor] Logging at 50 Hz → {log_path}")
    header_written = False  # deferred until obs/act dims are known

    # ---- timing ------------------------------------------------------------
    DISPLAY_INTERVAL = 1.0   # seconds between terminal refreshes
    LOG_INTERVAL     = 0.02  # seconds between log rows (50 Hz)

    last_display = 0.0
    last_log     = 0.0
    last_seq     = -1

    # ---- demo state --------------------------------------------------------
    demo_obs_dim = 48
    demo_act_dim = 12
    demo_seq     = 0

    try:
        while True:
            now = time.monotonic()

            # --- read data --------------------------------------------------
            if demo:
                demo_seq += 1
                obs_vec = np.random.randn(demo_obs_dim).astype(np.float32)
                act_vec = np.clip(np.random.randn(demo_act_dim).astype(np.float32), -1, 1)
                seq      = demo_seq
                obs_dim  = demo_obs_dim
                act_dim  = demo_act_dim
                ts_ns    = time.time_ns()
            else:
                assert shm is not None
                result = _read_shm(shm)
                if result is None or result[0] == last_seq:
                    time.sleep(0.0002)  # 200 µs — tight poll to not miss fast runs
                    continue
                seq, obs_dim, act_dim, ts_ns, obs_vec, act_vec = result

            last_seq = seq

            # --- 50 Hz log --------------------------------------------------
            if now - last_log >= LOG_INTERVAL:
                last_log = now
                if not header_written:
                    obs_hdrs = [f"obs_{i}" for i in range(obs_dim)]
                    act_hdrs = [f"act_{i}" for i in range(act_dim)]
                    csv_writer.writerow(["timestamp_s", "seq"] + obs_hdrs + act_hdrs)
                    header_written = True
                ts_s = ts_ns * 1e-9
                csv_writer.writerow([f"{ts_s:.6f}", seq] + obs_vec.tolist() + act_vec.tolist())
                log_file.flush()

            # --- 1 Hz display -----------------------------------------------
            if now - last_display >= DISPLAY_INTERVAL:
                last_display = now
                _clear()
                dt = datetime.now().strftime("%H:%M:%S")
                print("=" * 72)
                print(f"  RL Policy Monitor   |  {dt}  |  step #{seq}")
                print(f"  obs_dim={obs_dim}   act_dim={act_dim}   log→ {log_path.name}")
                print("=" * 72)
                print(_format_vec("INPUTS  (obs)", obs_vec))
                print()
                print(_format_vec("OUTPUTS (act)", act_vec))
                print()
                print(f"  obs  min={obs_vec.min():+.4f}  max={obs_vec.max():+.4f}"
                      f"  mean={obs_vec.mean():+.4f}  std={obs_vec.std():.4f}")
                print(f"  act  min={act_vec.min():+.4f}  max={act_vec.max():+.4f}"
                      f"  mean={act_vec.mean():+.4f}  std={act_vec.std():.4f}")
                print()
                print("  Press Ctrl-C to stop.")

            if demo:
                time.sleep(LOG_INTERVAL)

    except KeyboardInterrupt:
        print("\n[Monitor] Stopped.")
    finally:
        log_file.close()
        if shm is not None:
            shm.close()
        print(f"[Monitor] Log saved: {log_path}")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor RL policy I/O from Play.py")
    parser.add_argument("--log-dir", type=Path, default=None,
                        help="Directory to write CSV logs (default: logs/monitor/)")
    parser.add_argument("--demo", action="store_true",
                        help="Run with random data (no Play.py required)")
    args = parser.parse_args()
    run_monitor(log_dir=args.log_dir, demo=args.demo)


if __name__ == "__main__":
    main()
