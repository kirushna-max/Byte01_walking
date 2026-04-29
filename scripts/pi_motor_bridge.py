#!/usr/bin/env python3
"""Pi Motor Bridge — receive joint angles from Play_ps5.py and forward to motor controller.

Architecture:
  [PC: Play_ps5.py] ──TCP pickle──> [Pi: this script (port 50001)]
                                             │
                                             └──TCP pickle──> [motor controller (127.0.0.1:50000)]

Run this script on the Raspberry Pi before starting Play_ps5.py on the PC.

Usage:
  python pi_motor_bridge.py                  # listen on default port 50001
  python pi_motor_bridge.py --listen-port 50001 --motor-port 50000
"""

from __future__ import annotations

import argparse
import pickle
import socket
import time
from typing import Any, Dict, List

# ── config ────────────────────────────────────────────────────────────────────

LISTEN_HOST = "0.0.0.0"   # accept connections from any interface
LISTEN_PORT = 50001        # Play_ps5.py connects here

MOTOR_HOST  = "127.0.0.1" # local motor controller
MOTOR_PORT  = 50000

LEG_ORDER = ("fl", "bl", "fr", "br")
LegPayload = Dict[str, List[float]]


# ── validation ────────────────────────────────────────────────────────────────

def validate_leg_payload(payload: Any) -> LegPayload:
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")

    normalized: LegPayload = {}
    for leg in LEG_ORDER:
        if leg not in payload:
            raise ValueError(f"missing leg '{leg}'")

        values = payload[leg]
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"payload['{leg}'] must be a list or tuple")

        if len(values) != 3:
            raise ValueError(f"payload['{leg}'] must have exactly 3 values")

        normalized[leg] = [float(v) for v in values]

    return normalized


# ── motor sender (unchanged from original) ────────────────────────────────────

def send_leg_angles(
    payload: Any,
    host: str = MOTOR_HOST,
    port: int = MOTOR_PORT,
) -> None:
    normalized = validate_leg_payload(payload)
    data = pickle.dumps(normalized, protocol=pickle.HIGHEST_PROTOCOL)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(data)


# ── TCP receiver ──────────────────────────────────────────────────────────────

def _recv_all(conn: socket.socket, bufsize: int = 65536) -> bytes:
    """Read all available bytes from a connected socket."""
    chunks: list[bytes] = []
    while True:
        chunk = conn.recv(bufsize)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def serve(listen_host: str, listen_port: int, motor_host: str, motor_port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((listen_host, listen_port))
        srv.listen(1)
        print(f"[Bridge] Listening on {listen_host}:{listen_port}")
        print(f"[Bridge] Forwarding to motor controller at {motor_host}:{motor_port}")

        while True:
            conn, addr = srv.accept()
            with conn:
                try:
                    raw = _recv_all(conn)
                    if not raw:
                        continue

                    payload = pickle.loads(raw)  # noqa: S301 — trusted local network
                    normalized = validate_leg_payload(payload)

                    print(
                        f"[Bridge] {addr[0]} → "
                        + "  ".join(
                            f"{leg}: [{', '.join(f'{v:.1f}°' for v in normalized[leg])}]"
                            for leg in LEG_ORDER
                        )
                    )

                    send_leg_angles(normalized, host=motor_host, port=motor_port)

                except Exception as exc:
                    print(f"[Bridge] ERROR from {addr[0]}: {exc}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Receive joint angles from Play_ps5.py and forward to motor controller.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--listen-host", default=LISTEN_HOST, help="Interface to bind.")
    p.add_argument("--listen-port", type=int, default=LISTEN_PORT,
                   help="TCP port to receive angle stream from the PC.")
    p.add_argument("--motor-host", default=MOTOR_HOST,
                   help="Motor controller host.")
    p.add_argument("--motor-port", type=int, default=MOTOR_PORT,
                   help="Motor controller port.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    serve(args.listen_host, args.listen_port, args.motor_host, args.motor_port)


if __name__ == "__main__":
    main()
