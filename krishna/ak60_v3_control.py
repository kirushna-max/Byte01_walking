#!/usr/bin/env python3

import math
import struct
from typing import Optional, Dict, Any

import can


class AK60V3Motor:
    """
    AK60-6 V3.0 Motor Controller for MIT Mode
    Uses 29-bit Extended CAN frames

    TX-focused class for shared-bus systems.
    For robust RX in multi-motor setups, use the bus-level listener
    in can_runtime.py.
    """

    MODE_DUTY_CYCLE = 0x00
    MODE_CURRENT = 0x01
    MODE_CURRENT_BRAKE = 0x02
    MODE_VELOCITY = 0x03
    MODE_POSITION = 0x04
    MODE_SET_ORIGIN = 0x05
    MODE_POSITION_VELOCITY = 0x06
    MODE_MIT = 0x08

    FEEDBACK_MODE_ID = 0x29

    P_MIN = -12.56
    P_MAX = 12.56
    V_MIN = -60.0
    V_MAX = 60.0
    KP_MIN = 0.0
    KP_MAX = 500.0
    KD_MIN = 0.0
    KD_MAX = 5.0
    T_MIN = -12.0
    T_MAX = 12.0

    def __init__(self, motor_id, can_interface='can0', bitrate=1000000, bus=None):
        self.motor_id = int(motor_id)
        self.can_interface = can_interface
        self.bitrate = bitrate

        self.position = 0.0
        self.velocity = 0.0
        self.current = 0.0
        self.torque = 0.0
        self.temperature = 0
        self.error = 0

        if bus is None:
            self.bus = can.ThreadSafeBus(
                channel=can_interface,
                interface='socketcan',
                bitrate=bitrate,
                ignore_config=True,
            )
            self.owns_bus = True
        else:
            self.bus = bus
            self.owns_bus = False

    def _construct_can_id(self, control_mode: int) -> int:
        return (control_mode << 8) | self.motor_id

    @staticmethod
    def rx_motor_id(msg: can.Message) -> int:
        return msg.arbitration_id & 0xFF

    @staticmethod
    def rx_control_mode(msg: can.Message) -> int:
        return (msg.arbitration_id >> 8) & 0x1FFFFF

    @staticmethod
    def is_feedback_frame(msg: can.Message) -> bool:
        if msg is None:
            return False
        if not getattr(msg, "is_extended_id", False):
            return False
        if len(msg.data) != 8:
            return False
        if AK60V3Motor.rx_control_mode(msg) != AK60V3Motor.FEEDBACK_MODE_ID:
            return False
        return True

    @staticmethod
    def decode_feedback_message(msg: can.Message) -> Optional[Dict[str, Any]]:
        if not AK60V3Motor.is_feedback_frame(msg):
            return None

        pos_int = struct.unpack('>h', bytes(msg.data[0:2]))[0]
        vel_int = struct.unpack('>h', bytes(msg.data[2:4]))[0]
        cur_int = struct.unpack('>h', bytes(msg.data[4:6]))[0]
        temp = struct.unpack('b', bytes(msg.data[6:7]))[0]
        err = int(msg.data[7])

        position_deg = pos_int * 0.1
        position_rad = math.radians(position_deg)
        velocity_raw = float(vel_int)
        current_A = round(cur_int * 0.01, 3)

        return {
            "position_rad": position_rad,
            "position_deg": position_deg,
            "velocity_raw": velocity_raw,
            "current_A": current_A,
            "torque": current_A,
            "temperature_C": temp,
            "error_code": err,
        }

    def _constrain(self, value, min_val, max_val):
        return max(min_val, min(max_val, value))

    def _float_to_uint(self, x, x_min, x_max, bits):
        span = x_max - x_min
        offset = x - x_min
        return int((offset * ((1 << bits) - 1)) / span)

    def set_temporary_origin(self):
        can_id = self._construct_can_id(self.MODE_SET_ORIGIN)
        msg = can.Message(
            arbitration_id=can_id,
            data=[0x00],
            is_extended_id=True
        )
        self.bus.send(msg)

    def set_permanent_origin(self):
        can_id = self._construct_can_id(self.MODE_SET_ORIGIN)
        msg = can.Message(
            arbitration_id=can_id,
            data=[0x01],
            is_extended_id=True
        )
        self.bus.send(msg)

    def set_zero_position(self, permanent=False):
        if permanent:
            self.set_permanent_origin()
        else:
            self.set_temporary_origin()

    def send_mit_command(self, position=0.0, velocity=0.0, kp=0.0, kd=0.0, torque=0.0):
        position = self._constrain(position, self.P_MIN, self.P_MAX)
        velocity = self._constrain(velocity, self.V_MIN, self.V_MAX)
        kp = self._constrain(kp, self.KP_MIN, self.KP_MAX)
        kd = self._constrain(kd, self.KD_MIN, self.KD_MAX)
        torque = self._constrain(torque, self.T_MIN, self.T_MAX)

        p_int = self._float_to_uint(position, self.P_MIN, self.P_MAX, 16)
        v_int = self._float_to_uint(velocity, self.V_MIN, self.V_MAX, 12)
        kp_int = self._float_to_uint(kp, self.KP_MIN, self.KP_MAX, 12)
        kd_int = self._float_to_uint(kd, self.KD_MIN, self.KD_MAX, 12)
        t_int = self._float_to_uint(torque, self.T_MIN, self.T_MAX, 12)

        data = [
            (kp_int >> 4) & 0xFF,
            ((kp_int & 0x0F) << 4) | ((kd_int >> 8) & 0x0F),
            kd_int & 0xFF,
            (p_int >> 8) & 0xFF,
            p_int & 0xFF,
            (v_int >> 4) & 0xFF,
            ((v_int & 0x0F) << 4) | ((t_int >> 8) & 0x0F),
            t_int & 0xFF,
        ]

        can_id = self._construct_can_id(self.MODE_MIT)
        msg = can.Message(
            arbitration_id=can_id,
            data=data,
            is_extended_id=True
        )
        self.bus.send(msg)

    def send_current_brake(self, current):
        current = self._constrain(current, 0.0, 60.0)
        data = struct.pack('>i', int(current * 1000.0))

        can_id = self._construct_can_id(self.MODE_CURRENT_BRAKE)
        msg = can.Message(
            arbitration_id=can_id,
            data=data,
            is_extended_id=True
        )
        self.bus.send(msg)

    def disable(self):
        self.send_current_brake(0.0)

    def read_feedback(self, timeout=0.1):
        """
        Compatibility helper for single-motor tests only.
        Do not use this in shared-bus multi-motor runtime logic.
        """
        msg = self.bus.recv(timeout=timeout)
        if msg is None:
            return False
        if self.rx_motor_id(msg) != self.motor_id:
            return False

        decoded = self.decode_feedback_message(msg)
        if decoded is None:
            return False

        self.position = decoded["position_rad"]
        self.velocity = decoded["velocity_raw"]
        self.current = decoded["current_A"]
        self.torque = decoded["torque"]
        self.temperature = decoded["temperature_C"]
        self.error = decoded["error_code"]
        return True

    def get_feedback(self):
        return {
            "position": self.position,
            "velocity": self.velocity,
            "torque": self.torque,
            "current": self.current,
            "temperature": self.temperature,
            "error": self.error,
        }

    def close(self):
        if self.owns_bus:
            self.bus.shutdown()
