#!/usr/bin/env python3
"""
DYNAMIXEL step-response tester with numeric stability metrics.

Purpose:
- Run repeatable position step tests on real motors.
- Print quantitative metrics instead of relying on visual judgement.
- Save raw logs to CSV for later analysis.

Example:
  python legged_gym/scripts/dxl_pd_step_test.py \
    --port /dev/ttyUSB0 --baud 3000000 --ids 1,2,3 \
    --step-deg 5 --hold-s 0.8 --return-s 0.8 --repeats 4 \
    --p-gain 800 --d-gain 120 --i-gain 0
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    from dynamixel_sdk import PacketHandler, PortHandler  # type: ignore
except Exception as exc:  # pragma: no cover
    print("[error] failed to import dynamixel_sdk:", exc)
    print("Install with: pip install dynamixel-sdk")
    sys.exit(1)


def parse_ids(text: str) -> List[int]:
    ids = []
    for x in text.split(","):
        x = x.strip()
        if not x:
            continue
        ids.append(int(x))
    if not ids:
        raise ValueError("No motor IDs parsed from --ids")
    return ids


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def mean_abs(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return sum(abs(x) for x in xs) / len(xs)


def rms(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return math.sqrt(sum(x * x for x in xs) / len(xs))


@dataclass
class ControlTable:
    torque_enable: int = 64
    goal_pos: int = 116
    present_pos: int = 132
    present_vel: int = 128
    present_current: int = -1
    p_gain: int = 84
    i_gain: int = 82
    d_gain: int = 80
    profile_vel: int = 112
    profile_acc: int = 108

    torque_enable_len: int = 1
    goal_pos_len: int = 4
    present_pos_len: int = 4
    present_vel_len: int = 4
    present_current_len: int = 2
    gain_len: int = 2
    profile_len: int = 4


def make_control_table(preset: str) -> ControlTable:
    """
    Presets:
    - x_series: Protocol 2.0 X/XL/XM/XH/MX(2.0 fw)
    - mx_legacy: Protocol 1.0 MX classic table (no PID gain registers)
    """
    if preset == "x_series":
        return ControlTable()
    if preset == "mx_legacy":
        # MX protocol 1.0 classic control table.
        # Note: no direct PID gain registers exposed as in protocol 2.0.
        return ControlTable(
            torque_enable=24,
            goal_pos=30,
            present_pos=36,
            present_vel=38,  # Present Speed (10-bit + direction bit, not true signed rpm)
            present_current=-1,
            p_gain=-1,
            i_gain=-1,
            d_gain=-1,
            profile_vel=-1,
            profile_acc=-1,
            torque_enable_len=1,
            goal_pos_len=2,
            present_pos_len=2,
            present_vel_len=2,
            present_current_len=2,
            gain_len=2,
            profile_len=4,
        )
    raise ValueError(f"Unknown table preset: {preset}")


class DXLBus:
    def __init__(self, port: str, baud: int, protocol: float) -> None:
        self.port_name = port
        self.baud = baud
        self.protocol = protocol
        self.port = PortHandler(port)
        self.packet = PacketHandler(protocol)

    def open(self) -> None:
        if not self.port.openPort():
            raise RuntimeError(f"Failed to open port {self.port_name}")
        if not self.port.setBaudRate(self.baud):
            raise RuntimeError(f"Failed to set baud {self.baud}")

    def close(self) -> None:
        self.port.closePort()

    def _txrx(self, fn, *args):
        out = fn(*args)
        if len(out) == 3:
            val, comm, err = out
        else:
            comm, err = out
            val = None
        if comm != 0:
            raise RuntimeError(self.packet.getTxRxResult(comm))
        if err != 0:
            raise RuntimeError(self.packet.getRxPacketError(err))
        return val

    def write_u8(self, dxl_id: int, addr: int, value: int) -> None:
        self._txrx(self.packet.write1ByteTxRx, self.port, dxl_id, addr, int(value))

    def write_u16(self, dxl_id: int, addr: int, value: int) -> None:
        self._txrx(self.packet.write2ByteTxRx, self.port, dxl_id, addr, int(value))

    def write_u32(self, dxl_id: int, addr: int, value: int) -> None:
        self._txrx(self.packet.write4ByteTxRx, self.port, dxl_id, addr, int(value))

    def read_u16(self, dxl_id: int, addr: int) -> int:
        return int(self._txrx(self.packet.read2ByteTxRx, self.port, dxl_id, addr))

    def read_u32(self, dxl_id: int, addr: int) -> int:
        return int(self._txrx(self.packet.read4ByteTxRx, self.port, dxl_id, addr))

    def read_reg(self, dxl_id: int, addr: int, reg_len: int) -> int:
        if reg_len == 1:
            return int(self._txrx(self.packet.read1ByteTxRx, self.port, dxl_id, addr))
        if reg_len == 2:
            return int(self._txrx(self.packet.read2ByteTxRx, self.port, dxl_id, addr))
        if reg_len == 4:
            return int(self._txrx(self.packet.read4ByteTxRx, self.port, dxl_id, addr))
        raise ValueError(f"Unsupported register length: {reg_len}")

    def write_reg(self, dxl_id: int, addr: int, value: int, reg_len: int) -> None:
        if reg_len == 1:
            self.write_u8(dxl_id, addr, value)
            return
        if reg_len == 2:
            self.write_u16(dxl_id, addr, value)
            return
        if reg_len == 4:
            self.write_u32(dxl_id, addr, value)
            return
        raise ValueError(f"Unsupported register length: {reg_len}")


def unsigned_to_signed(v: int, bits: int) -> int:
    if v >= (1 << (bits - 1)):
        v -= 1 << bits
    return v


def ticks_to_deg(ticks: int, min_tick: int, max_tick: int) -> float:
    span = max_tick - min_tick
    return (ticks - min_tick) * (360.0 / span)


def deg_to_ticks(deg: float, min_tick: int, max_tick: int) -> int:
    span = max_tick - min_tick
    return int(round((deg / 360.0) * span + min_tick))


def estimate_step_metrics(
    t: List[float], y: List[float], y0: float, y_target: float, tol_ratio: float
) -> Dict[str, float]:
    if len(t) < 5:
        return {
            "rise_time_s": float("nan"),
            "settling_time_s": float("nan"),
            "overshoot_pct": float("nan"),
            "steady_state_err_deg": float("nan"),
            "peak_deg": float("nan"),
        }

    step = y_target - y0
    amp = abs(step)
    if amp < 1e-6:
        return {
            "rise_time_s": 0.0,
            "settling_time_s": 0.0,
            "overshoot_pct": 0.0,
            "steady_state_err_deg": 0.0,
            "peak_deg": y[-1],
        }

    low = y0 + 0.1 * step
    high = y0 + 0.9 * step
    t10, t90 = None, None
    for ti, yi in zip(t, y):
        if t10 is None and ((step > 0 and yi >= low) or (step < 0 and yi <= low)):
            t10 = ti
        if t90 is None and ((step > 0 and yi >= high) or (step < 0 and yi <= high)):
            t90 = ti
            break
    rise_time = (t90 - t10) if (t10 is not None and t90 is not None) else float("nan")

    peak = max(y) if step > 0 else min(y)
    overshoot = max(0.0, (abs(peak - y_target) / amp) * 100.0)

    band = tol_ratio * amp
    settle_idx = None
    for i in range(len(y)):
        ok_rest = all(abs(yy - y_target) <= band for yy in y[i:])
        if ok_rest:
            settle_idx = i
            break
    settling = t[settle_idx] if settle_idx is not None else float("nan")

    tail = y[max(0, int(0.8 * len(y))) :]
    steady = statistics.mean(tail) if tail else y[-1]
    ss_err = steady - y_target

    return {
        "rise_time_s": rise_time,
        "settling_time_s": settling,
        "overshoot_pct": overshoot,
        "steady_state_err_deg": ss_err,
        "peak_deg": peak,
    }


def count_error_sign_flips(err: List[float], eps: float = 1e-5) -> int:
    flips = 0
    prev_sign = 0
    for e in err:
        if abs(e) <= eps:
            continue
        sign = 1 if e > 0 else -1
        if prev_sign != 0 and sign != prev_sign:
            flips += 1
        prev_sign = sign
    return flips


def write_gain_and_profile(
    bus: DXLBus, table: ControlTable, dxl_id: int, args: argparse.Namespace
) -> None:
    if args.p_gain is not None and table.p_gain >= 0:
        bus.write_u16(dxl_id, table.p_gain, args.p_gain)
    if args.i_gain is not None and table.i_gain >= 0:
        bus.write_u16(dxl_id, table.i_gain, args.i_gain)
    if args.d_gain is not None and table.d_gain >= 0:
        bus.write_u16(dxl_id, table.d_gain, args.d_gain)
    if args.profile_vel is not None and table.profile_vel >= 0:
        bus.write_u32(dxl_id, table.profile_vel, args.profile_vel)
    if args.profile_acc is not None and table.profile_acc >= 0:
        bus.write_u32(dxl_id, table.profile_acc, args.profile_acc)


def read_state(
    bus: DXLBus,
    table: ControlTable,
    dxl_id: int,
    pos_scale_deg_per_tick: float,
    prev_pos_deg: Optional[float],
    dt: float,
) -> Tuple[int, float, Optional[float]]:
    pos_u = bus.read_reg(dxl_id, table.present_pos, table.present_pos_len)
    pos = pos_u

    pos_deg = pos_u * pos_scale_deg_per_tick
    vel_deg_s = float("nan")
    if table.present_vel >= 0:
        vel_u = bus.read_reg(dxl_id, table.present_vel, table.present_vel_len)
        if table.present_vel_len == 4:
            vel = unsigned_to_signed(vel_u, 32)
            # Protocol 2.0 position-series: 1 unit = 0.229 rpm.
            vel_rpm = vel * 0.229
            vel_deg_s = vel_rpm * 6.0
        else:
            # Legacy Present Speed is ambiguous by model; fallback to finite-diff velocity.
            if prev_pos_deg is not None:
                vel_deg_s = (pos_deg - prev_pos_deg) / dt
    elif prev_pos_deg is not None:
        vel_deg_s = (pos_deg - prev_pos_deg) / dt

    cur = None
    if table.present_current >= 0:
        cur_u = bus.read_reg(dxl_id, table.present_current, table.present_current_len)
        cur = float(unsigned_to_signed(cur_u, 8 * table.present_current_len))

    return pos, vel_deg_s, cur


def main() -> None:
    parser = argparse.ArgumentParser(description="DYNAMIXEL PD step-response tester")
    parser.add_argument("--port", type=str, required=True, help="Serial port (e.g., /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=3000000)
    parser.add_argument("--protocol", type=float, default=2.0)
    parser.add_argument("--ids", type=str, required=True, help="Comma-separated IDs, e.g., 1,2,3")
    parser.add_argument("--table-preset", type=str, default="x_series", choices=["x_series", "mx_legacy"])

    parser.add_argument("--step-deg", type=float, default=5.0)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--warmup-s", type=float, default=0.8)
    parser.add_argument("--hold-s", type=float, default=0.8)
    parser.add_argument("--return-s", type=float, default=0.8)
    parser.add_argument("--sample-hz", type=float, default=200.0)
    parser.add_argument("--tol-ratio", type=float, default=0.05, help="Settling band ratio (default: 5%)")

    parser.add_argument("--min-tick", type=int, default=0)
    parser.add_argument("--max-tick", type=int, default=4095)
    parser.add_argument("--center-mode", type=str, default="present", choices=["present"])

    parser.add_argument("--p-gain", type=int, default=None)
    parser.add_argument("--i-gain", type=int, default=None)
    parser.add_argument("--d-gain", type=int, default=None)
    parser.add_argument("--profile-vel", type=int, default=None)
    parser.add_argument("--profile-acc", type=int, default=None)

    parser.add_argument("--log-dir", type=str, default="/tmp")
    parser.add_argument("--tag", type=str, default="dxl_step")

    parser.add_argument("--addr-torque-enable", type=int, default=None)
    parser.add_argument("--addr-goal-pos", type=int, default=None)
    parser.add_argument("--addr-present-pos", type=int, default=None)
    parser.add_argument("--addr-present-vel", type=int, default=None)
    parser.add_argument("--addr-present-current", type=int, default=None)
    parser.add_argument("--addr-p-gain", type=int, default=None)
    parser.add_argument("--addr-i-gain", type=int, default=None)
    parser.add_argument("--addr-d-gain", type=int, default=None)
    parser.add_argument("--addr-profile-vel", type=int, default=None)
    parser.add_argument("--addr-profile-acc", type=int, default=None)
    parser.add_argument("--len-goal-pos", type=int, default=None, choices=[1, 2, 4])
    parser.add_argument("--len-present-pos", type=int, default=None, choices=[1, 2, 4])
    parser.add_argument("--len-present-vel", type=int, default=None, choices=[1, 2, 4])
    parser.add_argument("--len-present-current", type=int, default=None, choices=[1, 2, 4])

    args = parser.parse_args()
    ids = parse_ids(args.ids)
    dt = 1.0 / args.sample_hz

    table = make_control_table(args.table_preset)
    for field_name, arg_name in [
        ("torque_enable", "addr_torque_enable"),
        ("goal_pos", "addr_goal_pos"),
        ("present_pos", "addr_present_pos"),
        ("present_vel", "addr_present_vel"),
        ("present_current", "addr_present_current"),
        ("p_gain", "addr_p_gain"),
        ("i_gain", "addr_i_gain"),
        ("d_gain", "addr_d_gain"),
        ("profile_vel", "addr_profile_vel"),
        ("profile_acc", "addr_profile_acc"),
        ("goal_pos_len", "len_goal_pos"),
        ("present_pos_len", "len_present_pos"),
        ("present_vel_len", "len_present_vel"),
        ("present_current_len", "len_present_current"),
    ]:
        v = getattr(args, arg_name)
        if v is not None:
            table = replace(table, **{field_name: v})

    bus = DXLBus(args.port, args.baud, args.protocol)

    os.makedirs(args.log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.log_dir, f"{args.tag}_{ts}.csv")
    summary_path = os.path.join(args.log_dir, f"{args.tag}_{ts}_summary.csv")
    deg_per_tick = 360.0 / max(1, args.max_tick - args.min_tick)

    print("[info] opening bus...")
    bus.open()
    try:
        for dxl_id in ids:
            bus.write_u8(dxl_id, table.torque_enable, 0)
            write_gain_and_profile(bus, table, dxl_id, args)
            bus.write_u8(dxl_id, table.torque_enable, 1)

        centers_tick: Dict[int, int] = {}
        for dxl_id in ids:
            pos_tick, _, _ = read_state(bus, table, dxl_id, deg_per_tick, None, dt)
            centers_tick[dxl_id] = pos_tick

        print("[info] center ticks:", centers_tick)
        print("[info] logging to:", csv_path)

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "t",
                    "phase",
                    "repeat",
                    "id",
                    "goal_tick",
                    "goal_deg",
                    "pos_tick",
                    "pos_deg",
                    "vel_deg_s",
                    "acc_deg_s2",
                    "err_deg",
                    "present_current_raw",
                ]
            )
            summary = open(summary_path, "w", newline="")
            summary_writer = csv.writer(summary)
            summary_writer.writerow(
                [
                    "repeat",
                    "phase",
                    "id",
                    "rise_time_s",
                    "settling_time_s",
                    "overshoot_pct",
                    "steady_state_err_deg",
                    "vel_rms_deg_s",
                    "acc_rms_deg_s2",
                    "jerk_rms_deg_s3",
                    "peak_abs_vel_deg_s",
                    "peak_abs_acc_deg_s2",
                    "error_sign_flips",
                ]
            )

            t0 = time.perf_counter()
            phase = "warmup"
            active_repeat = -1
            goal_tick: Dict[int, int] = dict(centers_tick)

            def set_all_goals(value_map: Dict[int, int]) -> None:
                for did, g in value_map.items():
                    bus.write_reg(did, table.goal_pos, g, table.goal_pos_len)

            set_all_goals(goal_tick)

            # Segment buffers for metrics per motor and per step.
            seg_t: Dict[int, List[float]] = {i: [] for i in ids}
            seg_y: Dict[int, List[float]] = {i: [] for i in ids}
            seg_y0: Dict[int, float] = {i: ticks_to_deg(centers_tick[i], args.min_tick, args.max_tick) for i in ids}
            seg_yt: Dict[int, float] = dict(seg_y0)
            vel_buf: Dict[int, List[float]] = {i: [] for i in ids}
            acc_buf: Dict[int, List[float]] = {i: [] for i in ids}
            jerk_buf: Dict[int, List[float]] = {i: [] for i in ids}
            err_buf: Dict[int, List[float]] = {i: [] for i in ids}
            prev_vel: Dict[int, Optional[float]] = {i: None for i in ids}
            prev_pos: Dict[int, Optional[float]] = {i: None for i in ids}
            prev_acc: Dict[int, Optional[float]] = {i: None for i in ids}

            def flush_metrics(label: str) -> None:
                print(f"[metrics] {label}")
                for did in ids:
                    m = estimate_step_metrics(seg_t[did], seg_y[did], seg_y0[did], seg_yt[did], args.tol_ratio)
                    print(
                        f"  id={did:>3} rise={m['rise_time_s']:.3f}s settle={m['settling_time_s']:.3f}s "
                        f"overshoot={m['overshoot_pct']:.2f}% ss_err={m['steady_state_err_deg']:+.2f}deg "
                        f"vel_rms={rms(vel_buf[did]):.2f}deg/s acc_rms={rms(acc_buf[did]):.2f}deg/s^2 "
                        f"peak_vel={max([abs(v) for v in vel_buf[did]], default=0.0):.2f}deg/s "
                        f"peak_acc={max([abs(a) for a in acc_buf[did]], default=0.0):.2f}deg/s^2 "
                        f"err_flips={count_error_sign_flips(err_buf[did])}"
                    )
                    summary_writer.writerow(
                        [
                            active_repeat,
                            label.split("phase=")[-1] if "phase=" in label else label,
                            did,
                            f"{m['rise_time_s']:.6f}",
                            f"{m['settling_time_s']:.6f}",
                            f"{m['overshoot_pct']:.6f}",
                            f"{m['steady_state_err_deg']:.6f}",
                            f"{rms(vel_buf[did]):.6f}",
                            f"{rms(acc_buf[did]):.6f}",
                            f"{rms(jerk_buf[did]):.6f}",
                            f"{max([abs(v) for v in vel_buf[did]], default=0.0):.6f}",
                            f"{max([abs(a) for a in acc_buf[did]], default=0.0):.6f}",
                            count_error_sign_flips(err_buf[did]),
                        ]
                    )
                # reset segment buffers
                for did in ids:
                    seg_t[did].clear()
                    seg_y[did].clear()
                    vel_buf[did].clear()
                    acc_buf[did].clear()
                    jerk_buf[did].clear()
                    err_buf[did].clear()
                    prev_vel[did] = None
                    prev_acc[did] = None

            # Warmup.
            t_end = args.warmup_s
            while True:
                now = time.perf_counter() - t0
                if now >= t_end:
                    break
                for did in ids:
                    pos_tick, vel_deg_s, cur = read_state(
                        bus, table, did, deg_per_tick, prev_pos[did], dt
                    )
                    pos_deg = ticks_to_deg(pos_tick, args.min_tick, args.max_tick)
                    acc = float("nan")
                    if prev_vel[did] is not None and not math.isnan(vel_deg_s):
                        acc = (vel_deg_s - prev_vel[did]) / dt
                    prev_vel[did] = vel_deg_s if not math.isnan(vel_deg_s) else prev_vel[did]
                    prev_pos[did] = pos_deg
                    err = ticks_to_deg(goal_tick[did], args.min_tick, args.max_tick) - pos_deg
                    writer.writerow(
                        [
                            f"{now:.6f}",
                            phase,
                            active_repeat,
                            did,
                            goal_tick[did],
                            f"{ticks_to_deg(goal_tick[did], args.min_tick, args.max_tick):.6f}",
                            pos_tick,
                            f"{pos_deg:.6f}",
                            f"{vel_deg_s:.6f}",
                            f"{acc:.6f}",
                            f"{err:.6f}",
                            "" if cur is None else f"{cur:.6f}",
                        ]
                    )
                time.sleep(dt)

            # Repeated step -> return.
            step_ticks = deg_to_ticks(args.step_deg, args.min_tick, args.max_tick) - deg_to_ticks(0.0, args.min_tick, args.max_tick)
            for rep in range(args.repeats):
                active_repeat = rep

                # STEP UP
                phase = "step_up"
                for did in ids:
                    seg_y0[did] = ticks_to_deg(centers_tick[did], args.min_tick, args.max_tick)
                    goal_tick[did] = int(clamp(centers_tick[did] + step_ticks, args.min_tick, args.max_tick))
                    seg_yt[did] = ticks_to_deg(goal_tick[did], args.min_tick, args.max_tick)
                set_all_goals(goal_tick)
                t_phase0 = time.perf_counter()
                while (time.perf_counter() - t_phase0) < args.hold_s:
                    now = time.perf_counter() - t0
                    for did in ids:
                        pos_tick, vel_deg_s, cur = read_state(
                            bus, table, did, deg_per_tick, prev_pos[did], dt
                        )
                        pos_deg = ticks_to_deg(pos_tick, args.min_tick, args.max_tick)
                        seg_t[did].append(time.perf_counter() - t_phase0)
                        seg_y[did].append(pos_deg)
                        if not math.isnan(vel_deg_s):
                            vel_buf[did].append(vel_deg_s)
                            if prev_vel[did] is not None:
                                acc = (vel_deg_s - prev_vel[did]) / dt
                                acc_buf[did].append(acc)
                                if prev_acc[did] is not None:
                                    jerk_buf[did].append((acc - prev_acc[did]) / dt)
                                prev_acc[did] = acc
                            prev_vel[did] = vel_deg_s
                        prev_pos[did] = pos_deg
                        err = ticks_to_deg(goal_tick[did], args.min_tick, args.max_tick) - pos_deg
                        err_buf[did].append(err)
                        writer.writerow(
                            [
                                f"{now:.6f}",
                                phase,
                                active_repeat,
                                did,
                                goal_tick[did],
                                f"{ticks_to_deg(goal_tick[did], args.min_tick, args.max_tick):.6f}",
                                pos_tick,
                                f"{pos_deg:.6f}",
                                f"{vel_deg_s:.6f}",
                                f"{(acc_buf[did][-1] if acc_buf[did] else float('nan')):.6f}",
                                f"{err:.6f}",
                                "" if cur is None else f"{cur:.6f}",
                            ]
                        )
                    time.sleep(dt)
                flush_metrics(f"repeat={rep} phase=step_up")

                # RETURN
                phase = "return"
                for did in ids:
                    seg_y0[did] = ticks_to_deg(goal_tick[did], args.min_tick, args.max_tick)
                    goal_tick[did] = centers_tick[did]
                    seg_yt[did] = ticks_to_deg(goal_tick[did], args.min_tick, args.max_tick)
                set_all_goals(goal_tick)
                t_phase0 = time.perf_counter()
                while (time.perf_counter() - t_phase0) < args.return_s:
                    now = time.perf_counter() - t0
                    for did in ids:
                        pos_tick, vel_deg_s, cur = read_state(
                            bus, table, did, deg_per_tick, prev_pos[did], dt
                        )
                        pos_deg = ticks_to_deg(pos_tick, args.min_tick, args.max_tick)
                        seg_t[did].append(time.perf_counter() - t_phase0)
                        seg_y[did].append(pos_deg)
                        if not math.isnan(vel_deg_s):
                            vel_buf[did].append(vel_deg_s)
                            if prev_vel[did] is not None:
                                acc = (vel_deg_s - prev_vel[did]) / dt
                                acc_buf[did].append(acc)
                                if prev_acc[did] is not None:
                                    jerk_buf[did].append((acc - prev_acc[did]) / dt)
                                prev_acc[did] = acc
                            prev_vel[did] = vel_deg_s
                        prev_pos[did] = pos_deg
                        err = ticks_to_deg(goal_tick[did], args.min_tick, args.max_tick) - pos_deg
                        err_buf[did].append(err)
                        writer.writerow(
                            [
                                f"{now:.6f}",
                                phase,
                                active_repeat,
                                did,
                                goal_tick[did],
                                f"{ticks_to_deg(goal_tick[did], args.min_tick, args.max_tick):.6f}",
                                pos_tick,
                                f"{pos_deg:.6f}",
                                f"{vel_deg_s:.6f}",
                                f"{(acc_buf[did][-1] if acc_buf[did] else float('nan')):.6f}",
                                f"{err:.6f}",
                                "" if cur is None else f"{cur:.6f}",
                            ]
                        )
                    time.sleep(dt)
                flush_metrics(f"repeat={rep} phase=return")
            summary.close()

        print("[done] step test complete.")
        print("[done] csv:", csv_path)
        print("[done] summary:", summary_path)
    finally:
        for dxl_id in ids:
            try:
                bus.write_u8(dxl_id, table.torque_enable, 0)
            except Exception:
                pass
        bus.close()


if __name__ == "__main__":
    main()
