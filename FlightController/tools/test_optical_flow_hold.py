"""飞控原生光流定点飞行测试：起飞到 1 m，悬停 30 s，然后降落。

本程序只连接 ``FC_Controller``。它不会创建雷达、雷达 SLAM、相机导航或
伴随计算机位置控制对象；水平定点完全由飞控的 HOLD_POS（mode=2）完成。

为防止误触发，默认只打印任务计划。真实飞行必须显式传入 ``--execute``::

    PYTHONPATH=. python -u FlightController/tools/test_optical_flow_hold.py --execute
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any


HOLD_POS_MODE = 2
PROGRAM_MODE = 3
TAKEOFF_COMMAND = (0x10, 0x00, 0x05)
TAKEOFF_MIN_RISE_CM = 8.0
TAKEOFF_MIN_VZ_CM_S = 4.0


class _ConsecutiveRangeGuard:
    """Confirm a range violation across distinct state frames."""

    def __init__(
        self,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
        confirm_frames: int = 3,
    ) -> None:
        self.minimum = minimum
        self.maximum = maximum
        self.confirm_frames = max(1, int(confirm_frames))
        self.low_count = 0
        self.high_count = 0

    def observe(self, value: float) -> str | None:
        self.low_count = (
            self.low_count + 1
            if self.minimum is not None and value < self.minimum
            else 0
        )
        self.high_count = (
            self.high_count + 1
            if self.maximum is not None and value > self.maximum
            else 0
        )
        if self.low_count >= self.confirm_frames:
            return "low"
        if self.high_count >= self.confirm_frames:
            return "high"
        return None


def _setup_path() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in (root, root.parent):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用飞控自身光流定点：起飞到 1 m，定点 30 s，然后自动降落。"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="实际解锁、起飞和降落；不传时只做无硬件任务预览。",
    )
    parser.add_argument("--port", default=None, help="飞控串口；默认按 VID/PID 自动探测。")
    parser.add_argument("--height-cm", type=int, default=100, help="目标高度，默认 100 cm。")
    parser.add_argument("--hover-s", type=float, default=30.0, help="定点时间，默认 30 s。")
    parser.add_argument("--connection-timeout-s", type=float, default=10.0)
    parser.add_argument("--unlock-timeout-s", type=float, default=5.0)
    parser.add_argument(
        "--post-unlock-delay-s",
        type=float,
        default=2.0,
        help="确认解锁后等待飞控和电机状态稳定的时间，默认 2.0 s。",
    )
    parser.add_argument(
        "--takeoff-start-timeout-s",
        type=float,
        default=8.0,
        help="等待飞控进入起飞功能或检测到实际离地的时间，默认 8.0 s。",
    )
    parser.add_argument("--takeoff-timeout-s", type=float, default=25.0)
    parser.add_argument("--landing-timeout-s", type=float, default=30.0)
    parser.add_argument("--height-tolerance-cm", type=float, default=15.0)
    parser.add_argument("--stable-s", type=float, default=1.5, help="进入定点前高度稳定时间。")
    parser.add_argument("--max-drift-cm", type=float, default=80.0, help="允许的最大水平漂移。")
    parser.add_argument("--max-tilt-deg", type=float, default=25.0, help="飞行中允许的最大横滚/俯仰角。")
    parser.add_argument(
        "--min-battery-v",
        type=float,
        default=10.5,
        help="最低起飞电压，默认按 3S 电池 10.5 V；按实际电池修改。",
    )
    parser.add_argument("--status-interval-s", type=float, default=1.0)
    parser.add_argument(
        "--height-outlier-confirm-frames",
        type=int,
        default=3,
        help="融合高度连续越界多少个状态帧才触发安全降落，默认 3 帧。",
    )
    args = parser.parse_args(argv)

    if not 40 <= args.height_cm <= 500:
        parser.error("--height-cm 必须在飞控一键起飞支持的 40..500 cm 范围内")
    if args.hover_s < 0:
        parser.error("--hover-s 不能为负数")
    for name in (
        "connection_timeout_s",
        "unlock_timeout_s",
        "post_unlock_delay_s",
        "takeoff_start_timeout_s",
        "takeoff_timeout_s",
        "landing_timeout_s",
        "height_tolerance_cm",
        "stable_s",
        "max_drift_cm",
        "max_tilt_deg",
        "status_interval_s",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须大于 0")
    if args.height_tolerance_cm >= args.height_cm - 20:
        parser.error("--height-tolerance-cm 过大，必须给起飞确认保留至少 20 cm 高度")
    if args.min_battery_v <= 0:
        parser.error("--min-battery-v 必须大于 0")
    if args.height_outlier_confirm_frames < 2:
        parser.error("--height-outlier-confirm-frames 必须至少为 2")
    return args


def _print_plan(args: argparse.Namespace) -> None:
    print("无硬件任务预览：未连接飞控，也不会解锁。")
    print(
        f"计划：飞控原生 HOLD_POS(mode=2)，起飞到 {args.height_cm} cm，"
        f"定点 {args.hover_s:g} s，然后降落。"
    )
    print("外部雷达/雷达 SLAM/相机导航/位置回灌：全部不启动。")
    print("确认场地、光流纹理与照明、桨叶和遥控接管条件后，加 --execute 执行真实飞行。")


def _wait_for_fresh_state(fc: Any, timeout_s: float) -> None:
    fc.state.update_event.clear()
    if not fc.state.update_event.wait(timeout_s):
        raise RuntimeError("等待飞控状态数据超时")
    if not fc.connected:
        raise RuntimeError("飞控连接已断开")


def _wait_for_next_state(fc: Any, previous_count: int, timeout_s: float = 1.0) -> int:
    """Wait for a state frame newer than ``previous_count`` without missing races."""
    state = fc.state
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        current_count = int(getattr(state, "update_count", previous_count + 1))
        if current_count != previous_count:
            if not fc.connected:
                raise RuntimeError("等待飞控状态更新时连接已断开")
            return current_count
        state.update_event.clear()
        current_count = int(getattr(state, "update_count", previous_count + 1))
        if current_count != previous_count:
            continue
        remaining = deadline - time.perf_counter()
        if remaining > 0:
            state.update_event.wait(remaining)
    raise RuntimeError("等待下一帧飞控状态数据超时")


def _flight_state_raw(fc: Any) -> int:
    unlock_var = fc.state.unlock
    return int(getattr(unlock_var, "raw_value", int(bool(unlock_var.value))))


def _command_now(fc: Any) -> tuple[int, int, int]:
    state = fc.state
    return (int(state.cid.value), int(state.cmd_0.value), int(state.cmd_1.value))


def _raw_state_hex(fc: Any) -> str:
    raw = getattr(fc.state, "last_raw_bytes", b"")
    return bytes(raw).hex(" ") if raw else "unavailable"


def _height_values(fc: Any) -> tuple[float, float, float]:
    state = fc.state
    return (
        float(state.alt_fused.value),
        float(state.alt_add.value),
        float(state.vel_z.value),
    )


def _takeoff_evidence(fc: Any, baseline_fused_cm: float) -> dict[str, bool]:
    fused_cm, _add_cm, vertical_speed = _height_values(fc)
    return {
        "command": _command_now(fc) == TAKEOFF_COMMAND,
        "airborne_state": _flight_state_raw(fc) >= 2,
        "height_rise": fused_cm - baseline_fused_cm >= TAKEOFF_MIN_RISE_CM,
        "vertical_speed": vertical_speed >= TAKEOFF_MIN_VZ_CM_S,
    }


def _wait_for_mode(fc: Any, target_mode: int, timeout_s: float = 5.0) -> None:
    fc.set_flight_mode(target_mode)
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if not fc.connected:
            raise RuntimeError("切换飞行模式时飞控断开")
        if int(fc.state.mode.value) == target_mode:
            return
        time.sleep(0.05)
    raise RuntimeError(f"飞行模式切换失败：期望 mode={target_mode}，实际 mode={fc.state.mode.value}")


def _check_tilt(fc: Any, max_tilt_deg: float) -> None:
    roll = float(fc.state.rol.value)
    pitch = float(fc.state.pit.value)
    if abs(roll) > max_tilt_deg or abs(pitch) > max_tilt_deg:
        raise RuntimeError(
            f"机体倾角超过安全限制：roll={roll:.1f}°, pitch={pitch:.1f}°，"
            f"限制={max_tilt_deg:.1f}°"
        )


def _preflight(fc: Any, args: argparse.Namespace) -> float:
    _wait_for_fresh_state(fc, args.connection_timeout_s)
    state = fc.state
    battery_v = float(state.bat.value)

    if bool(state.unlock.value):
        raise RuntimeError("飞控在程序启动前已经解锁；拒绝接管，请先人工锁定")
    if battery_v <= 1.0:
        raise RuntimeError("飞控未报告有效电池电压；禁止自动起飞")
    if battery_v < args.min_battery_v:
        raise RuntimeError(
            f"电池电压过低：{battery_v:.2f} V < {args.min_battery_v:.2f} V"
        )
    _check_tilt(fc, min(args.max_tilt_deg, 15.0))

    fused_samples: list[float] = []
    add_samples: list[float] = []
    update_count = int(getattr(state, "update_count", 0))
    for _ in range(12):
        update_count = _wait_for_next_state(fc, update_count, timeout_s=1.0)
        fused_cm, add_cm, _vertical_speed = _height_values(fc)
        fused_samples.append(fused_cm)
        add_samples.append(add_cm)

    fused_cm = float(statistics.median(fused_samples))
    add_cm = float(statistics.median(add_samples))
    fused_span = max(fused_samples) - min(fused_samples)
    add_span = max(add_samples) - min(add_samples)
    if abs(fused_cm) > 30.0 or fused_span > 10.0:
        raise RuntimeError(
            "起飞前融合高度不是稳定地面值："
            f"alt_fused中值={fused_cm:.1f} cm, 范围={min(fused_samples):.1f}..{max(fused_samples):.1f} cm"
        )
    if abs(add_cm) > 30.0 or add_span > 10.0:
        raise RuntimeError(
            "起飞前附加测高不稳定，禁止光流起飞："
            f"alt_add中值={add_cm:.1f} cm, 范围={min(add_samples):.1f}..{max(add_samples):.1f} cm, "
            f"raw={_raw_state_hex(fc)}"
        )
    print(
        "起飞前检查通过："
        f"battery={battery_v:.2f} V, alt_fused={fused_cm:.1f} cm, alt_add={add_cm:.1f} cm, "
        f"roll={state.rol.value:.1f}°, pitch={state.pit.value:.1f}°"
    )
    return fused_cm


def _wait_for_unlock(fc: Any, timeout_s: float) -> int:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if not fc.connected:
            raise RuntimeError("等待解锁时飞控断开")
        if bool(fc.state.unlock.value):
            return _flight_state_raw(fc)
        time.sleep(0.05)
    raise RuntimeError("飞控解锁确认超时")


def _wait_for_takeoff_start(fc: Any, args: argparse.Namespace, baseline_fused_cm: float) -> None:
    """Require command-state or physical evidence that one-key takeoff started."""
    deadline = time.perf_counter() + args.takeoff_start_timeout_s
    update_count = int(getattr(fc.state, "update_count", 0))
    command_seen = False
    next_status = 0.0
    while time.perf_counter() < deadline:
        update_count = _wait_for_next_state(fc, update_count, timeout_s=1.0)
        fused_cm, add_cm, vertical_speed = _height_values(fc)
        flight_state = _flight_state_raw(fc)
        command = _command_now(fc)
        evidence = _takeoff_evidence(fc, baseline_fused_cm)
        command_seen = command_seen or evidence["command"]
        physical_evidence = (
            evidence["airborne_state"]
            or evidence["height_rise"]
            or evidence["vertical_speed"]
        )
        if physical_evidence:
            reasons = [name for name, present in evidence.items() if present]
            print(
                "起飞已确认："
                f"依据={','.join(reasons)}, alt_fused={fused_cm:.1f} cm, "
                f"alt_add={add_cm:.1f} cm, vz={vertical_speed:.1f} cm/s, "
                f"flight_state={flight_state}, cmd={command}"
            )
            if not command_seen:
                print("警告：未观察到起飞命令状态，但已检测到实际离地；继续按融合高度监控。")
            return

        now = time.perf_counter()
        if now >= next_status:
            print(
                "等待起飞确认："
                f"alt_fused={fused_cm:.1f} cm, alt_add={add_cm:.1f} cm, "
                f"vz={vertical_speed:.1f} cm/s, flight_state={flight_state}, cmd={command}"
            )
            next_status = now + args.status_interval_s

    fused_cm, add_cm, vertical_speed = _height_values(fc)
    raise RuntimeError(
        "飞控未确认起飞，已停止继续等待："
        f"takeoff_cmd_seen={command_seen}, alt_fused={fused_cm:.1f} cm, "
        f"alt_add={add_cm:.1f} cm, vz={vertical_speed:.1f} cm/s, "
        f"flight_state={_flight_state_raw(fc)}, cmd={_command_now(fc)}, "
        f"raw={_raw_state_hex(fc)}"
    )


def _wait_for_takeoff_height(fc: Any, args: argparse.Namespace) -> None:
    deadline = time.perf_counter() + args.takeoff_timeout_s
    stable_since: float | None = None
    next_status = 0.0
    minimum_height = args.height_cm - args.height_tolerance_cm
    hard_ceiling = args.height_cm + max(60.0, args.height_tolerance_cm * 3.0)
    ceiling_guard = _ConsecutiveRangeGuard(
        maximum=hard_ceiling,
        confirm_frames=args.height_outlier_confirm_frames,
    )
    update_count = int(getattr(fc.state, "update_count", 0))
    add_outlier_active = False

    while time.perf_counter() < deadline:
        update_count = _wait_for_next_state(fc, update_count, timeout_s=1.0)
        now = time.perf_counter()
        if not fc.connected:
            raise RuntimeError("起飞过程中飞控断开")
        if not bool(fc.state.unlock.value):
            raise RuntimeError("起飞过程中飞控意外锁定")
        _check_tilt(fc, args.max_tilt_deg)

        fused_cm, add_cm, vertical_speed = _height_values(fc)
        violation = ceiling_guard.observe(fused_cm)
        if fused_cm > hard_ceiling and ceiling_guard.high_count == 1:
            print(
                "警告：融合高度出现单帧越界，等待连续状态帧确认："
                f"alt_fused={fused_cm:.1f} cm, alt_add={add_cm:.1f} cm, raw={_raw_state_hex(fc)}"
            )
        if violation == "high":
            raise RuntimeError(
                "融合高度连续超过安全上限："
                f"alt_fused={fused_cm:.1f} cm > {hard_ceiling:.1f} cm, "
                f"连续帧={ceiling_guard.high_count}, alt_add={add_cm:.1f} cm, raw={_raw_state_hex(fc)}"
            )

        add_outlier = add_cm < -30.0 or add_cm > hard_ceiling
        if add_outlier and not add_outlier_active:
            print(
                "警告：ALT_ADD 测距异常，本帧不用于高度安全判断："
                f"alt_add={add_cm:.1f} cm, alt_fused={fused_cm:.1f} cm, raw={_raw_state_hex(fc)}"
            )
        add_outlier_active = add_outlier

        if fused_cm >= minimum_height and abs(vertical_speed) <= 10.0:
            stable_since = stable_since or now
            if now - stable_since >= args.stable_s:
                print(
                    "已到达目标高度："
                    f"alt_fused={fused_cm:.1f} cm, alt_add={add_cm:.1f} cm, "
                    f"vz={vertical_speed:.1f} cm/s"
                )
                return
        else:
            stable_since = None

        if now >= next_status:
            print(
                f"起飞中：alt_fused={fused_cm:.1f} cm, alt_add={add_cm:.1f} cm, "
                f"vz={vertical_speed:.1f} cm/s, flight_state={_flight_state_raw(fc)}, "
                f"cmd={_command_now(fc)}, mode={fc.state.mode.value}"
            )
            next_status = now + args.status_interval_s

    fused_cm, add_cm, vertical_speed = _height_values(fc)
    raise RuntimeError(
        "起飞高度确认超时："
        f"alt_fused={fused_cm:.1f} cm, alt_add={add_cm:.1f} cm, "
        f"vz={vertical_speed:.1f} cm/s, 目标={args.height_cm} cm, "
        f"flight_state={_flight_state_raw(fc)}, cmd={_command_now(fc)}"
    )


def _hold_with_fc_optical_flow(fc: Any, args: argparse.Namespace) -> tuple[float, float]:
    """Monitor the hover without sending any external position or velocity data."""
    _wait_for_mode(fc, HOLD_POS_MODE)
    fc.stablize()
    time.sleep(0.5)

    origin_x = float(fc.state.pos_x.value)
    origin_y = float(fc.state.pos_y.value)
    deadline = time.perf_counter() + args.hover_s
    next_status = 0.0
    max_drift = 0.0
    max_height_error = 0.0
    height_guard = _ConsecutiveRangeGuard(
        minimum=25.0,
        maximum=args.height_cm + 80.0,
        confirm_frames=args.height_outlier_confirm_frames,
    )
    update_count = int(getattr(fc.state, "update_count", 0))
    add_outlier_active = False

    print(
        f"进入飞控原生光流定点：mode={fc.state.mode.value}，持续 {args.hover_s:g} s；"
        "程序不发送外部位置或速度控制量。"
    )
    while time.perf_counter() < deadline:
        update_count = _wait_for_next_state(fc, update_count, timeout_s=1.0)
        now = time.perf_counter()
        if not fc.connected:
            raise RuntimeError("定点过程中飞控断开")
        if not bool(fc.state.unlock.value):
            raise RuntimeError("定点过程中飞控意外锁定")
        if int(fc.state.mode.value) != HOLD_POS_MODE:
            raise RuntimeError(f"定点模式丢失：当前 mode={fc.state.mode.value}")
        _check_tilt(fc, args.max_tilt_deg)

        fused_cm, add_cm, _vertical_speed = _height_values(fc)
        dx = float(fc.state.pos_x.value) - origin_x
        dy = float(fc.state.pos_y.value) - origin_y
        drift_cm = math.hypot(dx, dy)
        max_drift = max(max_drift, drift_cm)
        max_height_error = max(max_height_error, abs(fused_cm - args.height_cm))

        if drift_cm > args.max_drift_cm:
            raise RuntimeError(
                f"光流定点漂移超过限制：{drift_cm:.1f} cm > {args.max_drift_cm:.1f} cm"
            )
        height_violation = height_guard.observe(fused_cm)
        if height_violation is not None:
            raise RuntimeError(
                "定点融合高度连续越过安全范围："
                f"alt_fused={fused_cm:.1f} cm, alt_add={add_cm:.1f} cm, "
                f"方向={height_violation}, 连续帧="
                f"{height_guard.low_count if height_violation == 'low' else height_guard.high_count}, "
                f"raw={_raw_state_hex(fc)}"
            )

        add_outlier = add_cm < -30.0 or add_cm > args.height_cm + 80.0
        if add_outlier and not add_outlier_active:
            print(
                "警告：定点期间 ALT_ADD 测距异常，继续使用融合高度："
                f"alt_add={add_cm:.1f} cm, alt_fused={fused_cm:.1f} cm, raw={_raw_state_hex(fc)}"
            )
        add_outlier_active = add_outlier

        if now >= next_status:
            remaining = max(0.0, deadline - now)
            print(
                f"定点中：剩余={remaining:4.1f} s, alt_fused={fused_cm:5.1f} cm, "
                f"alt_add={add_cm:5.1f} cm, drift={drift_cm:5.1f} cm, mode={fc.state.mode.value}"
            )
            next_status = now + args.status_interval_s

    return max_drift, max_height_error


def _land_and_wait_for_lock(fc: Any, args: argparse.Namespace) -> bool:
    """Request native landing and wait for the FC to lock; never force-lock."""
    print("请求飞控原生降落……")
    try:
        if not fc.connected:
            print("错误：飞控已断开，无法发送降落命令；请立即遥控接管。")
            return False
        fc.stablize()
        fc.land()
        deadline = time.perf_counter() + args.landing_timeout_s
        next_request = time.perf_counter() + 2.0
        next_status = 0.0

        while time.perf_counter() < deadline:
            now = time.perf_counter()
            fused_cm, add_cm, _vertical_speed = _height_values(fc)
            unlocked = bool(fc.state.unlock.value)
            if not unlocked:
                print("降落完成：飞控已锁定。")
                return True
            if not fc.connected:
                print("错误：降落过程中飞控断开；请立即遥控接管。")
                return False
            if now >= next_request:
                fc.land()
                next_request = now + 2.0
            if now >= next_status:
                print(
                    f"降落中：alt_fused={fused_cm:.1f} cm, alt_add={add_cm:.1f} cm, "
                    f"flight_state={_flight_state_raw(fc)}, unlock={unlocked}"
                )
                next_status = now + args.status_interval_s
            time.sleep(0.1)

        print("错误：降落确认超时；已发送降落命令，但为避免空中锁桨，没有强制锁定。")
        print("请立即使用遥控器接管并人工降落。")
        return False
    except Exception as exc:
        print(f"错误：发送/确认降落失败：{type(exc).__name__}: {exc}")
        print("请立即使用遥控器接管。")
        return False


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.execute:
        _print_plan(args)
        return 0

    _setup_path()
    from FlightController import FC_Controller

    fc = FC_Controller()
    flight_owned = False
    result = 1
    mission_ok = False
    landed_ok = True
    try:
        print("连接飞控（本程序不会打开任何雷达设备）……")
        fc.start_listen_serial(
            serial_dev=args.port,
            block_until_connected=True,
            open_timeout_s=args.connection_timeout_s,
        )
        if not fc.wait_for_connection(timeout_s=args.connection_timeout_s):
            raise RuntimeError("飞控连接超时")
        _preflight(fc, args)

        # One-key takeoff is most reliable in PROGRAM mode. Once the target
        # height is stable, mode=2 hands horizontal hold to the FC optical flow.
        _wait_for_mode(fc, PROGRAM_MODE)
        print("请求飞控解锁……")
        fc.unlock()
        flight_owned = True
        flight_state = _wait_for_unlock(fc, args.unlock_timeout_s)
        print(
            f"解锁状态已确认：flight_state={flight_state}；"
            f"等待 {args.post_unlock_delay_s:.1f} s 让飞控和电机状态稳定……"
        )
        time.sleep(args.post_unlock_delay_s)
        if not fc.connected:
            raise RuntimeError("解锁稳定等待期间飞控断开")
        if not bool(fc.state.unlock.value):
            raise RuntimeError("解锁稳定等待期间飞控重新锁定")

        takeoff_baseline_fused = float(fc.state.alt_fused.value)
        print(
            "发送飞控一键起飞命令："
            f"target={args.height_cm} cm, baseline_alt_fused={takeoff_baseline_fused:.1f} cm"
        )
        fc.take_off(args.height_cm)
        _wait_for_takeoff_start(fc, args, takeoff_baseline_fused)
        _wait_for_takeoff_height(fc, args)

        max_drift, max_height_error = _hold_with_fc_optical_flow(fc, args)
        print(
            f"{args.hover_s:g} 秒定点测试完成：最大水平漂移={max_drift:.1f} cm，"
            f"最大高度误差={max_height_error:.1f} cm。"
        )
        mission_ok = True
        result = 0
    except KeyboardInterrupt:
        print("收到 Ctrl+C，中止测试并请求降落。")
        result = 130
    except Exception as exc:
        print(f"测试失败：{type(exc).__name__}: {exc}")
        result = 1
    finally:
        if flight_owned:
            landed_ok = _land_and_wait_for_lock(fc, args)
        try:
            fc.close()
        except Exception as exc:
            print(f"关闭飞控连接时出现异常：{exc}")

    if mission_ok and not landed_ok:
        return 1
    return result


if __name__ == "__main__":
    raise SystemExit(main())
