"""飞控原生光流定点飞行测试：起飞到 1 m，悬停 30 s，然后降落。

本程序只连接 ``FC_Controller``。它不会创建雷达、雷达 SLAM、相机导航或
伴随计算机位置控制对象；水平定点完全由飞控的 HOLD_POS（mode=2）完成。
所有任务高度判断只使用飞控 ``ALT_ADD``（光流模组激光测距）；已禁用不可靠的
``ALT_FU`` 融合高度参与起飞、悬停或安全判断。

为防止误触发，默认只打印任务计划。真实飞行必须显式传入 ``--execute``::

    PYTHONPATH=. python -u FlightController/tools/test_optical_flow_hold.py --execute
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
from datetime import datetime
import math
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any


HOLD_POS_MODE = 2
PROGRAM_MODE = 3
TAKEOFF_COMMAND = (0x10, 0x00, 0x05)
TAKEOFF_MIN_RISE_CM = 8.0
TAKEOFF_MIN_VZ_CM_S = 4.0


DIAGNOSTIC_FIELDS = (
    "record_type",
    "wall_time",
    "capture_monotonic_s",
    "session_elapsed_s",
    "phase",
    "message",
    "update_count",
    "state_monotonic_s",
    "state_age_ms",
    "frame_dt_ms",
    "raw_state_hex",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
    "tilt_deg",
    "roll_rate_deg_s",
    "pitch_rate_deg_s",
    "yaw_rate_deg_s",
    "alt_fused_cm",
    "alt_add_cm",
    "alt_fused_minus_add_cm",
    "alt_add_rate_cm_s",
    "vel_x_cm_s",
    "vel_y_cm_s",
    "vel_z_cm_s",
    "vel_xy_cm_s",
    "pos_x_cm",
    "pos_y_cm",
    "origin_x_cm",
    "origin_y_cm",
    "offset_x_cm",
    "offset_y_cm",
    "drift_cm",
    "dpos_vel_x_cm_s",
    "dpos_vel_y_cm_s",
    "dpos_vel_xy_cm_s",
    "dpos_minus_fused_vel_x_cm_s",
    "dpos_minus_fused_vel_y_cm_s",
    "dpos_window_s",
    "dpos_window_vel_x_cm_s",
    "dpos_window_vel_y_cm_s",
    "dpos_window_vel_xy_cm_s",
    "dpos_window_minus_fused_vel_x_cm_s",
    "dpos_window_minus_fused_vel_y_cm_s",
    "hold_elapsed_s",
    "mean_from_origin_vel_x_cm_s",
    "mean_from_origin_vel_y_cm_s",
    "mean_from_origin_vel_xy_cm_s",
    "battery_v",
    "mode",
    "unlock",
    "flight_state_raw",
    "cid",
    "cmd_0",
    "cmd_1",
)


class _DiagnosticLogger:
    """Persist every FC state frame without blocking the serial receive thread."""

    def __init__(self, path: Path, *, queue_size: int = 8192) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=DIAGNOSTIC_FIELDS,
            extrasaction="ignore",
        )
        self._writer.writeheader()
        self._file.flush()

        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(
            maxsize=max(128, int(queue_size))
        )
        self._state_lock = threading.Lock()
        self._phase = "initializing"
        self._origin: tuple[float, float, float] | None = None
        self._previous_state: tuple[float, float, float, float, float, float, float] | None = None
        self._position_history: deque[tuple[float, float, float]] = deque()
        self._latest_state_record: dict[str, Any] = {}
        self._start_monotonic = time.monotonic()
        self._rows_written = 0
        self._dropped_rows = 0
        self._capture_errors = 0
        self._worker_error: Exception | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._write_loop,
            name="optical-flow-diagnostic-writer",
            daemon=True,
        )
        self._thread.start()
        self.mark("diagnostic logger started")

    @staticmethod
    def _state_value(state: Any, name: str) -> Any:
        field = getattr(state, name, None)
        return getattr(field, "value", "")

    @staticmethod
    def _wall_time() -> str:
        return datetime.now().astimezone().isoformat(timespec="milliseconds")

    def _base_record(self, record_type: str, message: str = "") -> dict[str, Any]:
        now = time.monotonic()
        with self._state_lock:
            phase = self._phase
        return {
            "record_type": record_type,
            "wall_time": self._wall_time(),
            "capture_monotonic_s": f"{now:.6f}",
            "session_elapsed_s": f"{now - self._start_monotonic:.6f}",
            "phase": phase,
            "message": message,
        }

    def _enqueue(self, record: dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self._dropped_rows += 1

    def set_phase(self, phase: str, message: str = "") -> None:
        with self._state_lock:
            self._phase = phase
        self.mark(message or f"phase={phase}")

    def set_origin(self, x_cm: float, y_cm: float) -> None:
        with self._state_lock:
            self._origin = (time.monotonic(), float(x_cm), float(y_cm))
        self.mark(f"hold origin set: x={x_cm:.1f} cm, y={y_cm:.1f} cm")

    def mark(self, message: str) -> None:
        self._enqueue(self._base_record("event", message))

    def capture_state(self, state: Any) -> None:
        """Serial callback boundary: diagnostics must never break FC reception."""
        try:
            self._capture_state(state)
        except Exception as exc:
            self._capture_errors += 1
            self._enqueue(
                self._base_record(
                    "event",
                    f"state capture failed: {type(exc).__name__}: {exc}",
                )
            )

    def _capture_state(self, state: Any) -> None:
        """Copy one immutable snapshot; called in the FC serial listener thread."""
        capture_monotonic = time.monotonic()
        state_monotonic = float(
            getattr(state, "last_update_monotonic", capture_monotonic)
            or capture_monotonic
        )
        roll = float(self._state_value(state, "rol"))
        pitch = float(self._state_value(state, "pit"))
        yaw = float(self._state_value(state, "yaw"))
        alt_fused = float(self._state_value(state, "alt_fused"))
        alt_add = float(self._state_value(state, "alt_add"))
        vel_x = float(self._state_value(state, "vel_x"))
        vel_y = float(self._state_value(state, "vel_y"))
        pos_x = float(self._state_value(state, "pos_x"))
        pos_y = float(self._state_value(state, "pos_y"))

        with self._state_lock:
            phase = self._phase
            origin = self._origin
            previous = self._previous_state
            self._previous_state = (
                state_monotonic,
                pos_x,
                pos_y,
                roll,
                pitch,
                yaw,
                alt_add,
            )

        frame_dt_s: float | None = None
        dpos_vel_x: float | None = None
        dpos_vel_y: float | None = None
        roll_rate: float | None = None
        pitch_rate: float | None = None
        yaw_rate: float | None = None
        alt_add_rate: float | None = None
        if previous is not None:
            frame_dt_s = state_monotonic - previous[0]
            if frame_dt_s > 0:
                dpos_vel_x = (pos_x - previous[1]) / frame_dt_s
                dpos_vel_y = (pos_y - previous[2]) / frame_dt_s
                roll_rate = (roll - previous[3]) / frame_dt_s
                pitch_rate = (pitch - previous[4]) / frame_dt_s
                yaw_delta = (yaw - previous[5] + 180.0) % 360.0 - 180.0
                yaw_rate = yaw_delta / frame_dt_s
                alt_add_rate = (alt_add - previous[6]) / frame_dt_s

        self._position_history.append((state_monotonic, pos_x, pos_y))
        while (
            len(self._position_history) > 2
            and state_monotonic - self._position_history[0][0] > 1.5
        ):
            self._position_history.popleft()
        window_dt_s: float | None = None
        window_vel_x: float | None = None
        window_vel_y: float | None = None
        if self._position_history:
            window_start = self._position_history[0]
            window_dt_s = state_monotonic - window_start[0]
            if window_dt_s >= 0.5:
                window_vel_x = (pos_x - window_start[1]) / window_dt_s
                window_vel_y = (pos_y - window_start[2]) / window_dt_s

        offset_x: float | None = None
        offset_y: float | None = None
        hold_elapsed_s: float | None = None
        mean_origin_vel_x: float | None = None
        mean_origin_vel_y: float | None = None
        if origin is not None:
            offset_x = pos_x - origin[1]
            offset_y = pos_y - origin[2]
            hold_elapsed_s = state_monotonic - origin[0]
            if hold_elapsed_s > 0:
                mean_origin_vel_x = offset_x / hold_elapsed_s
                mean_origin_vel_y = offset_y / hold_elapsed_s

        unlock_field = getattr(state, "unlock", None)
        unlock = bool(getattr(unlock_field, "value", False))
        flight_state = int(getattr(unlock_field, "raw_value", int(unlock)))
        raw_state = bytes(getattr(state, "last_raw_bytes", b""))

        record = self._base_record("state")
        record.update(
            {
                "capture_monotonic_s": f"{capture_monotonic:.6f}",
                "session_elapsed_s": f"{capture_monotonic - self._start_monotonic:.6f}",
                "phase": phase,
                "update_count": int(getattr(state, "update_count", 0)),
                "state_monotonic_s": f"{state_monotonic:.6f}",
                "state_age_ms": f"{max(0.0, capture_monotonic - state_monotonic) * 1000.0:.3f}",
                "frame_dt_ms": "" if frame_dt_s is None else f"{frame_dt_s * 1000.0:.3f}",
                "raw_state_hex": raw_state.hex(" "),
                "roll_deg": roll,
                "pitch_deg": pitch,
                "yaw_deg": yaw,
                "tilt_deg": math.hypot(roll, pitch),
                "roll_rate_deg_s": "" if roll_rate is None else roll_rate,
                "pitch_rate_deg_s": "" if pitch_rate is None else pitch_rate,
                "yaw_rate_deg_s": "" if yaw_rate is None else yaw_rate,
                # ALT_FU is captured for diagnosis only; no flight decision reads it.
                "alt_fused_cm": alt_fused,
                "alt_add_cm": alt_add,
                "alt_fused_minus_add_cm": alt_fused - alt_add,
                "alt_add_rate_cm_s": "" if alt_add_rate is None else alt_add_rate,
                "vel_x_cm_s": vel_x,
                "vel_y_cm_s": vel_y,
                "vel_z_cm_s": float(self._state_value(state, "vel_z")),
                "vel_xy_cm_s": math.hypot(vel_x, vel_y),
                "pos_x_cm": pos_x,
                "pos_y_cm": pos_y,
                "origin_x_cm": "" if origin is None else origin[1],
                "origin_y_cm": "" if origin is None else origin[2],
                "offset_x_cm": "" if offset_x is None else offset_x,
                "offset_y_cm": "" if offset_y is None else offset_y,
                "drift_cm": ""
                if offset_x is None or offset_y is None
                else math.hypot(offset_x, offset_y),
                "dpos_vel_x_cm_s": "" if dpos_vel_x is None else dpos_vel_x,
                "dpos_vel_y_cm_s": "" if dpos_vel_y is None else dpos_vel_y,
                "dpos_vel_xy_cm_s": ""
                if dpos_vel_x is None or dpos_vel_y is None
                else math.hypot(dpos_vel_x, dpos_vel_y),
                "dpos_minus_fused_vel_x_cm_s": ""
                if dpos_vel_x is None
                else dpos_vel_x - vel_x,
                "dpos_minus_fused_vel_y_cm_s": ""
                if dpos_vel_y is None
                else dpos_vel_y - vel_y,
                "dpos_window_s": "" if window_dt_s is None else window_dt_s,
                "dpos_window_vel_x_cm_s": ""
                if window_vel_x is None
                else window_vel_x,
                "dpos_window_vel_y_cm_s": ""
                if window_vel_y is None
                else window_vel_y,
                "dpos_window_vel_xy_cm_s": ""
                if window_vel_x is None or window_vel_y is None
                else math.hypot(window_vel_x, window_vel_y),
                "dpos_window_minus_fused_vel_x_cm_s": ""
                if window_vel_x is None
                else window_vel_x - vel_x,
                "dpos_window_minus_fused_vel_y_cm_s": ""
                if window_vel_y is None
                else window_vel_y - vel_y,
                "hold_elapsed_s": "" if hold_elapsed_s is None else hold_elapsed_s,
                "mean_from_origin_vel_x_cm_s": ""
                if mean_origin_vel_x is None
                else mean_origin_vel_x,
                "mean_from_origin_vel_y_cm_s": ""
                if mean_origin_vel_y is None
                else mean_origin_vel_y,
                "mean_from_origin_vel_xy_cm_s": ""
                if mean_origin_vel_x is None or mean_origin_vel_y is None
                else math.hypot(mean_origin_vel_x, mean_origin_vel_y),
                "battery_v": float(self._state_value(state, "bat")),
                "mode": int(self._state_value(state, "mode")),
                "unlock": int(unlock),
                "flight_state_raw": flight_state,
                "cid": int(self._state_value(state, "cid")),
                "cmd_0": int(self._state_value(state, "cmd_0")),
                "cmd_1": int(self._state_value(state, "cmd_1")),
            }
        )
        with self._state_lock:
            self._latest_state_record = dict(record)
        self._enqueue(record)

    def latest_state_record(self) -> dict[str, Any]:
        with self._state_lock:
            return dict(self._latest_state_record)

    def _write_loop(self) -> None:
        last_flush = time.monotonic()
        try:
            while True:
                record = self._queue.get()
                if record is None:
                    break
                self._writer.writerow(record)
                self._rows_written += 1
                now = time.monotonic()
                if now - last_flush >= 1.0:
                    self._file.flush()
                    last_flush = now
            self._file.flush()
        except Exception as exc:
            self._worker_error = exc

    def close(self) -> None:
        if self._closed:
            return
        self.mark(
            f"diagnostic logger stopping: rows={self._rows_written}, "
            f"dropped={self._dropped_rows}, capture_errors={self._capture_errors}"
        )
        self._closed = True
        if self._thread.is_alive():
            try:
                self._queue.put(None, timeout=2.0)
            except queue.Full as exc:
                raise RuntimeError("诊断日志队列已满，无法正常结束写入线程") from exc
            self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            raise RuntimeError("诊断日志写入线程未能在 10 秒内结束")
        self._file.close()
        if self._worker_error is not None:
            raise RuntimeError(f"诊断日志写入失败：{self._worker_error}")

    @property
    def rows_written(self) -> int:
        return self._rows_written

    @property
    def dropped_rows(self) -> int:
        return self._dropped_rows

    @property
    def capture_errors(self) -> int:
        return self._capture_errors


def _default_diagnostic_path() -> Path:
    repository_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return repository_root / "fc_log" / f"optical_flow_hold_{timestamp}.csv"


def _median(values: list[float]) -> float:
    """Small dependency-free median helper for split Yocto Python images."""
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(float(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


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
        "--diagnostic-log",
        type=Path,
        default=None,
        help="完整逐帧诊断 CSV 路径；默认自动写入 fc_log/optical_flow_hold_时间.csv。",
    )
    parser.add_argument(
        "--height-outlier-confirm-frames",
        type=int,
        default=3,
        help="ALT_ADD 连续越界多少个状态帧才触发安全降落，默认 3 帧。",
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
    print("任务高度源：ALT_ADD（光流模组激光测距）；ALT_FU 已禁用。")
    print("真实执行会自动保存逐帧诊断 CSV 和飞控运行日志；ALT_FU 仅记录，不参与控制判断。")
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


def _height_values(fc: Any) -> tuple[float, float]:
    state = fc.state
    return (
        float(state.alt_add.value),
        float(state.vel_z.value),
    )


def _takeoff_evidence(fc: Any, baseline_add_cm: float) -> dict[str, bool]:
    add_cm, vertical_speed = _height_values(fc)
    return {
        "command": _command_now(fc) == TAKEOFF_COMMAND,
        "airborne_state": _flight_state_raw(fc) >= 2,
        "height_rise": add_cm - baseline_add_cm >= TAKEOFF_MIN_RISE_CM,
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

    add_samples: list[float] = []
    update_count = int(getattr(state, "update_count", 0))
    for _ in range(12):
        update_count = _wait_for_next_state(fc, update_count, timeout_s=1.0)
        add_cm, _vertical_speed = _height_values(fc)
        add_samples.append(add_cm)

    add_cm = _median(add_samples)
    add_span = max(add_samples) - min(add_samples)
    if abs(add_cm) > 30.0 or add_span > 10.0:
        raise RuntimeError(
            "起飞前 ALT_ADD 光流激光测距不是稳定地面值，禁止起飞："
            f"alt_add中值={add_cm:.1f} cm, 范围={min(add_samples):.1f}..{max(add_samples):.1f} cm, "
            f"raw={_raw_state_hex(fc)}"
        )
    print(
        "起飞前检查通过："
        f"battery={battery_v:.2f} V, alt_add={add_cm:.1f} cm, "
        f"roll={state.rol.value:.1f}°, pitch={state.pit.value:.1f}°"
    )
    return add_cm


def _wait_for_unlock(fc: Any, timeout_s: float) -> int:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if not fc.connected:
            raise RuntimeError("等待解锁时飞控断开")
        if bool(fc.state.unlock.value):
            return _flight_state_raw(fc)
        time.sleep(0.05)
    raise RuntimeError("飞控解锁确认超时")


def _wait_for_takeoff_start(fc: Any, args: argparse.Namespace, baseline_add_cm: float) -> None:
    """Require command-state or physical evidence that one-key takeoff started."""
    deadline = time.perf_counter() + args.takeoff_start_timeout_s
    update_count = int(getattr(fc.state, "update_count", 0))
    command_seen = False
    next_status = 0.0
    while time.perf_counter() < deadline:
        update_count = _wait_for_next_state(fc, update_count, timeout_s=1.0)
        add_cm, vertical_speed = _height_values(fc)
        flight_state = _flight_state_raw(fc)
        command = _command_now(fc)
        evidence = _takeoff_evidence(fc, baseline_add_cm)
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
                f"依据={','.join(reasons)}, alt_add={add_cm:.1f} cm, "
                f"vz={vertical_speed:.1f} cm/s, "
                f"flight_state={flight_state}, cmd={command}"
            )
            if not command_seen:
                print("警告：未观察到起飞命令状态，但已检测到实际离地；继续按 ALT_ADD 监控。")
            return

        now = time.perf_counter()
        if now >= next_status:
            print(
                "等待起飞确认："
                f"alt_add={add_cm:.1f} cm, "
                f"vz={vertical_speed:.1f} cm/s, flight_state={flight_state}, cmd={command}"
            )
            next_status = now + args.status_interval_s

    add_cm, vertical_speed = _height_values(fc)
    raise RuntimeError(
        "飞控未确认起飞，已停止继续等待："
        f"takeoff_cmd_seen={command_seen}, alt_add={add_cm:.1f} cm, "
        f"vz={vertical_speed:.1f} cm/s, "
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
    while time.perf_counter() < deadline:
        update_count = _wait_for_next_state(fc, update_count, timeout_s=1.0)
        now = time.perf_counter()
        if not fc.connected:
            raise RuntimeError("起飞过程中飞控断开")
        if not bool(fc.state.unlock.value):
            raise RuntimeError("起飞过程中飞控意外锁定")
        _check_tilt(fc, args.max_tilt_deg)

        add_cm, vertical_speed = _height_values(fc)
        violation = ceiling_guard.observe(add_cm)
        if add_cm > hard_ceiling and ceiling_guard.high_count == 1:
            print(
                "警告：ALT_ADD 出现单帧越界，等待连续状态帧确认："
                f"alt_add={add_cm:.1f} cm, raw={_raw_state_hex(fc)}"
            )
        if violation == "high":
            raise RuntimeError(
                "ALT_ADD 连续超过安全上限："
                f"alt_add={add_cm:.1f} cm > {hard_ceiling:.1f} cm, "
                f"连续帧={ceiling_guard.high_count}, raw={_raw_state_hex(fc)}"
            )

        if add_cm >= minimum_height and abs(vertical_speed) <= 10.0:
            stable_since = stable_since or now
            if now - stable_since >= args.stable_s:
                print(
                    "已到达目标高度："
                    f"alt_add={add_cm:.1f} cm, "
                    f"vz={vertical_speed:.1f} cm/s"
                )
                return
        else:
            stable_since = None

        if now >= next_status:
            print(
                f"起飞中：alt_add={add_cm:.1f} cm, "
                f"vz={vertical_speed:.1f} cm/s, flight_state={_flight_state_raw(fc)}, "
                f"cmd={_command_now(fc)}, mode={fc.state.mode.value}"
            )
            next_status = now + args.status_interval_s

    add_cm, vertical_speed = _height_values(fc)
    raise RuntimeError(
        "起飞高度确认超时："
        f"alt_add={add_cm:.1f} cm, "
        f"vz={vertical_speed:.1f} cm/s, 目标={args.height_cm} cm, "
        f"flight_state={_flight_state_raw(fc)}, cmd={_command_now(fc)}"
    )


def _hold_with_fc_optical_flow(
    fc: Any,
    args: argparse.Namespace,
    diagnostic: _DiagnosticLogger,
) -> tuple[float, float]:
    """Monitor the hover without sending any external position or velocity data."""
    diagnostic.set_phase("hold_transition", "switching to native HOLD_POS")
    _wait_for_mode(fc, HOLD_POS_MODE)
    fc.stablize()
    time.sleep(0.5)

    origin_x = float(fc.state.pos_x.value)
    origin_y = float(fc.state.pos_y.value)
    diagnostic.set_origin(origin_x, origin_y)
    diagnostic.set_phase("hold", "native optical-flow hold started")
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

        add_cm, _vertical_speed = _height_values(fc)
        dx = float(fc.state.pos_x.value) - origin_x
        dy = float(fc.state.pos_y.value) - origin_y
        drift_cm = math.hypot(dx, dy)
        max_drift = max(max_drift, drift_cm)
        max_height_error = max(max_height_error, abs(add_cm - args.height_cm))

        if drift_cm > args.max_drift_cm:
            raise RuntimeError(
                f"光流定点漂移超过限制：{drift_cm:.1f} cm > {args.max_drift_cm:.1f} cm"
            )
        height_violation = height_guard.observe(add_cm)
        if height_violation is not None:
            raise RuntimeError(
                "定点 ALT_ADD 连续越过安全范围："
                f"alt_add={add_cm:.1f} cm, "
                f"方向={height_violation}, 连续帧="
                f"{height_guard.low_count if height_violation == 'low' else height_guard.high_count}, "
                f"raw={_raw_state_hex(fc)}"
            )

        if now >= next_status:
            remaining = max(0.0, deadline - now)
            current_x = float(fc.state.pos_x.value)
            current_y = float(fc.state.pos_y.value)
            vel_x = float(fc.state.vel_x.value)
            vel_y = float(fc.state.vel_y.value)
            latest = diagnostic.latest_state_record()
            dpos_vel_x = latest.get("dpos_window_vel_x_cm_s", "")
            dpos_vel_y = latest.get("dpos_window_vel_y_cm_s", "")
            dpos_text = (
                "n/a"
                if dpos_vel_x == "" or dpos_vel_y == ""
                else f"({float(dpos_vel_x):5.1f},{float(dpos_vel_y):5.1f})"
            )
            print(
                f"定点中：剩余={remaining:4.1f} s, alt_add={add_cm:5.1f} cm, "
                f"pos=({current_x:6.1f},{current_y:6.1f}) cm, "
                f"offset=({dx:5.1f},{dy:5.1f}) cm, drift={drift_cm:5.1f} cm, "
                f"vel=({vel_x:5.1f},{vel_y:5.1f}) cm/s, dpos_vel={dpos_text} cm/s, "
                f"rpy=({float(fc.state.rol.value):4.1f},{float(fc.state.pit.value):4.1f},"
                f"{float(fc.state.yaw.value):5.1f})°, mode={fc.state.mode.value}"
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
            add_cm, _vertical_speed = _height_values(fc)
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
                    f"降落中：alt_add={add_cm:.1f} cm, "
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
    from FlightController import FC_Controller, logger

    diagnostic_path = args.diagnostic_log or _default_diagnostic_path()
    diagnostic = _DiagnosticLogger(diagnostic_path)
    runtime_log_path = diagnostic.path.with_suffix(".runtime.log")
    print(f"逐帧诊断日志：{diagnostic.path}")
    print(f"飞控运行日志：{runtime_log_path}")

    fc: Any = None
    runtime_log_sink: int | None = None
    flight_owned = False
    result = 1
    mission_ok = False
    landed_ok = True
    try:
        runtime_log_sink = logger.add(
            str(runtime_log_path),
            level="DEBUG",
            enqueue=True,
            backtrace=True,
            diagnose=True,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} - {message}",
        )
        fc = FC_Controller()
        diagnostic.set_phase("connecting", "opening FC serial connection")
        print("连接飞控（本程序不会打开任何雷达设备）……")
        fc.start_listen_serial(
            serial_dev=args.port,
            callback=diagnostic.capture_state,
            block_until_connected=True,
            open_timeout_s=args.connection_timeout_s,
        )
        if not fc.wait_for_connection(timeout_s=args.connection_timeout_s):
            raise RuntimeError("飞控连接超时")
        diagnostic.set_phase("preflight", "FC connected; starting preflight checks")
        _preflight(fc, args)

        # One-key takeoff is most reliable in PROGRAM mode. Once the target
        # height is stable, mode=2 hands horizontal hold to the FC optical flow.
        diagnostic.set_phase("program_mode", "requesting PROGRAM mode")
        _wait_for_mode(fc, PROGRAM_MODE)
        print("请求飞控解锁……")
        diagnostic.set_phase("unlock", "requesting FC unlock")
        fc.unlock()
        flight_owned = True
        flight_state = _wait_for_unlock(fc, args.unlock_timeout_s)
        print(
            f"解锁状态已确认：flight_state={flight_state}；"
            f"等待 {args.post_unlock_delay_s:.1f} s 让飞控和电机状态稳定……"
        )
        diagnostic.set_phase("post_unlock_delay")
        time.sleep(args.post_unlock_delay_s)
        if not fc.connected:
            raise RuntimeError("解锁稳定等待期间飞控断开")
        if not bool(fc.state.unlock.value):
            raise RuntimeError("解锁稳定等待期间飞控重新锁定")

        takeoff_baseline_add = float(fc.state.alt_add.value)
        print(
            "发送飞控一键起飞命令："
            f"target={args.height_cm} cm, baseline_alt_add={takeoff_baseline_add:.1f} cm"
        )
        diagnostic.set_phase(
            "takeoff_start",
            f"one-key takeoff requested: target={args.height_cm} cm",
        )
        fc.take_off(args.height_cm)
        _wait_for_takeoff_start(fc, args, takeoff_baseline_add)
        diagnostic.set_phase("takeoff_climb", "takeoff confirmed; monitoring climb")
        _wait_for_takeoff_height(fc, args)

        max_drift, max_height_error = _hold_with_fc_optical_flow(
            fc,
            args,
            diagnostic,
        )
        print(
            f"{args.hover_s:g} 秒定点测试完成：最大水平漂移={max_drift:.1f} cm，"
            f"最大高度误差={max_height_error:.1f} cm。"
        )
        mission_ok = True
        result = 0
    except KeyboardInterrupt:
        logger.warning("[OPTICAL_FLOW_TEST] Mission interrupted by Ctrl+C")
        diagnostic.mark("mission interrupted by Ctrl+C")
        print("收到 Ctrl+C，中止测试并请求降落。")
        result = 130
    except Exception as exc:
        logger.exception(f"[OPTICAL_FLOW_TEST] Mission failed: {type(exc).__name__}: {exc}")
        diagnostic.mark(f"mission failed: {type(exc).__name__}: {exc}")
        print(f"测试失败：{type(exc).__name__}: {exc}")
        result = 1
    finally:
        if flight_owned and fc is not None:
            diagnostic.set_phase("landing", "requesting native FC landing")
            landed_ok = _land_and_wait_for_lock(fc, args)
            diagnostic.mark(f"landing finished: locked={landed_ok}")
        if fc is not None:
            try:
                diagnostic.set_phase("closing", "closing FC connection")
                fc.close()
            except Exception as exc:
                diagnostic.mark(f"FC close failed: {type(exc).__name__}: {exc}")
                print(f"关闭飞控连接时出现异常：{exc}")
        diagnostic.mark(
            f"mission summary: mission_ok={mission_ok}, landed_ok={landed_ok}, "
            f"result={result}"
        )
        if runtime_log_sink is not None:
            try:
                logger.remove(runtime_log_sink)
            except Exception as exc:
                diagnostic.mark(
                    f"runtime log close failed: {type(exc).__name__}: {exc}"
                )
        try:
            diagnostic.close()
            print(
                f"诊断日志已保存：CSV={diagnostic.path}，运行日志={runtime_log_path} "
                f"（记录={diagnostic.rows_written}，丢弃={diagnostic.dropped_rows}，"
                f"采集错误={diagnostic.capture_errors}）"
            )
        except Exception as exc:
            print(f"错误：诊断日志关闭失败：{type(exc).__name__}: {exc}")
            result = 1

    if mission_ok and not landed_ok:
        return 1
    return result


if __name__ == "__main__":
    raise SystemExit(main())
