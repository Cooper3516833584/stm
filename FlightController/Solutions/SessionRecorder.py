"""Lightweight flight/demo session recorder.

The recorder is intentionally simple and synchronous. It samples camera frames
and radar snapshots at a configurable interval so demo capture does not flood
the SD card during control loops.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
from loguru import logger


@dataclass
class SessionRecorderConfig:
    root_dir: str | None = "/media/sdcard/stm_records"
    enabled: bool = True
    mode: str = "session"
    frame_every_n: int = 10
    radar_every_n: int = 1
    jpeg_quality: int = 85


class SessionRecorder:
    def __init__(self, config: SessionRecorderConfig | None = None):
        self.config = config or SessionRecorderConfig()
        self.enabled = bool(self.config.enabled and self.config.root_dir)
        self.session_dir: Path | None = None
        self._frame_dir: Path | None = None
        self._radar_points_dir: Path | None = None
        self._radar_log = None
        self._command_log = None

        if not self.enabled:
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(str(self.config.root_dir)) / f"{timestamp}_{self.config.mode}"
        try:
            self._frame_dir = self.session_dir / "frames"
            self._radar_points_dir = self.session_dir / "radar_points"
            self._frame_dir.mkdir(parents=True, exist_ok=True)
            self._radar_points_dir.mkdir(parents=True, exist_ok=True)
            self._radar_log = open(self.session_dir / "radar.jsonl", "a", encoding="utf-8")
            self._command_log = open(self.session_dir / "commands.jsonl", "a", encoding="utf-8")
            _write_json(
                self.session_dir / "session.json",
                {
                    "mode": self.config.mode,
                    "created_wall_time_s": time.time(),
                    "created_wall_time_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "frame_every_n": self.config.frame_every_n,
                    "radar_every_n": self.config.radar_every_n,
                    "jpeg_quality": self.config.jpeg_quality,
                },
            )
            logger.info(f"[REC] recording session to {self.session_dir}")
        except OSError as exc:
            logger.warning(f"[REC] recording disabled, cannot create {self.session_dir}: {exc}")
            self.enabled = False
            self.close()

    def record_frame(self, *, loop_count: int, now_s: float, frame: np.ndarray | None, label: str = "camera") -> str | None:
        if not self.enabled or frame is None or self._frame_dir is None:
            return None
        every = max(1, int(self.config.frame_every_n))
        if loop_count % every != 0:
            return None

        filename = f"{label}_{loop_count:06d}_{int(now_s * 1000):013d}.jpg"
        path = self._frame_dir / filename
        quality = int(max(1, min(100, self.config.jpeg_quality)))
        ok = cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            logger.warning(f"[REC] failed to write frame {path}")
            return None
        return str(path)

    def record_radar(
        self,
        *,
        loop_count: int,
        now_s: float,
        radar_field,
        multi_radar=None,
        radar_age_s: float | None = None,
        radar_connected: bool | None = None,
        desired=None,
        safe_command=None,
        decision_reason: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled or self._radar_log is None:
            return
        every = max(1, int(self.config.radar_every_n))
        if loop_count % every != 0:
            return

        points = np.asarray(getattr(radar_field, "points_body_cm", np.empty((0, 2))), dtype=float).reshape(-1, 2)
        raw_points = np.asarray(getattr(radar_field, "raw_points_body_cm", np.empty((0, 2))), dtype=float).reshape(-1, 2)
        points_path = self._write_points(loop_count, now_s, points, raw_points)
        record = {
            "loop": int(loop_count),
            "time_perf_s": float(now_s),
            "time_wall_s": time.time(),
            "radar_connected": radar_connected,
            "radar_age_s": _json_float(radar_age_s),
            "raw_point_count": int(len(raw_points)),
            "point_count": int(len(points)),
            "nearest_forward_cm": _json_float(_safe_call(getattr(radar_field, "nearest_forward_obstacle_cm", None))),
            "points_file": points_path,
            "multi_radar_health": _multi_radar_health(multi_radar, now_s),
            "desired": _command_dict(desired),
            "safe": _command_dict(safe_command),
            "decision_reason": decision_reason,
        }
        if extra:
            record["extra"] = extra
        self._write_log(self._radar_log, record)

    def record_command(
        self,
        *,
        loop_count: int,
        now_s: float,
        desired=None,
        safe_command=None,
        decision_reason: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled or self._command_log is None:
            return
        record = {
            "loop": int(loop_count),
            "time_perf_s": float(now_s),
            "time_wall_s": time.time(),
            "desired": _command_dict(desired),
            "safe": _command_dict(safe_command),
            "decision_reason": decision_reason,
        }
        if extra:
            record["extra"] = extra
        self._write_log(self._command_log, record)

    def close(self) -> None:
        for handle_name in ("_radar_log", "_command_log"):
            handle = getattr(self, handle_name, None)
            if handle is not None:
                handle.close()
                setattr(self, handle_name, None)

    def _write_points(self, loop_count: int, now_s: float, points: np.ndarray, raw_points: np.ndarray) -> str | None:
        if self._radar_points_dir is None:
            return None
        filename = f"radar_{loop_count:06d}_{int(now_s * 1000):013d}.npz"
        path = self._radar_points_dir / filename
        try:
            np.savez_compressed(path, points_body_cm=points, raw_points_body_cm=raw_points)
        except OSError as exc:
            logger.warning(f"[REC] failed to write radar points {path}: {exc}")
            return None
        return str(path)

    @staticmethod
    def _write_log(handle, record: dict[str, Any]) -> None:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _command_dict(command) -> dict[str, Any] | None:
    if command is None:
        return None
    return {
        "vx_cm_s": _json_float(getattr(command, "vx_cm_s", None)),
        "vy_cm_s": _json_float(getattr(command, "vy_cm_s", None)),
        "vz_cm_s": _json_float(getattr(command, "vz_cm_s", None)),
        "yaw_rate_deg_s": _json_float(getattr(command, "yaw_rate_deg_s", None)),
        "reason": getattr(command, "reason", ""),
    }


def _multi_radar_health(multi_radar, now_s: float) -> dict[str, Any] | None:
    if multi_radar is None:
        return None
    try:
        return multi_radar.get_health_snapshot(now_s=now_s)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _safe_call(fn) -> Any:
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        return None


def _json_float(value) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


__all__ = ["SessionRecorder", "SessionRecorderConfig"]
