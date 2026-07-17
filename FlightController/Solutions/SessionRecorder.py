"""Flight/demo session recorder with non-blocking camera capture.

Command and radar metadata are flushed synchronously so a stopped flight still
has useful diagnostics. Camera encoding is delegated to a bounded background
queue because JPEG/video writes must not stall the control loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import queue
import threading
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
    video_enabled: bool = True
    video_every_n: int = 1
    video_fps: float = 10.0
    video_codec: str = "MJPG"
    frame_queue_size: int = 8
    metadata: dict[str, Any] | None = None


class SessionRecorder:
    def __init__(self, config: SessionRecorderConfig | None = None):
        self.config = config or SessionRecorderConfig()
        self.enabled = bool(self.config.enabled and self.config.root_dir)
        self.session_dir: Path | None = None
        self._frame_dir: Path | None = None
        self._radar_points_dir: Path | None = None
        self._frame_log = None
        self._radar_log = None
        self._command_log = None
        self._video_path: Path | None = None
        self._video_writer = None
        self._video_failed = False
        self._frame_queue: queue.Queue | None = None
        self._frame_thread: threading.Thread | None = None
        self._frame_stop = object()
        self._frame_jobs_queued = 0
        self._frame_jobs_dropped = 0
        self._video_frames_written = 0
        self._keyframes_written = 0
        self._last_drop_warning_s = 0.0
        self._created_wall_time_s = time.time()
        self._created_wall_time_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")

        if not self.enabled:
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(str(self.config.root_dir)) / f"{timestamp}_{self.config.mode}"
        try:
            self._frame_dir = self.session_dir / "frames"
            self._radar_points_dir = self.session_dir / "radar_points"
            self._frame_dir.mkdir(parents=True, exist_ok=True)
            self._radar_points_dir.mkdir(parents=True, exist_ok=True)
            self._frame_log = open(self.session_dir / "frames.jsonl", "a", encoding="utf-8")
            self._radar_log = open(self.session_dir / "radar.jsonl", "a", encoding="utf-8")
            self._command_log = open(self.session_dir / "commands.jsonl", "a", encoding="utf-8")
            self._video_path = self.session_dir / "camera.avi"
            self._frame_queue = queue.Queue(maxsize=max(1, int(self.config.frame_queue_size)))
            self._frame_thread = threading.Thread(
                target=self._frame_writer_task,
                name="session-frame-writer",
                daemon=True,
            )
            self._frame_thread.start()
            self._write_session_manifest()
            logger.info(f"[REC] recording session to {self.session_dir}")
        except OSError as exc:
            logger.warning(f"[REC] recording disabled, cannot create {self.session_dir}: {exc}")
            self.enabled = False
            self.close()

    @property
    def runtime_log_path(self) -> Path | None:
        if not self.enabled or self.session_dir is None:
            return None
        return self.session_dir / "runtime.log"

    def frame_due(self, loop_count: int) -> bool:
        if not self.enabled:
            return False
        jpeg_due = loop_count % max(1, int(self.config.frame_every_n)) == 0
        video_due = bool(
            self.config.video_enabled
            and loop_count % max(1, int(self.config.video_every_n)) == 0
        )
        return bool(jpeg_due or video_due)

    def record_frame(
        self,
        *,
        loop_count: int,
        now_s: float,
        frame: np.ndarray | None,
        label: str = "camera",
        source_time_s: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str | None:
        if not self.enabled or frame is None or self._frame_dir is None or self._frame_queue is None:
            return None

        jpeg_due = loop_count % max(1, int(self.config.frame_every_n)) == 0
        video_due = bool(
            self.config.video_enabled
            and loop_count % max(1, int(self.config.video_every_n)) == 0
        )
        if not jpeg_due and not video_due:
            return None

        filename = f"{label}_{loop_count:06d}_{int(now_s * 1000):013d}.jpg"
        path = self._frame_dir / filename if jpeg_due else None
        job = {
            "loop_count": int(loop_count),
            "now_s": float(now_s),
            "source_time_s": _json_float(source_time_s),
            "frame": np.asarray(frame).copy(),
            "jpeg_path": path,
            "video_due": video_due,
            "extra": dict(extra or {}),
        }
        try:
            self._frame_queue.put_nowait(job)
            self._frame_jobs_queued += 1
        except queue.Full:
            self._frame_jobs_dropped += 1
            if now_s - self._last_drop_warning_s >= 2.0:
                self._last_drop_warning_s = now_s
                logger.warning(
                    "[REC] frame queue full; dropped={} queue_size={}",
                    self._frame_jobs_dropped,
                    self._frame_queue.maxsize,
                )
            return None

        if path is not None:
            return str(path)
        return str(self._video_path) if self._video_path is not None else None

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
        frame_queue = self._frame_queue
        frame_thread = self._frame_thread
        if frame_queue is not None and frame_thread is not None:
            frame_queue.put(self._frame_stop)
            frame_thread.join(timeout=15.0)
            if frame_thread.is_alive():
                logger.warning("[REC] frame writer did not stop within 15s")
            self._frame_thread = None
            self._frame_queue = None

        for handle_name in ("_frame_log", "_radar_log", "_command_log"):
            handle = getattr(self, handle_name, None)
            if handle is not None:
                handle.close()
                setattr(self, handle_name, None)
        if self.session_dir is not None:
            try:
                self._write_session_manifest()
            except OSError as exc:
                logger.warning(f"[REC] failed to finalize session manifest: {exc}")

    def _frame_writer_task(self) -> None:
        assert self._frame_queue is not None
        try:
            while True:
                job = self._frame_queue.get()
                try:
                    if job is self._frame_stop:
                        return
                    self._write_frame_job(job)
                except Exception as exc:
                    logger.warning(f"[REC] frame writer error: {type(exc).__name__}: {exc}")
                finally:
                    self._frame_queue.task_done()
        finally:
            if self._video_writer is not None:
                self._video_writer.release()
                self._video_writer = None

    def _write_frame_job(self, job: dict[str, Any]) -> None:
        frame = np.asarray(job["frame"])
        video_frame_index: int | None = None
        video_written = False
        if bool(job["video_due"]) and self._ensure_video_writer(frame):
            assert self._video_writer is not None
            video_frame_index = self._video_frames_written
            self._video_writer.write(frame)
            self._video_frames_written += 1
            video_written = True

        jpeg_path = job["jpeg_path"]
        keyframe_written = False
        if jpeg_path is not None:
            quality = int(max(1, min(100, self.config.jpeg_quality)))
            keyframe_written = bool(
                cv2.imwrite(
                    str(jpeg_path),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), quality],
                )
            )
            if keyframe_written:
                self._keyframes_written += 1
            else:
                logger.warning(f"[REC] failed to write frame {jpeg_path}")

        if self._frame_log is not None:
            record = {
                "loop": int(job["loop_count"]),
                "time_perf_s": float(job["now_s"]),
                "time_wall_s": time.time(),
                "source_time_perf_s": job["source_time_s"],
                "video_written": video_written,
                "video_frame_index": video_frame_index,
                "keyframe_written": keyframe_written,
                "keyframe_file": str(jpeg_path) if keyframe_written else None,
                "extra": job["extra"],
            }
            self._write_log(self._frame_log, record)

    def _ensure_video_writer(self, frame: np.ndarray) -> bool:
        if self._video_writer is not None:
            return True
        if self._video_failed or self._video_path is None:
            return False
        if frame.ndim != 3 or frame.shape[2] != 3:
            self._video_failed = True
            logger.warning(f"[REC] video disabled for invalid frame shape={frame.shape}")
            return False

        codec = str(self.config.video_codec or "MJPG")[:4].ljust(4)
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(self._video_path),
            cv2.VideoWriter_fourcc(*codec),
            max(0.1, float(self.config.video_fps)),
            (int(width), int(height)),
        )
        if not writer.isOpened():
            writer.release()
            self._video_failed = True
            logger.warning(f"[REC] cannot open video writer {self._video_path} codec={codec!r}")
            return False
        self._video_writer = writer
        logger.info(
            "[REC] video recording started path={} codec={} fps={:.1f} size={}x{}",
            self._video_path,
            codec,
            float(self.config.video_fps),
            width,
            height,
        )
        return True

    def _write_session_manifest(self) -> None:
        if self.session_dir is None:
            return
        _write_json(
            self.session_dir / "session.json",
            {
                "mode": self.config.mode,
                "created_wall_time_s": self._created_wall_time_s,
                "created_wall_time_iso": self._created_wall_time_iso,
                "frame_every_n": self.config.frame_every_n,
                "radar_every_n": self.config.radar_every_n,
                "jpeg_quality": self.config.jpeg_quality,
                "video_enabled": bool(self.config.video_enabled),
                "video_every_n": self.config.video_every_n,
                "video_fps": self.config.video_fps,
                "video_codec": self.config.video_codec,
                "video_file": str(self._video_path) if self._video_path is not None else None,
                "frame_queue_size": self.config.frame_queue_size,
                "frame_jobs_queued": self._frame_jobs_queued,
                "frame_jobs_dropped": self._frame_jobs_dropped,
                "video_frames_written": self._video_frames_written,
                "keyframes_written": self._keyframes_written,
                "metadata": self.config.metadata or {},
            },
        )

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
