"""Process-isolated flight/demo session recorder.

The public API intentionally matches the former threaded recorder. The parent
only prepares small records and enqueues bounded jobs; JSON flushing, drawing,
JPEG/video encoding and NPZ compression run in a low-priority spawned child.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import multiprocessing as mp
from pathlib import Path
import queue
import time
from typing import Any

import numpy as np
from loguru import logger


_STOP = ("__session_recorder_stop__",)


@dataclass
class SessionRecorderConfig:
    root_dir: str | None = "/data/stm_records"
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
    critical_queue_size: int = 512
    metadata: dict[str, Any] | None = None
    # Fixed-rate scheduling is enabled by entry points once loop_hz is known.
    frame_rate_hz: float | None = None
    radar_rate_hz: float | None = None
    command_rate_hz: float | None = None
    frame_ring_descriptor: dict[str, Any] | None = None


class SessionRecorder:
    def __init__(self, config: SessionRecorderConfig | None = None):
        self.config = config or SessionRecorderConfig()
        self.enabled = bool(self.config.enabled and self.config.root_dir)
        self.session_dir: Path | None = None
        self._ctx = mp.get_context("spawn")
        self._critical_queue = None
        self._media_queue = None
        self._process = None
        self._ready = None
        self._failed = None
        self._worker_counters = None
        self._frame_jobs_queued = 0
        self._frame_jobs_dropped = 0
        self._radar_jobs_dropped = 0
        self._critical_jobs_dropped = 0
        self._last_due_s: dict[str, float] = {}
        self._failure_warned = False
        self._created_wall_time_s = time.time()
        self._created_wall_time_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if not self.enabled:
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(str(self.config.root_dir)) / f"{timestamp}_{self.config.mode}"
        self._critical_queue = self._ctx.Queue(maxsize=max(8, int(self.config.critical_queue_size)))
        self._media_queue = self._ctx.Queue(maxsize=max(1, int(self.config.frame_queue_size)))
        self._ready = self._ctx.Event()
        self._failed = self._ctx.Event()
        self._worker_counters = self._ctx.Array("Q", 4, lock=True)
        self._process = self._ctx.Process(
            target=_recorder_process_main,
            args=(
                self.config,
                str(self.session_dir),
                self._created_wall_time_s,
                self._created_wall_time_iso,
                self._critical_queue,
                self._media_queue,
                self._ready,
                self._failed,
                self._worker_counters,
            ),
            name="session-recorder",
            daemon=True,
        )
        self._process.start()
        if not self._ready.wait(timeout=5.0) or self._failed.is_set() or not self._process.is_alive():
            logger.warning("[REC] recording disabled: recorder process failed to start")
            self.enabled = False
            self.close()
        else:
            logger.info(f"[REC] recording session to {self.session_dir}")

    @property
    def runtime_log_path(self) -> Path | None:
        if not self.enabled or self.session_dir is None:
            return None
        return self.session_dir / "runtime.log"

    @property
    def healthy(self) -> bool:
        return bool(
            self.enabled
            and self._process is not None
            and self._process.is_alive()
            and self._failed is not None
            and not self._failed.is_set()
        )

    def log_sink(self, message) -> None:
        if not self._accepting_jobs():
            return
        text = str(message)
        self._put_critical(("runtime_log", text))

    def frame_due(self, loop_count: int, now_s: float | None = None) -> bool:
        if not self._accepting_jobs():
            return False
        if self.config.frame_rate_hz is not None:
            now_s = time.perf_counter() if now_s is None else float(now_s)
            return self._rate_due("frame", now_s, self.config.frame_rate_hz)
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
        frame,
        label: str = "camera",
        source_time_s: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str | None:
        if not self._accepting_jobs() or frame is None or self.session_dir is None:
            return None
        if self.config.frame_rate_hz is not None:
            jpeg_due = self._rate_due("jpeg", now_s, self.config.frame_rate_hz)
            video_due = bool(
                self.config.video_enabled
                and self._rate_due("video", now_s, max(0.1, self.config.video_fps))
            )
        else:
            jpeg_due = loop_count % max(1, int(self.config.frame_every_n)) == 0
            video_due = bool(
                self.config.video_enabled
                and loop_count % max(1, int(self.config.video_every_n)) == 0
            )
        if not jpeg_due and not video_due:
            return None
        filename = f"{label}_{loop_count:06d}_{int(now_s * 1000):013d}.jpg"
        jpeg_path = str(self.session_dir / "frames" / filename) if jpeg_due else None
        video_path = str(self.session_dir / "camera.avi")
        payload = frame
        # Arrays are copied before enqueue so the producer may immediately
        # reuse its camera buffer. FrameRef objects remain zero-copy references.
        if isinstance(frame, np.ndarray):
            payload = np.asarray(frame).copy()
        job = (
            "frame",
            {
                "loop_count": int(loop_count),
                "now_s": float(now_s),
                "source_time_s": _json_float(source_time_s),
                "frame": payload,
                "jpeg_path": jpeg_path,
                "video_due": video_due,
                "extra": dict(extra or {}),
            },
        )
        if not self._put_media(job):
            self._frame_jobs_dropped += 1
            return None
        self._frame_jobs_queued += 1
        return jpeg_path or video_path

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
        if not self._accepting_jobs() or self.session_dir is None:
            return
        if self.config.radar_rate_hz is not None:
            if not self._rate_due("radar", now_s, self.config.radar_rate_hz):
                return
        elif loop_count % max(1, int(self.config.radar_every_n)) != 0:
            return
        points = np.asarray(getattr(radar_field, "points_body_cm", np.empty((0, 2))), dtype=float).reshape(-1, 2)
        raw_points = np.asarray(getattr(radar_field, "raw_points_body_cm", np.empty((0, 2))), dtype=float).reshape(-1, 2)
        filename = f"radar_{loop_count:06d}_{int(now_s * 1000):013d}.npz"
        points_path = str(self.session_dir / "radar_points" / filename)
        points_queued = self._put_media(
            ("points", points_path, points.astype(np.float32), raw_points.astype(np.float32))
        )
        if not points_queued:
            self._radar_jobs_dropped += 1
            points_path = None
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
        self._put_critical(("jsonl", "radar", record))

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
        if not self._accepting_jobs():
            return
        if self.config.command_rate_hz is not None and not self._rate_due(
            "command", now_s, self.config.command_rate_hz
        ):
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
        self._put_critical(("jsonl", "commands", record))

    def stats(self) -> dict[str, int | bool | None]:
        return {
            "healthy": self.healthy,
            "frame_jobs_queued": self._frame_jobs_queued,
            "frame_jobs_dropped": self._frame_jobs_dropped,
            "radar_jobs_dropped": self._radar_jobs_dropped,
            "critical_jobs_dropped": self._critical_jobs_dropped,
            "critical_queue_depth": _queue_depth(self._critical_queue),
            "media_queue_depth": _queue_depth(self._media_queue),
            "frames_written": _counter_value(self._worker_counters, 0),
            "frame_ref_misses": _counter_value(self._worker_counters, 1),
            "radar_point_files_written": _counter_value(self._worker_counters, 2),
            "worker_media_errors": _counter_value(self._worker_counters, 3),
        }

    def close(self) -> None:
        process = self._process
        if process is not None and process.is_alive():
            self._enqueue_stop(self._critical_queue)
            self._enqueue_stop(self._media_queue)
            process.join(timeout=15.0)
            if process.is_alive():
                logger.warning("[REC] recorder process did not stop within 15s")
                process.terminate()
                process.join(timeout=1.0)
        for q in (self._critical_queue, self._media_queue):
            if q is not None:
                try:
                    q.close()
                    q.join_thread()
                except (OSError, ValueError):
                    pass
        self._process = None

    @staticmethod
    def _enqueue_stop(target_queue) -> None:
        if target_queue is None:
            return
        try:
            target_queue.put(_STOP, timeout=2.0)
        except (queue.Full, OSError, ValueError):
            pass

    def _accepting_jobs(self) -> bool:
        if not self.enabled:
            return False
        if self.healthy:
            return True
        if not self._failure_warned:
            self._failure_warned = True
            logger.warning("[REC] recorder process unavailable; recording degraded")
        return False

    def _put_critical(self, item, *, count_drop: bool = True) -> bool:
        if self._critical_queue is None:
            return False
        try:
            self._critical_queue.put_nowait(item)
            return True
        except queue.Full:
            if count_drop:
                self._critical_jobs_dropped += 1
            return False

    def _put_media(self, item) -> bool:
        if self._media_queue is None:
            return False
        try:
            self._media_queue.put_nowait(item)
            return True
        except queue.Full:
            return False

    def _rate_due(self, key: str, now_s: float, rate_hz: float | None) -> bool:
        if rate_hz is None or rate_hz <= 0.0:
            return False
        last = self._last_due_s.get(key)
        period = 1.0 / float(rate_hz)
        if last is not None and now_s - last < period:
            return False
        self._last_due_s[key] = float(now_s)
        return True


class _RecorderWorker:
    def __init__(self, config, session_dir: Path, created_s: float, created_iso: str, counters):
        self.config = config
        self.session_dir = session_dir
        self.created_s = created_s
        self.created_iso = created_iso
        self.handles: dict[str, Any] = {}
        self.video_writer = None
        self.video_failed = False
        self.video_frames_written = 0
        self.keyframes_written = 0
        self.frame_jobs = 0
        self.points_jobs = 0
        self.frame_ring = None
        self.cv2 = None
        self.counters = counters

    def open(self) -> None:
        import cv2

        self.cv2 = cv2
        (self.session_dir / "frames").mkdir(parents=True, exist_ok=True)
        (self.session_dir / "radar_points").mkdir(parents=True, exist_ok=True)
        for key, filename in (
            ("frames", "frames.jsonl"),
            ("radar", "radar.jsonl"),
            ("commands", "commands.jsonl"),
        ):
            self.handles[key] = open(self.session_dir / filename, "a", encoding="utf-8")
        self.handles["runtime"] = open(self.session_dir / "runtime.log", "a", encoding="utf-8")
        if self.config.frame_ring_descriptor:
            from FlightController.Runtime.ProcessRuntime import _SharedArrayRing
            self.frame_ring = _SharedArrayRing.attach(self.config.frame_ring_descriptor)
        self.write_manifest()

    def handle_critical(self, job) -> None:
        kind = job[0]
        if kind == "jsonl":
            _, stream, record = job
            handle = self.handles[stream]
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
        elif kind == "runtime_log":
            self.handles["runtime"].write(job[1])
            self.handles["runtime"].flush()

    def handle_media(self, job) -> None:
        kind = job[0]
        if kind == "points":
            _, path, points, raw_points = job
            np.savez_compressed(path, points_body_cm=points, raw_points_body_cm=raw_points)
            self.points_jobs += 1
            _counter_add(self.counters, 2)
        elif kind == "frame":
            self._write_frame(job[1])

    def _resolve_frame(self, value):
        if isinstance(value, np.ndarray):
            return value
        if self.frame_ring is None:
            return None
        try:
            frame = self.frame_ring.read(int(value.slot), int(value.generation))
            if frame is None:
                return None
            return frame.reshape(tuple(value.shape))
        except (AttributeError, IndexError, ValueError):
            return None

    def _write_frame(self, job: dict[str, Any]) -> None:
        frame = self._resolve_frame(job["frame"])
        if frame is None:
            _counter_add(self.counters, 1)
            return
        frame = self._draw_diagnostics(frame, job.get("extra") or {})
        video_index = None
        video_written = False
        if job["video_due"] and self._ensure_video(frame):
            video_index = self.video_frames_written
            self.video_writer.write(frame)
            self.video_frames_written += 1
            video_written = True
        jpeg_path = job["jpeg_path"]
        keyframe_written = False
        if jpeg_path is not None:
            cv2 = self.cv2
            assert cv2 is not None
            quality = int(max(1, min(100, self.config.jpeg_quality)))
            keyframe_written = bool(
                cv2.imwrite(jpeg_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            )
            if keyframe_written:
                self.keyframes_written += 1
        record = {
            "loop": job["loop_count"],
            "time_perf_s": job["now_s"],
            "time_wall_s": time.time(),
            "source_time_perf_s": job["source_time_s"],
            "video_written": video_written,
            "video_frame_index": video_index,
            "keyframe_written": keyframe_written,
            "keyframe_file": jpeg_path if keyframe_written else None,
            "extra": job["extra"],
        }
        self.handle_critical(("jsonl", "frames", record))
        self.frame_jobs += 1
        _counter_add(self.counters, 0)

    def _draw_diagnostics(self, frame: np.ndarray, extra: dict[str, Any]) -> np.ndarray:
        lines = _diagnostic_lines(extra)
        if not lines:
            return frame
        cv2 = self.cv2
        assert cv2 is not None
        rendered = np.asarray(frame).copy()
        for index, line in enumerate(lines[:8]):
            y = 20 + index * 18
            cv2.putText(
                rendered,
                line,
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                rendered,
                line,
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return rendered

    def _ensure_video(self, frame: np.ndarray) -> bool:
        if self.video_writer is not None:
            return True
        if self.video_failed or frame.ndim != 3 or frame.shape[2] != 3:
            self.video_failed = True
            return False
        codec = str(self.config.video_codec or "MJPG")[:4].ljust(4)
        cv2 = self.cv2
        assert cv2 is not None
        height, width = frame.shape[:2]
        path = self.session_dir / "camera.avi"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*codec),
            max(0.1, float(self.config.video_fps)), (int(width), int(height)),
        )
        if not writer.isOpened():
            writer.release()
            self.video_failed = True
            return False
        self.video_writer = writer
        return True

    def write_manifest(self) -> None:
        payload = {
            "mode": self.config.mode,
            "created_wall_time_s": self.created_s,
            "created_wall_time_iso": self.created_iso,
            "frame_every_n": self.config.frame_every_n,
            "radar_every_n": self.config.radar_every_n,
            "jpeg_quality": self.config.jpeg_quality,
            "video_enabled": bool(self.config.video_enabled),
            "video_every_n": self.config.video_every_n,
            "video_fps": self.config.video_fps,
            "video_codec": self.config.video_codec,
            "video_file": str(self.session_dir / "camera.avi"),
            "frame_queue_size": self.config.frame_queue_size,
            "frame_rate_hz": self.config.frame_rate_hz,
            "radar_rate_hz": self.config.radar_rate_hz,
            "command_rate_hz": self.config.command_rate_hz,
            "frame_jobs_queued": self.frame_jobs,
            "frame_jobs_dropped": 0,
            "video_frames_written": self.video_frames_written,
            "keyframes_written": self.keyframes_written,
            "radar_point_files_written": self.points_jobs,
            "metadata": self.config.metadata or {},
        }
        (self.session_dir / "session.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )

    def close(self) -> None:
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.write_manifest()
        for handle in self.handles.values():
            handle.close()
        if self.frame_ring is not None:
            self.frame_ring.close()


def _recorder_process_main(
    config, session_dir, created_s, created_iso, critical, media, ready, failed, counters
):
    worker = _RecorderWorker(config, Path(session_dir), created_s, created_iso, counters)
    critical_stop = False
    media_stop = False
    try:
        try:
            import os
            if os.name == "posix":
                os.nice(10)
        except OSError:
            pass
        worker.open()
        ready.set()
        while not (critical_stop and media_stop):
            handled = False
            if not critical_stop:
                try:
                    job = critical.get(timeout=0.02)
                    if job == _STOP:
                        critical_stop = True
                    else:
                        worker.handle_critical(job)
                    handled = True
                except queue.Empty:
                    pass
            if not media_stop:
                try:
                    job = media.get_nowait()
                    if job == _STOP:
                        media_stop = True
                    else:
                        worker.handle_media(job)
                    handled = True
                except queue.Empty:
                    pass
            if not handled:
                time.sleep(0.002)
    except Exception:
        import traceback

        traceback.print_exc()
        _counter_add(counters, 3)
        failed.set()
        ready.set()
    finally:
        try:
            worker.close()
        except Exception:
            failed.set()


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


def _diagnostic_lines(extra: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    road_state = extra.get("road_state")
    if road_state is not None:
        lines.append(f"road={road_state}")
    visual = extra.get("visual")
    if isinstance(visual, dict):
        lines.append(
            "road={} conf={} age_s={}".format(
                visual.get("road_found"), visual.get("confidence"), visual.get("age_s")
            )
        )
    controller = extra.get("controller")
    if isinstance(controller, dict):
        lines.append(
            f"controller={controller.get('state', controller.get('controller_mode'))}"
        )
    bypass = extra.get("tube_obstacle_bypass")
    if isinstance(bypass, dict):
        lines.append(
            "bypass={} side={} target_y={}".format(
                bypass.get("state"), bypass.get("active_bypass_side"), bypass.get("target_y_cm")
            )
        )
    commands = extra.get("commands")
    if isinstance(commands, dict):
        lines.append(
            f"safe={commands.get('safe')} state={commands.get('safety_state')}"
        )
    return lines


def _multi_radar_health(multi_radar, now_s: float) -> dict[str, Any] | None:
    if multi_radar is None:
        return None
    try:
        if isinstance(multi_radar, dict):
            return multi_radar
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
    return value if np.isfinite(value) else None


def _queue_depth(target_queue) -> int | None:
    if target_queue is None:
        return None
    try:
        return int(target_queue.qsize())
    except (AttributeError, NotImplementedError, OSError, ValueError):
        return None


def _counter_add(counters, index: int, amount: int = 1) -> None:
    if counters is None:
        return
    with counters.get_lock():
        counters[index] += int(amount)


def _counter_value(counters, index: int) -> int:
    if counters is None:
        return 0
    with counters.get_lock():
        return int(counters[index])


__all__ = ["SessionRecorder", "SessionRecorderConfig"]
