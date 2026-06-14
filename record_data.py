"""Radar + camera raw data recorder — no FC control.

Captures raw radar bin data (both upper/lower D500), radar-local point clouds,
and camera frames to the SD card at /media/sdcard/recordings/.
Runs until Ctrl+C, never sends any command to the flight controller.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger


# ── cli ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record radar + camera data to SD card")
    parser.add_argument("--upper-port", default="/dev/ttySTM4")
    parser.add_argument("--lower-port", default="/dev/ttySTM9")
    parser.add_argument("--camera-index", type=int, default=9,
                        help="cv2.VideoCapture index, default 9 (road cam on OpenSTLinux)")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--output-dir", default="/media/sdcard/recordings",
                        help="Base output directory on SD card")
    parser.add_argument("--loop-hz", type=float, default=10.0,
                        help="Recording loop rate, default 10Hz")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--frame-every-n", type=int, default=10,
                        help="Save camera frame every N loops (1 = every loop)")
    parser.add_argument("--radar-every-n", type=int, default=1,
                        help="Save radar snapshot every N loops (1 = every loop)")
    parser.add_argument("--no-radar", action="store_true",
                        help="Skip radar entirely (camera-only mode)")
    parser.add_argument("--no-camera", action="store_true",
                        help="Skip camera entirely (radar-only mode)")
    parser.add_argument("--single-radar", action="store_true",
                        help="Only open upper radar (skip lower)")
    return parser.parse_args()


# ── recorder ─────────────────────────────────────────────────────────

@dataclass
class Recorder:
    """A self-contained session recorder that writes everything under one
    timestamped directory on the SD card."""

    session_dir: Path
    frame_dir: Path
    radar_bins_dir: Path
    radar_points_dir: Path
    jpeg_quality: int = 85
    frame_every_n: int = 10
    radar_every_n: int = 1
    _radar_log: Any = None
    _frame_count: int = 0
    _radar_count: int = 0

    @classmethod
    def create(cls, output_dir: str, *, jpeg_quality: int = 85,
               frame_every_n: int = 10, radar_every_n: int = 1) -> "Recorder":
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        session_dir = Path(output_dir) / f"{timestamp}_record"
        session_dir.mkdir(parents=True, exist_ok=True)

        frame_dir = session_dir / "frames"
        radar_bins_dir = session_dir / "radar_bins"
        radar_points_dir = session_dir / "radar_points"
        frame_dir.mkdir(exist_ok=True)
        radar_bins_dir.mkdir(exist_ok=True)
        radar_points_dir.mkdir(exist_ok=True)

        radar_log = open(session_dir / "radar.jsonl", "a", encoding="utf-8")

        rec = cls(
            session_dir=session_dir,
            frame_dir=frame_dir,
            radar_bins_dir=radar_bins_dir,
            radar_points_dir=radar_points_dir,
            jpeg_quality=jpeg_quality,
            frame_every_n=frame_every_n,
            radar_every_n=radar_every_n,
            _radar_log=radar_log,
        )

        session_meta = {
            "created_wall_time_s": time.time(),
            "created_wall_time_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "jpeg_quality": jpeg_quality,
            "frame_every_n": frame_every_n,
            "radar_every_n": radar_every_n,
        }
        (session_dir / "session.json").write_text(
            json.dumps(session_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("[REC] session → {}", session_dir)
        return rec

    def write_frame(self, loop_count: int, now_s: float, frame: np.ndarray) -> str | None:
        if loop_count % max(1, self.frame_every_n) != 0:
            return None
        stem = f"frame_{loop_count:07d}_{int(now_s * 1000):013d}"
        path = self.frame_dir / f"{stem}.jpg"
        ok = cv2.imwrite(
            str(path), frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, self.jpeg_quality))],
        )
        if not ok:
            logger.warning("[REC] failed to write frame {}", path)
            return None
        self._frame_count += 1
        return str(path)

    def write_radar(
        self,
        loop_count: int,
        now_s: float,
        *,
        raw_bins: dict[str, np.ndarray],
        points_xy: dict[str, np.ndarray],
        points_body: dict[str, np.ndarray],
        health: dict[str, Any],
    ) -> None:
        if loop_count % max(1, self.radar_every_n) != 0:
            return

        # save raw radar bin arrays (full 1080-bin distance map)
        stem = f"radar_{loop_count:07d}_{int(now_s * 1000):013d}"
        bins_path = None
        if raw_bins:
            bins_path = str(self.radar_bins_dir / f"{stem}_bins.npz")
            np.savez_compressed(self.radar_bins_dir / f"{stem}_bins.npz", **raw_bins)

        # save radar-local XY points
        points_path = None
        if points_xy:
            points_path = str(self.radar_points_dir / f"{stem}_pts.npz")
            np.savez_compressed(self.radar_points_dir / f"{stem}_pts.npz", **points_xy)

        # human-readable metadata line
        record = {
            "loop": loop_count,
            "time_perf_s": now_s,
            "time_wall_s": time.time(),
            "health": health,
            "bins_file": bins_path,
            "points_file": points_path,
            "point_counts": {k: int(v.shape[0]) for k, v in points_body.items()},
        }
        self._radar_log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self._radar_log.flush()
        self._radar_count += 1

    def close(self) -> None:
        if self._radar_log is not None:
            self._radar_log.close()
            self._radar_log = None
        logger.info(
            "[REC] done — {} frames, {} radar snapshots saved to {}",
            self._frame_count, self._radar_count, self.session_dir,
        )


# ── helpers ──────────────────────────────────────────────────────────

def _open_camera(args: argparse.Namespace):
    """Open the USB camera via V4L2."""
    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
    return cap


def _snapshot_radar_raw(radar) -> dict[str, np.ndarray]:
    """Grab raw bin distances and radar-local XY points from one LD_Radar.

    Copies `map.data` (1080-bin distance array, -1 = no return) under the
    radar's lock for minimum contention.
    """
    with radar._lock:
        bin_data = radar.map.data.copy()
    points_xy = radar.get_points_xy_cm()
    return {"bins": bin_data, "points_xy_cm": points_xy}


def _radar_health(radar) -> dict[str, Any]:
    """Quick health snapshot for one radar."""
    return {
        "name": radar.name,
        "connected": radar.connected,
        "running": radar.running,
        "last_frame_age_s": radar.get_last_frame_age_s(),
        "frames_ok": radar._frames_ok_total,
        "crc_errors": radar._crc_errors,
    }


def _wait_for_radar(radar, name: str, timeout_s: float = 10.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if radar.connected and radar.is_fresh(max_age_s=0.5):
            logger.info("[REC] {} radar ready", name)
            return
        time.sleep(0.1)
    raise RuntimeError(f"{name} radar not ready after {timeout_s}s")


# ── main ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── ensure output directory exists ──
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        (out_dir / ".write_test").touch()
        (out_dir / ".write_test").unlink()
    except OSError:
        logger.error("[REC] output dir {} is not writable", out_dir)
        return

    # ── open hardware ──
    cap = None
    multi_radar = None
    radars: list = []

    if not args.no_radar:
        from FlightController.Components import MultiRadar, RadarConfig

        configs = [RadarConfig("upper", 0, (0.0, 0.0), 0.0, port=args.upper_port)]
        if not args.single_radar:
            configs.append(
                RadarConfig(
                    "lower", 1, (0.96, 0.15), 0.0,
                    port=args.lower_port, mount_mirror_y=True,
                )
            )
        multi_radar = MultiRadar(configs)
        multi_radar.start()
        radars = list(multi_radar.radars)

        wait_s = 15.0
        deadline = time.perf_counter() + wait_s
        while time.perf_counter() < deadline:
            if multi_radar.connected and multi_radar.is_fresh(max_age_s=0.5):
                logger.info("[REC] all radars ready")
                break
            time.sleep(0.1)
        else:
            health = multi_radar.get_health_snapshot(max_age_s=0.5)
            logger.warning("[REC] radar(s) not fully ready, starting anyway:\n{}",
                           json.dumps(health, indent=2, default=str))

    if not args.no_camera:
        cap = _open_camera(args)
        if cap is None:
            logger.warning("[REC] camera index {} not available", args.camera_index)
        else:
            logger.info("[REC] camera {} open ({}x{}@{})",
                         args.camera_index, args.camera_width, args.camera_height, args.camera_fps)

    if cap is None and not radars:
        logger.error("[REC] no hardware available — both radar and camera are disabled/unavailable")
        if multi_radar is not None:
            multi_radar.stop()
        return

    # ── create recorder ──
    recorder = Recorder.create(
        str(out_dir),
        jpeg_quality=args.jpeg_quality,
        frame_every_n=args.frame_every_n,
        radar_every_n=args.radar_every_n,
    )

    period_s = 1.0 / max(args.loop_hz, 0.1)

    # graceful shutdown on Ctrl+C
    shutdown_flag = False

    def _on_signal(sig, frame):
        nonlocal shutdown_flag
        logger.info("[REC] received signal {}, stopping...", sig)
        shutdown_flag = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    logger.info("[REC] recording at {:.1f} Hz  |  frame every {}  |  radar every {}  |  {}",
                 args.loop_hz, args.frame_every_n, args.radar_every_n, out_dir)

    last_log_s = 0.0
    loop_count = 0

    try:
        while not shutdown_flag:
            loop_start = time.perf_counter()

            # ── camera ──
            frame = None
            if cap is not None and cap.isOpened():
                ok, frame = cap.read()
                if ok:
                    recorder.write_frame(loop_count, loop_start, frame)

            # ── radar ──
            if radars:
                raw_bins: dict[str, np.ndarray] = {}
                points_xy: dict[str, np.ndarray] = {}
                points_body: dict[str, np.ndarray] = {}
                health_info: dict[str, Any] = {"radars": []}

                for r in radars:
                    snap = _snapshot_radar_raw(r)
                    raw_bins[r.name] = snap["bins"]
                    points_xy[r.name] = snap["points_xy_cm"]
                    points_body[r.name] = r.get_points_body_cm()
                    health_info["radars"].append(_radar_health(r))

                recorder.write_radar(
                    loop_count, loop_start,
                    raw_bins=raw_bins,
                    points_xy=points_xy,
                    points_body=points_body,
                    health=health_info,
                )

            # ── periodic status ──
            if loop_start - last_log_s >= 2.0:
                last_log_s = loop_start
                frame_size = frame.shape if frame is not None else None
                parts = [f"loop={loop_count}"]
                if radars:
                    parts.append("radar_fresh=" + str(
                        all(r.is_fresh(max_age_s=0.5) for r in radars)
                    ))
                if frame_size is not None:
                    parts.append(f"frame={frame_size[1]}x{frame_size[0]}")
                logger.info("[REC] {}", "  ".join(parts))

            loop_count += 1
            elapsed = time.perf_counter() - loop_start
            if elapsed < period_s:
                time.sleep(period_s - elapsed)

    except KeyboardInterrupt:
        logger.info("[REC] Ctrl+C")
    finally:
        recorder.close()
        if multi_radar is not None:
            multi_radar.stop()
        if cap is not None:
            cap.release()
        logger.info("[REC] stopped")


if __name__ == "__main__":
    main()
