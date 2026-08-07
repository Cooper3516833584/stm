"""Camera/NPU worker entry point used by :mod:`ProcessRuntime`."""

from __future__ import annotations

import copy
import time

import numpy as np

from .ProcessRuntime import (
    FrameRef,
    VisionSnapshot,
    _SharedArrayRing,
    _send_latest,
    _set_affinity,
)


def vision_worker_main(config, frame_ring_descriptor, output, ready_event, stop_event) -> None:
    """Own camera capture and NPU inference for the lifetime of the worker."""
    _set_affinity(1, bool(config.apply_cpu_affinity))
    ring = _SharedArrayRing.attach(frame_ring_descriptor)
    pipeline = None
    sequence = 0
    frame_sequence = 0
    latest_frame_ref: FrameRef | None = None
    last_frame_identity = None
    last_result_identity = None
    publish_drops = 0
    try:
        # Import only after CPU affinity is installed and inside the spawned
        # child, so vendor NPU/OpenCV state is never inherited from the parent.
        from perception_pipeline import PerceptionPipeline

        pipeline = PerceptionPipeline(
            camera_index=config.camera_index,
            camera_width=config.camera_width,
            camera_height=config.camera_height,
            camera_fps=config.camera_fps,
            model_path=config.model_path,
            npu_model_path=config.npu_model_path,
            inference_backend=config.inference_backend,
            postprocess_mode=config.postprocess_mode,
            instance_selection=config.instance_selection,
            flight_height_m=config.flight_height_m,
            wb_enable=config.wb_enable,
            wb_r=config.wb_r,
            wb_g=config.wb_g,
            wb_b=config.wb_b,
            offset_comp_config=config.offset_comp_config,
        )
        pipeline.start()
        ready_event.set()

        while not stop_event.is_set():
            frame, frame_time_s = pipeline.latest_frame()
            if frame is not None and id(frame) != last_frame_identity:
                array = np.asarray(frame)
                if array.shape == tuple(frame_ring_descriptor["shape"]) and array.dtype == np.uint8:
                    frame_sequence += 1
                    slot, generation = ring.write(frame_sequence, array)
                    latest_frame_ref = FrameRef(
                        slot=slot,
                        generation=generation,
                        sequence=frame_sequence,
                        capture_time_s=float(frame_time_s),
                        shape=tuple(array.shape),
                    )
                last_frame_identity = id(frame)

            result, age_s, stale = pipeline.latest_perception()
            result_identity = id(result) if result is not None else None
            if result is not None and result_identity != last_result_identity:
                sequence += 1
                compact = copy.copy(result)
                if hasattr(compact, "debug_mask"):
                    compact.debug_mask = None
                now_s = time.perf_counter()
                capture_s = max(0.0, now_s - max(0.0, float(age_s)))
                stage_timing = dict(pipeline.yolo.last_stage_timing)
                publish_drops += _send_latest(
                    output,
                    VisionSnapshot(
                        sequence=sequence,
                        capture_time_s=capture_s,
                        completed_time_s=now_s,
                        camera_ok=bool(pipeline.camera_ok),
                        perception=compact,
                        frame_ref=latest_frame_ref,
                        inference_ms=float(pipeline.yolo.last_inference_ms),
                        normalize_ms=float(stage_timing.get("normalize_ms", 0.0)),
                        preprocess_ms=float(stage_timing.get("preprocess_ms", 0.0)),
                        npu_ms=float(stage_timing.get("npu_ms", 0.0)),
                        postprocess_ms=float(stage_timing.get("postprocess_ms", 0.0)),
                        error_count=int(pipeline.yolo.error_count),
                        publish_drops=publish_drops,
                    ),
                )
                last_result_identity = result_identity
            elif stale and result is None:
                time.sleep(0.005)
            else:
                time.sleep(0.002)
    finally:
        ready_event.clear()
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        ring.close()
        try:
            output.close()
        except (OSError, ValueError):
            pass


__all__ = ["vision_worker_main"]
