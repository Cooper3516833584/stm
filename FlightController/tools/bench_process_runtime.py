#!/usr/bin/env python3
"""No-actuation static-route-flight-v1 rate sweep for the STM32MP257 board.

This benchmark deliberately does not import FCConnector or any takeoff/flight
entry point. It opens only the road camera and the two D500 radar serial ports,
then computes the frozen static-route planner and safety-arbiter output in RAM.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def _setup_path() -> None:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", default="10,12,15")
    parser.add_argument("--warmup-s", type=float, default=20.0)
    parser.add_argument("--sample-s", type=float, default=120.0)
    parser.add_argument("--camera-index", type=int, default=7)
    parser.add_argument("--model-npu", default="FlightController/Solutions/model/new_road_seg_v5_final_fp32.nb")
    parser.add_argument("--upper-port", default="/dev/ttySTM4")
    parser.add_argument("--lower-port", default="/dev/ttySTM9")
    parser.add_argument("--radar-timeout-s", type=float, default=0.5)
    parser.add_argument("--record-dir", default="/data/stm_records")
    parser.add_argument("--with-recording", action="store_true")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args(argv)


def _run_rate(runtime, recorder, rate_hz: float, warmup_s: float, sample_s: float):
    from FlightController.Runtime import LoopRateMonitor
    from FlightController.Solutions.Safety import (
        FlightStatus,
        RadarObstacleField,
        SafetyArbiter,
        SafetyConfig,
    )
    from experiments.visual_radar_bypass.static_route_bypass import (
        STATIC_ROUTE_PROFILE_NAME,
        StaticRouteBypassConfig,
        StaticRouteBypassPlanner,
    )
    from experiments.visual_radar_bypass.visual_guidance import (
        FrozenVisualConfig,
        FrozenVisualGuidance,
    )

    period_s = 1.0 / rate_hz
    radar_field = RadarObstacleField()
    visual_config = FrozenVisualConfig()
    guidance = FrozenVisualGuidance(visual_config, process_runtime=runtime)
    planner = StaticRouteBypassPlanner(
        StaticRouteBypassConfig(visual_max_vx_cm_s=visual_config.max_vx_cm_s)
    )
    arbiter = SafetyArbiter(
        SafetyConfig(
            require_fc=False,
            require_hold_pos_mode=False,
            require_unlocked=False,
            require_radar=True,
            radar_timeout_s=runtime.config.radar_timeout_s,
            max_vx_cm_s=visual_config.max_vx_cm_s,
            max_vy_cm_s=visual_config.max_vy_cm_s,
            max_yaw_rate_deg_s=visual_config.max_yaw_rate_deg_s,
            obstacle_stop_distance_cm=80.0,
            obstacle_slow_distance_cm=150.0,
            slow_speed_limit_cm_s=10.0,
            side_stop_distance_cm=45.0,
        )
    )
    last_step_s = None

    def control_step(now_s: float):
        nonlocal last_step_s
        sample = guidance.sample(now_s)
        radar, points = runtime.latest_radar()
        radar_field.update(points, now_s)
        planned = planner.update(
            desired=sample.desired,
            perception=sample.perception,
            radar_field=radar_field,
            now_s=now_s,
        )
        radar_fresh = bool(
            radar is not None
            and radar.connected
            and (radar.point_count == 0 or len(points) == radar.point_count)
            and radar.crc_errors == 0
            and runtime._radar_process is not None
            and runtime._radar_process.is_alive()
            and radar.age_s(now_s) <= runtime.config.radar_timeout_s
        )
        safe = arbiter.filter(
            planned,
            flight=FlightStatus(),
            radar_connected=radar_fresh,
            radar_age_s=radar.age_s(now_s) if radar is not None else None,
            radar_field=radar_field,
            enable_flight=False,
        )
        dt_s = 0.0 if last_step_s is None else min(0.5, max(0.0, now_s - last_step_s))
        last_step_s = now_s
        planner.report_applied_command(safe.command, dt_s, False)
        return sample, radar, planned, safe

    deadline = time.perf_counter() + warmup_s
    while time.perf_counter() < deadline:
        started = time.perf_counter()
        control_step(started)
        _sleep_to_rate(started, period_s)

    monitor = LoopRateMonitor(rate_hz)
    radar_ages = []
    vision_ages = []
    crc_start = 0
    crc_end = 0
    max_parse_buffer = 0
    last_vision_sequence = None
    first_vision_sequence = None
    first_frame_sequence = None
    last_frame_sequence = None
    inference_error_start = None
    inference_error_end = 0
    inference_ms = []
    preprocess_ms = []
    npu_ms = []
    postprocess_ms = []
    cpu0_samples = []
    previous_cpu0 = _read_cpu_times("cpu0")
    last_cpu_sample_s = time.perf_counter()
    rss_start_kib = _runtime_rss_kib(runtime)
    vision_snapshot = None
    radar = None
    loops = 0
    deadline = time.perf_counter() + sample_s
    while time.perf_counter() < deadline:
        started = time.perf_counter()
        sample, radar, planned, safe = control_step(started)
        if started - last_cpu_sample_s >= 0.5:
            current_cpu0 = _read_cpu_times("cpu0")
            utilization = _cpu_utilization(previous_cpu0, current_cpu0)
            previous_cpu0 = current_cpu0
            last_cpu_sample_s = started
            if utilization is not None:
                cpu0_samples.append(utilization)
        vision_snapshot = runtime.latest_vision()
        if sample.perception is not None:
            vision_ages.append(sample.perception_age_s)
        if vision_snapshot is not None:
            if first_vision_sequence is None:
                first_vision_sequence = vision_snapshot.sequence
                inference_error_start = vision_snapshot.error_count
            if vision_snapshot.sequence != last_vision_sequence:
                last_vision_sequence = vision_snapshot.sequence
                inference_ms.append(vision_snapshot.inference_ms)
                preprocess_ms.append(vision_snapshot.preprocess_ms)
                npu_ms.append(vision_snapshot.npu_ms)
                postprocess_ms.append(vision_snapshot.postprocess_ms)
            inference_error_end = vision_snapshot.error_count
            if vision_snapshot.frame_ref is not None:
                if first_frame_sequence is None:
                    first_frame_sequence = vision_snapshot.frame_ref.sequence
                last_frame_sequence = vision_snapshot.frame_ref.sequence
        if radar is not None:
            radar_ages.append(radar.age_s(started))
            crc_end = radar.crc_errors
            if loops == 0:
                crc_start = crc_end
            max_parse_buffer = max(max_parse_buffer, radar.parse_buffer_bytes)
        if recorder is not None:
            recorder.record_radar(
                loop_count=loops,
                now_s=started,
                radar_field=radar_field,
                multi_radar=None,
                radar_age_s=radar.age_s(started) if radar is not None else None,
                radar_connected=bool(
                    radar
                    and radar.connected
                    and radar.age_s(started) <= runtime.config.radar_timeout_s
                ),
                desired=sample.desired,
                safe_command=safe.command,
                decision_reason=safe.state,
                extra={"profile": STATIC_ROUTE_PROFILE_NAME, "planned": planned.as_fc_tuple()},
            )
            recorder.record_command(
                loop_count=loops,
                now_s=started,
                desired=sample.desired,
                safe_command=safe.command,
                decision_reason=safe.state,
                extra={"profile": STATIC_ROUTE_PROFILE_NAME, "planned": planned.as_fc_tuple()},
            )
            if sample.frame is not None and recorder.frame_due(loops, started):
                recorder.record_frame(
                    loop_count=loops,
                    now_s=started,
                    frame=sample.frame,
                    source_time_s=sample.frame_time_s,
                )
        monitor.record(started, time.perf_counter())
        loops += 1
        _sleep_to_rate(started, period_s)
    metrics = asdict(monitor.snapshot())
    period_ms = 1000.0 / rate_hz
    metrics.update(
        {
            "radar_age_p99_ms": _p99_ms(radar_ages),
            "radar_age_max_ms": max(radar_ages, default=float("inf")) * 1000.0,
            "vision_age_p99_ms": _p99_ms(vision_ages),
            "vision_hz": _sequence_rate(first_vision_sequence, last_vision_sequence, sample_s),
            "camera_hz": _sequence_rate(first_frame_sequence, last_frame_sequence, sample_s),
            "inference_error_delta": inference_error_end - (inference_error_start or 0),
            "inference_p50_ms": _percentile(inference_ms, 50.0),
            "inference_p99_ms": _percentile(inference_ms, 99.0),
            "preprocess_p50_ms": _percentile(preprocess_ms, 50.0),
            "npu_p50_ms": _percentile(npu_ms, 50.0),
            "postprocess_p50_ms": _percentile(postprocess_ms, 50.0),
            "cpu0_utilization_p95_pct": _percentile(cpu0_samples, 95.0),
            "rss_start_kib": rss_start_kib,
            "rss_end_kib": _runtime_rss_kib(runtime),
            "crc_error_delta": crc_end - crc_start,
            "max_parse_buffer_bytes": max_parse_buffer,
            "vision_publish_drops": (
                vision_snapshot.publish_drops if vision_snapshot is not None else 0
            ),
            "vision_worker_restarts": runtime.health().vision_restarts,
            "radar_publish_drops": radar.publish_drops if radar is not None else 0,
            "radar_health": list(radar.radar_health) if radar is not None else [],
            "recorder": recorder.stats() if recorder is not None else None,
        }
    )
    metrics["passes_timing"] = bool(
        metrics["achieved_hz"] >= rate_hz * 0.99
        and metrics["work_p99_ms"] <= period_ms * 0.8
        and metrics["deadline_misses"] <= max(0, int(metrics["samples"] * 0.001))
    )
    metrics["passes_radar"] = bool(
        metrics["crc_error_delta"] == 0
        and metrics["max_parse_buffer_bytes"] < 94
        and metrics["radar_age_p99_ms"] <= 150.0
        and metrics["radar_age_max_ms"] < 500.0
    )
    metrics["passes_recording"] = bool(
        recorder is None
        or (
            recorder.stats()["healthy"]
            and recorder.stats()["critical_jobs_dropped"] == 0
            and recorder.stats()["frame_jobs_dropped"] == 0
            and recorder.stats()["radar_jobs_dropped"] == 0
            and recorder.stats()["frame_ref_misses"] == 0
            and recorder.stats()["worker_media_errors"] == 0
        )
    )
    metrics["passes_visual"] = metrics["inference_error_delta"] == 0
    metrics["passes_resources"] = bool(
        (not cpu0_samples or metrics["cpu0_utilization_p95_pct"] <= 80.0)
        and (
            metrics["rss_start_kib"] is None
            or metrics["rss_end_kib"] is None
            or metrics["rss_end_kib"] - metrics["rss_start_kib"] <= 16 * 1024
        )
    )
    metrics["passed"] = bool(
        metrics["passes_timing"]
        and metrics["passes_radar"]
        and metrics["passes_recording"]
        and metrics["passes_visual"]
        and metrics["passes_resources"]
    )
    return metrics


def main(argv=None) -> int:
    _setup_path()
    args = parse_args(argv)
    rates = [float(item.strip()) for item in args.rates.split(",") if item.strip()]
    if not rates or any(rate <= 0.0 for rate in rates):
        raise ValueError("--rates must contain positive values")
    if args.warmup_s < 0.0 or args.sample_s <= 0.0:
        raise ValueError("--warmup-s must be non-negative and --sample-s must be positive")

    from FlightController.Runtime import ProcessRuntime, ProcessRuntimeConfig
    from FlightController.Solutions.SessionRecorder import SessionRecorder, SessionRecorderConfig

    runtime = ProcessRuntime(
        ProcessRuntimeConfig(
            camera_index=args.camera_index,
            npu_model_path=args.model_npu,
            upper_port=args.upper_port,
            lower_port=args.lower_port,
            radar_timeout_s=args.radar_timeout_s,
        )
    )
    recorder = None
    results = []
    try:
        if args.with_recording:
            recorder = SessionRecorder(
                SessionRecorderConfig(
                    root_dir=args.record_dir,
                    mode="process_runtime_benchmark",
                    frame_rate_hz=1.0,
                    radar_rate_hz=10.0,
                    command_rate_hz=10.0,
                    video_fps=5.0,
                    frame_ring_descriptor=runtime.frame_ring_descriptor,
                    metadata={
                        "safety": "NO_FC_IMPORT_NO_ACTUATION",
                        "acceptance_profile": "static-route-flight-v1",
                        "rates": rates,
                    },
                )
            )
            if not recorder.healthy:
                raise RuntimeError("recording benchmark requires a healthy recorder process")
        runtime.start()
        baseline_camera_hz = None
        baseline_vision_hz = None
        for rate in rates:
            result = _run_rate(runtime, recorder, rate, args.warmup_s, args.sample_s)
            if baseline_camera_hz is None:
                baseline_camera_hz = result["camera_hz"]
                baseline_vision_hz = result["vision_hz"]
            result["passes_visual"] = bool(
                result["inference_error_delta"] == 0
                and result["camera_hz"] >= baseline_camera_hz * 0.90
                and result["vision_hz"] >= baseline_vision_hz * 0.90
            )
            result["passed"] = bool(
                result["passes_timing"]
                and result["passes_radar"]
                and result["passes_recording"]
                and result["passes_visual"]
                and result["passes_resources"]
            )
            results.append(result)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    finally:
        runtime.stop_workers()
        if recorder is not None:
            recorder.close()
        runtime.stop()

    passed_rates = [row["target_hz"] for row in results if row["passed"]]
    report = {
        "safety": "NO_FC_IMPORTED_OR_OPENED; NO_UNLOCK; NO_TAKEOFF; NO_MOTOR_COMMAND",
        "acceptance_profile": "static-route-flight-v1",
        "with_recording": args.with_recording,
        "results": results,
        "highest_stable_hz": max(passed_rates) if passed_rates else None,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0 if passed_rates else 2


def _sleep_to_rate(started_s: float, period_s: float) -> None:
    remaining = period_s - (time.perf_counter() - started_s)
    if remaining > 0.0:
        time.sleep(remaining)


def _p99_ms(values) -> float:
    if not values:
        return float("inf")
    return float(np.percentile(np.asarray(values, dtype=float), 99.0) * 1000.0)


def _percentile(values, percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _sequence_rate(first, last, duration_s: float) -> float:
    if first is None or last is None or duration_s <= 0.0:
        return 0.0
    return max(0.0, float(last - first)) / duration_s


def _read_cpu_times(label: str):
    try:
        with open("/proc/stat", "r", encoding="ascii") as handle:
            for line in handle:
                fields = line.split()
                if fields and fields[0] == label:
                    values = [int(value) for value in fields[1:]]
                    idle = values[3] + (values[4] if len(values) > 4 else 0)
                    return idle, sum(values)
    except (OSError, ValueError):
        return None
    return None


def _cpu_utilization(previous, current):
    if previous is None or current is None:
        return None
    idle_delta = current[0] - previous[0]
    total_delta = current[1] - previous[1]
    if total_delta <= 0:
        return None
    return max(0.0, min(100.0, (1.0 - idle_delta / total_delta) * 100.0))


def _runtime_rss_kib(runtime):
    pids = [os.getpid()]
    for process in (runtime._vision_process, runtime._radar_process):
        if process is not None and process.pid is not None:
            pids.append(process.pid)
    values = [_process_rss_kib(pid) for pid in pids]
    values = [value for value in values if value is not None]
    return sum(values) if values else None


def _process_rss_kib(pid: int):
    try:
        fields = Path(f"/proc/{pid}/statm").read_text(encoding="ascii").split()
        pages = int(fields[1])
        return pages * os.sysconf("SC_PAGE_SIZE") // 1024
    except (OSError, ValueError, IndexError, AttributeError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
