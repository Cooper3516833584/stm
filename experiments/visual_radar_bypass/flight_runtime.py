"""Copied guarded preflight, takeoff and landing helpers for this experiment."""

from __future__ import annotations

from dataclasses import dataclass
import time

from loguru import logger


@dataclass(frozen=True)
class FlightRuntimeConfig:
    takeoff_height_cm: int = 100
    post_unlock_delay_s: float = 2.0
    takeoff_timeout_s: float = 25.0
    takeoff_height_tolerance_cm: float = 15.0
    min_takeoff_battery_v: float = 10.5
    takeoff_low_battery_confirm_frames: int = 3
    landing_timeout_s: float = 30.0


def wait_for_radars(multi_radar, timeout_s: float, max_age_s: float) -> None:
    if multi_radar is None:
        raise RuntimeError("two physical radars are required")
    deadline = time.perf_counter() + max(0.0, float(timeout_s))
    while True:
        try:
            if bool(multi_radar.connected) and bool(
                multi_radar.is_fresh(max_age_s=max_age_s)
            ):
                logger.info("[VIS-RADAR] both physical radars connected and fresh")
                return
        except Exception:
            pass
        if time.perf_counter() >= deadline:
            break
        time.sleep(0.05)
    raise RuntimeError("both physical radars must be connected and fresh before test")


def wait_for_visual_road(
    guidance,
    *,
    timeout_s: float = 10.0,
    consecutive_frames: int = 3,
    min_confidence: float = 0.4,
) -> None:
    deadline = time.perf_counter() + max(0.0, float(timeout_s))
    ready_count = 0
    while True:
        perception, _age_s, stale = guidance.latest_perception()
        usable = bool(
            guidance.pipeline.camera_ok
            and not stale
            and perception is not None
            and getattr(perception, "is_road_found", False)
            and float(getattr(perception, "confidence", 0.0)) >= min_confidence
            and len(getattr(perception, "trajectory_points", []) or []) >= 2
        )
        ready_count = ready_count + 1 if usable else 0
        if ready_count >= max(1, int(consecutive_frames)):
            logger.info(
                "[VIS-RADAR] real camera/NPU road perception ready for {} frames",
                ready_count,
            )
            return
        if time.perf_counter() >= deadline:
            break
        time.sleep(0.05)
    raise RuntimeError(
        "real camera/NPU road perception was not continuously valid before test"
    )


def wait_for_fc_mode(fc, target_mode: int, timeout_s: float = 5.0) -> None:
    fc.set_flight_mode(target_mode)
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if not fc.connected:
            raise RuntimeError("FC disconnected while changing flight mode")
        if int(fc.state.mode.value) == target_mode:
            return
        time.sleep(0.05)
    raise RuntimeError(
        f"FC mode change timed out: expected={target_mode}, got={fc.state.mode.value}"
    )


def auto_takeoff(fc, config: FlightRuntimeConfig) -> None:
    if fc is None or not fc.connected:
        raise RuntimeError("FC is not connected; automatic takeoff refused")
    if bool(fc.state.unlock.value):
        raise RuntimeError("FC is already unlocked; automatic takeoff refused")
    battery_v = float(fc.state.bat.value)
    if battery_v <= 1.0:
        raise RuntimeError("FC has not reported a valid battery voltage")
    if battery_v < config.min_takeoff_battery_v:
        raise RuntimeError(
            f"battery too low: {battery_v:.2f}V < {config.min_takeoff_battery_v:.2f}V"
        )

    logger.info("[VIS-RADAR] switching FC to PROGRAM mode")
    wait_for_fc_mode(fc, fc.PROGRAM_MODE)
    fc.unlock()
    unlock_deadline = time.perf_counter() + 5.0
    while time.perf_counter() < unlock_deadline:
        if not fc.connected:
            raise RuntimeError("FC disconnected while waiting for unlock")
        if bool(fc.state.unlock.value):
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("FC unlock confirmation timed out")

    time.sleep(config.post_unlock_delay_s)
    if not fc.connected or not bool(fc.state.unlock.value):
        raise RuntimeError("FC is no longer unlocked before takeoff")

    logger.warning(
        "[VIS-RADAR] requesting real takeoff to {}cm",
        config.takeoff_height_cm,
    )
    fc.take_off(config.takeoff_height_cm)
    deadline = time.perf_counter() + config.takeoff_timeout_s
    minimum_height_cm = (
        config.takeoff_height_cm - config.takeoff_height_tolerance_cm
    )
    low_battery_frames = 0
    while time.perf_counter() < deadline:
        if not fc.connected:
            raise RuntimeError("FC disconnected during takeoff")
        if not bool(fc.state.unlock.value):
            raise RuntimeError("FC locked unexpectedly during takeoff")
        battery_v = float(fc.state.bat.value)
        low_battery_frames = (
            low_battery_frames + 1
            if battery_v < config.min_takeoff_battery_v
            else 0
        )
        if low_battery_frames >= config.takeoff_low_battery_confirm_frames:
            raise RuntimeError("battery stayed below threshold during takeoff")
        altitude_cm = float(fc.state.alt_add.value)
        if altitude_cm >= minimum_height_cm:
            wait_for_fc_mode(fc, fc.HOLD_POS_MODE)
            fc.stablize()
            logger.info("[VIS-RADAR] takeoff complete at {:.1f}cm", altitude_cm)
            return
        time.sleep(0.1)
    raise RuntimeError("takeoff height confirmation timed out")


def land_and_wait_for_lock(fc, config: FlightRuntimeConfig) -> bool:
    try:
        if not fc.connected:
            logger.error("[VIS-RADAR] FC disconnected; use RC to land immediately")
            return False
        if not bool(fc.state.unlock.value):
            return True
        logger.warning("[VIS-RADAR] requesting native in-place landing")
        fc.stablize()
        time.sleep(0.1)
        fc.land()
        deadline = time.perf_counter() + config.landing_timeout_s
        next_request = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            now_s = time.perf_counter()
            if not fc.connected:
                logger.error("[VIS-RADAR] FC disconnected during landing; use RC")
                return False
            if not bool(fc.state.unlock.value):
                logger.info("[VIS-RADAR] landing confirmed and FC locked")
                return True
            if now_s >= next_request:
                fc.land()
                next_request = now_s + 2.0
            time.sleep(0.1)
        logger.error("[VIS-RADAR] landing confirmation timed out; use RC")
        return False
    except Exception as exc:
        logger.exception("[VIS-RADAR] landing failed: {}; use RC", exc)
        return False
