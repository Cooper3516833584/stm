"""Purple-target pursuit and payload mission for the fused static-route entry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math

from FlightController.Solutions.Safety import Command


class PurpleTargetMissionState(str, Enum):
    ROAD_SEARCH = "road_search"
    TARGET_CLEARANCE = "target_clearance"
    TARGET_APPROACH = "target_approach"
    HIGH_HOVER = "high_hover"
    DESCEND_60 = "descend_60"
    LOW_CALIBRATE = "low_calibrate"
    LOW_HOVER = "low_hover"
    RELEASE_PENDING = "release_pending"
    POST_RELEASE_WAIT = "post_release_wait"
    ASCEND_100 = "ascend_100"
    ABORT_RECOVERY = "abort_recovery"
    FAILSAFE_HOLD = "failsafe_hold"
    COMPLETE = "complete"
    ABORTED = "aborted"


@dataclass(frozen=True)
class PurpleTargetMissionConfig:
    high_planar_speed_cm_s: float = 13.2
    low_planar_speed_cm_s: float = 5.0
    target_position_kp_s: float = 0.5
    camera_ground_width_cm_at_reference: float = 130.0
    camera_reference_altitude_cm: float = 100.0
    camera_width_px: float = 640.0
    offset_filter_tau_s: float = 0.12
    offset_filter_max_rate_px_s: float = 500.0
    max_planar_accel_cm_s2: float = 36.0
    acquire_confirm_frames: int = 3
    clearance_confirm_frames: int = 3
    reach_confirm_frames: int = 3
    high_reach_x_px: float = 30.0
    high_reach_y_px: float = 40.0
    low_reach_x_px: float = 18.0
    low_reach_y_px: float = 24.0
    high_hover_s: float = 2.0
    low_hover_s: float = 1.0
    low_calibrate_timeout_s: float = 6.0
    target_altitude_cm: float = 60.0
    return_altitude_cm: float = 100.0
    max_vz_cm_s: float = 10.0
    altitude_kp_s: float = 0.5
    altitude_tolerance_cm: float = 5.0
    altitude_confirm_frames: int = 3
    altitude_phase_timeout_s: float = 12.0
    target_loss_timeout_s: float = 2.0
    approach_timeout_s: float = 30.0
    post_release_wait_s: float = 1.0


@dataclass(frozen=True)
class PurpleTargetMissionDecision:
    desired: Command
    state: PurpleTargetMissionState
    mission_owns_command: bool
    guidance_usable: bool
    preserve_guidance_yaw: bool


class PurpleTargetMissionController:
    """Select road or target guidance without owning radar or FC I/O."""

    TERMINAL_TARGET_STATES = frozenset(
        {
            PurpleTargetMissionState.HIGH_HOVER,
            PurpleTargetMissionState.DESCEND_60,
            PurpleTargetMissionState.LOW_CALIBRATE,
            PurpleTargetMissionState.LOW_HOVER,
            PurpleTargetMissionState.RELEASE_PENDING,
        }
    )
    TARGET_REQUIRED_STATES = frozenset(
        {
            PurpleTargetMissionState.TARGET_CLEARANCE,
            PurpleTargetMissionState.TARGET_APPROACH,
            *TERMINAL_TARGET_STATES,
        }
    )

    def __init__(self, config: PurpleTargetMissionConfig | None = None) -> None:
        self.config = config or PurpleTargetMissionConfig()
        self.state = PurpleTargetMissionState.ROAD_SEARCH
        self.previous_state = self.state
        self.transition_reason = "initialized"
        self.payload_released = False
        self._state_started_s = 0.0
        self._mission_started_s: float | None = None
        self._target_lost_since_s: float | None = None
        self._last_target_capture_s = 0.0
        self._acquire_count = 0
        self._clearance_count = 0
        self._reach_count = 0
        self._height_count = 0
        self._filtered_x_px: float | None = None
        self._filtered_y_px: float | None = None
        self._last_filter_s: float | None = None
        self._last_control_s: float | None = None
        self._limited_vx_cm_s = 0.0
        self._limited_vy_cm_s = 0.0
        self._target_error_x_cm: float | None = None
        self._target_error_y_cm: float | None = None
        self._planar_speed_limit_cm_s = 0.0
        self._disable_target_requested = False
        self._disable_target_consumed = False
        self._release_requested = False

    @property
    def mission_active(self) -> bool:
        return self.state not in {
            PurpleTargetMissionState.ROAD_SEARCH,
            PurpleTargetMissionState.COMPLETE,
            PurpleTargetMissionState.ABORTED,
        }

    @property
    def target_required(self) -> bool:
        return self.state in self.TARGET_REQUIRED_STATES

    def update(
        self,
        *,
        now_s: float,
        road_desired: Command,
        target,
        target_stale: bool,
        planner_state: str,
        radar_fresh: bool,
        altitude_cm: float | None,
        obstacle_conflict: bool = False,
    ) -> PurpleTargetMissionDecision:
        now = float(now_s)
        planner_normal = str(planner_state) == "normal"
        target_valid = self._target_valid(target, target_stale)
        new_target = self._is_new_target(target)

        if self.state == PurpleTargetMissionState.ROAD_SEARCH:
            self._update_acquisition(target_valid, new_target)
            if self._acquire_count >= max(1, self.config.acquire_confirm_frames):
                self._mission_started_s = now
                self._transition(
                    PurpleTargetMissionState.TARGET_CLEARANCE,
                    "target_confirmed",
                    now,
                )
            else:
                return self._road_decision(road_desired)

        if self._approach_timed_out(now):
            self._abort("approach_timeout", now)

        if self.state in self.TERMINAL_TARGET_STATES and (
            not planner_normal or obstacle_conflict
        ):
            self._abort("terminal_obstacle_conflict", now)

        if self.state in self.TARGET_REQUIRED_STATES:
            if not target_valid:
                return self._target_loss_decision(now)
            if (
                self._target_lost_since_s is not None
                and self.state in {
                    PurpleTargetMissionState.HIGH_HOVER,
                    PurpleTargetMissionState.LOW_HOVER,
                }
            ):
                self._state_started_s += now - self._target_lost_since_s
            self._target_lost_since_s = None
            if new_target:
                self._update_filtered_offsets(target, now)

        if self.state == PurpleTargetMissionState.TARGET_CLEARANCE:
            if planner_normal and radar_fresh and not obstacle_conflict:
                self._clearance_count += 1
            else:
                self._clearance_count = 0
            if self._clearance_count >= max(1, self.config.clearance_confirm_frames):
                self._transition(
                    PurpleTargetMissionState.TARGET_APPROACH,
                    "radar_clearance_confirmed",
                    now,
                )
            return self._mission_decision(Command.zero("purple_target:clearance_wait"))

        if self.state == PurpleTargetMissionState.TARGET_APPROACH:
            if not planner_normal or obstacle_conflict:
                self._clearance_count = 0
                self._transition(
                    PurpleTargetMissionState.TARGET_CLEARANCE,
                    "avoidance_active",
                    now,
                )
                return self._mission_decision(Command.zero("purple_target:avoidance_pause"))
            if new_target:
                self._update_reach_count(
                    target,
                    self.config.high_reach_x_px,
                    self.config.high_reach_y_px,
                )
            if self._reach_count >= max(1, self.config.reach_confirm_frames):
                self._transition(PurpleTargetMissionState.HIGH_HOVER, "high_target_centered", now)
                return self._mission_decision(Command.zero("purple_target:high_hover"))
            return self._mission_decision(self._target_command(now, altitude_cm))

        if self.state == PurpleTargetMissionState.HIGH_HOVER:
            if not self._within(target, self.config.high_reach_x_px, self.config.high_reach_y_px):
                self._reach_count = 0
                self._transition(PurpleTargetMissionState.TARGET_APPROACH, "high_hover_drift", now)
                return self._mission_decision(self._target_command(now, altitude_cm))
            if now - self._state_started_s >= self.config.high_hover_s:
                self._height_count = 0
                self._transition(PurpleTargetMissionState.DESCEND_60, "high_hover_complete", now)
            return self._mission_decision(Command.zero("purple_target:high_hover"))

        if self.state == PurpleTargetMissionState.DESCEND_60:
            return self._height_stage(
                now,
                altitude_cm,
                target_cm=self.config.target_altitude_cm,
                next_state=PurpleTargetMissionState.LOW_CALIBRATE,
                complete_reason="descent_complete",
                command_reason="purple_target:descend_60",
            )

        if self.state == PurpleTargetMissionState.LOW_CALIBRATE:
            if new_target:
                self._update_reach_count(
                    target,
                    self.config.low_reach_x_px,
                    self.config.low_reach_y_px,
                )
            if self._reach_count >= max(1, self.config.reach_confirm_frames):
                self._transition(PurpleTargetMissionState.LOW_HOVER, "low_target_centered", now)
                return self._mission_decision(Command.zero("purple_target:low_hover"))
            if now - self._state_started_s >= self.config.low_calibrate_timeout_s:
                self._transition(
                    PurpleTargetMissionState.RELEASE_PENDING,
                    "low_calibrate_timeout_release",
                    now,
                )
                return self._mission_decision(Command.zero("purple_target:release_pending"))
            return self._mission_decision(self._target_command(now, altitude_cm))

        if self.state == PurpleTargetMissionState.LOW_HOVER:
            if not self._within(target, self.config.low_reach_x_px, self.config.low_reach_y_px):
                self._reach_count = 0
                self._transition(PurpleTargetMissionState.LOW_CALIBRATE, "low_hover_drift", now)
                return self._mission_decision(self._target_command(now, altitude_cm))
            if now - self._state_started_s >= self.config.low_hover_s:
                self._transition(PurpleTargetMissionState.RELEASE_PENDING, "low_hover_complete", now)
            return self._mission_decision(Command.zero("purple_target:low_hover"))

        if self.state == PurpleTargetMissionState.RELEASE_PENDING:
            self._release_requested = True
            return self._mission_decision(Command.zero("purple_target:release_pending"))

        if self.state == PurpleTargetMissionState.POST_RELEASE_WAIT:
            if now - self._state_started_s >= self.config.post_release_wait_s:
                self._height_count = 0
                self._transition(PurpleTargetMissionState.ASCEND_100, "release_wait_complete", now)
            return self._mission_decision(Command.zero("purple_target:post_release_wait"))

        if self.state == PurpleTargetMissionState.ASCEND_100:
            return self._height_stage(
                now,
                altitude_cm,
                target_cm=self.config.return_altitude_cm,
                next_state=PurpleTargetMissionState.COMPLETE,
                complete_reason="return_altitude_reached",
                command_reason="purple_target:ascend_100",
            )

        if self.state == PurpleTargetMissionState.ABORT_RECOVERY:
            if altitude_cm is not None and altitude_cm >= (
                self.config.return_altitude_cm - self.config.altitude_tolerance_cm
            ):
                self._transition(PurpleTargetMissionState.ABORTED, "abort_altitude_reached", now)
                return self._road_decision(road_desired)
            if now - self._state_started_s >= self.config.altitude_phase_timeout_s:
                self._transition(PurpleTargetMissionState.FAILSAFE_HOLD, "abort_climb_timeout", now)
                return self._mission_decision(Command.zero("purple_target:abort_climb_timeout"))
            return self._mission_decision(
                self._height_command(altitude_cm, self.config.return_altitude_cm, "purple_target:abort_climb")
            )

        if self.state == PurpleTargetMissionState.FAILSAFE_HOLD:
            return self._mission_decision(Command.zero("purple_target:failsafe_hold"))

        return self._road_decision(road_desired)

    def release_is_authorized(
        self,
        *,
        planner_state: str,
        radar_fresh: bool,
        final_command: Command | None = None,
        obstacle_clear: bool = True,
    ) -> bool:
        command_zero = bool(
            final_command is None
            or (
                abs(final_command.vx_cm_s) <= 1e-9
                and abs(final_command.vy_cm_s) <= 1e-9
                and abs(final_command.vz_cm_s) <= 1e-9
                and abs(final_command.yaw_rate_deg_s) <= 1e-9
            )
        )
        return bool(
            self.state == PurpleTargetMissionState.RELEASE_PENDING
            and self._release_requested
            and str(planner_state) == "normal"
            and radar_fresh
            and command_zero
            and obstacle_clear
        )

    def mark_payload_released(self, now_s: float) -> None:
        if self.state != PurpleTargetMissionState.RELEASE_PENDING or self.payload_released:
            return
        self.payload_released = True
        self._release_requested = False
        self._disable_target_requested = True
        self._transition(PurpleTargetMissionState.POST_RELEASE_WAIT, "payload_released", now_s)

    def consume_disable_target_request(self) -> bool:
        if not self._disable_target_requested or self._disable_target_consumed:
            return False
        self._disable_target_consumed = True
        return True

    def diagnostics(self, now_s: float | None = None) -> dict[str, object]:
        now = float(now_s) if now_s is not None else None
        mission_elapsed = (
            max(0.0, now - self._mission_started_s)
            if now is not None and self._mission_started_s is not None
            else None
        )
        return {
            "state": self.state.value,
            "previous_state": self.previous_state.value,
            "transition_reason": self.transition_reason,
            "mission_active": self.mission_active,
            "mission_elapsed_s": mission_elapsed,
            "acquire_count": self._acquire_count,
            "clearance_count": self._clearance_count,
            "reach_count": self._reach_count,
            "height_count": self._height_count,
            "filtered_offset_x_px": self._filtered_x_px,
            "filtered_offset_y_px": self._filtered_y_px,
            "target_error_x_cm": self._target_error_x_cm,
            "target_error_y_cm": self._target_error_y_cm,
            "planar_speed_limit_cm_s": self._planar_speed_limit_cm_s,
            "limited_vx_cm_s": self._limited_vx_cm_s,
            "limited_vy_cm_s": self._limited_vy_cm_s,
            "payload_released": self.payload_released,
            "target_detection_disable_requested": self._disable_target_requested,
        }

    def parameter_dict(self) -> dict[str, object]:
        return asdict(self.config)

    def _update_acquisition(self, target_valid: bool, new_target: bool) -> None:
        if not new_target:
            return
        self._acquire_count = self._acquire_count + 1 if target_valid else 0

    def _target_loss_decision(self, now: float) -> PurpleTargetMissionDecision:
        if self._target_lost_since_s is None:
            self._target_lost_since_s = now
            self._reset_motion_limits()
        if now - self._target_lost_since_s >= self.config.target_loss_timeout_s:
            self._abort("target_loss_timeout", now)
        return self._mission_decision(Command.zero("purple_target:target_loss_hold"))

    def _height_stage(
        self,
        now: float,
        altitude_cm: float | None,
        *,
        target_cm: float,
        next_state: PurpleTargetMissionState,
        complete_reason: str,
        command_reason: str,
    ) -> PurpleTargetMissionDecision:
        if now - self._state_started_s >= self.config.altitude_phase_timeout_s:
            if self.state == PurpleTargetMissionState.ASCEND_100:
                self._transition(PurpleTargetMissionState.FAILSAFE_HOLD, "return_height_timeout", now)
            else:
                self._abort("height_phase_timeout", now)
            return self._mission_decision(Command.zero(f"{command_reason}:timeout"))
        if altitude_cm is not None and abs(float(altitude_cm) - target_cm) <= self.config.altitude_tolerance_cm:
            self._height_count += 1
        else:
            self._height_count = 0
        if self._height_count >= max(1, self.config.altitude_confirm_frames):
            self._reach_count = 0
            self._transition(next_state, complete_reason, now)
            if next_state in {PurpleTargetMissionState.COMPLETE, PurpleTargetMissionState.ABORTED}:
                return PurpleTargetMissionDecision(
                    Command.zero(f"{command_reason}:complete"),
                    self.state,
                    False,
                    False,
                    True,
                )
            return self._mission_decision(Command.zero(f"{command_reason}:complete"))
        return self._mission_decision(self._height_command(altitude_cm, target_cm, command_reason))

    def _height_command(self, altitude_cm: float | None, target_cm: float, reason: str) -> Command:
        if altitude_cm is None or not math.isfinite(float(altitude_cm)):
            return Command.zero(f"{reason}:altitude_unavailable")
        error_cm = target_cm - float(altitude_cm)
        vz = _clamp(
            self.config.altitude_kp_s * error_cm,
            -self.config.max_vz_cm_s,
            self.config.max_vz_cm_s,
        )
        return Command(0.0, 0.0, vz, 0.0, reason)

    def _target_command(self, now: float, altitude_cm: float | None) -> Command:
        x = self._filtered_x_px
        y = self._filtered_y_px
        if x is None or y is None:
            return Command.zero("purple_target:offset_unavailable")
        reference_altitude = max(1e-6, float(self.config.camera_reference_altitude_cm))
        fallback_altitude = (
            self.config.target_altitude_cm
            if self.state == PurpleTargetMissionState.LOW_CALIBRATE
            else self.config.return_altitude_cm
        )
        altitude = (
            float(altitude_cm)
            if altitude_cm is not None
            and math.isfinite(float(altitude_cm))
            and float(altitude_cm) > 0.0
            else float(fallback_altitude)
        )
        ground_width_cm = (
            float(self.config.camera_ground_width_cm_at_reference)
            * altitude
            / reference_altitude
        )
        cm_per_px = ground_width_cm / max(1e-6, float(self.config.camera_width_px))
        error_x_cm = float(x) * cm_per_px
        error_y_cm = float(y) * cm_per_px
        self._target_error_x_cm = error_x_cm
        self._target_error_y_cm = error_y_cm
        speed_limit = max(
            0.0,
            float(
                self.config.low_planar_speed_cm_s
                if self.state == PurpleTargetMissionState.LOW_CALIBRATE
                else self.config.high_planar_speed_cm_s
            ),
        )
        self._planar_speed_limit_cm_s = speed_limit
        requested_vx, requested_vy = _limit_vector_magnitude(
            self.config.target_position_kp_s * error_x_cm,
            self.config.target_position_kp_s * error_y_cm,
            speed_limit,
        )
        dt = self._control_dt(now)
        vx, vy = _limit_vector_rate(
            requested_vx,
            requested_vy,
            self._limited_vx_cm_s,
            self._limited_vy_cm_s,
            self.config.max_planar_accel_cm_s2,
            dt,
        )
        vx, vy = _limit_vector_magnitude(vx, vy, speed_limit)
        self._limited_vx_cm_s = vx
        self._limited_vy_cm_s = vy
        return Command(vx, vy, 0.0, 0.0, "purple_target:approach")

    def _update_filtered_offsets(self, target, now: float) -> None:
        raw_x = float(target.offset_x_px)
        raw_y = float(target.offset_y_px)
        dt = 0.1 if self._last_filter_s is None else _clamp(now - self._last_filter_s, 0.02, 0.5)
        self._last_filter_s = now
        self._filtered_x_px = _low_pass_limited(
            raw_x,
            self._filtered_x_px,
            self.config.offset_filter_tau_s,
            self.config.offset_filter_max_rate_px_s,
            dt,
        )
        self._filtered_y_px = _low_pass_limited(
            raw_y,
            self._filtered_y_px,
            self.config.offset_filter_tau_s,
            self.config.offset_filter_max_rate_px_s,
            dt,
        )

    def _update_reach_count(self, target, x_limit: float, y_limit: float) -> None:
        self._reach_count = self._reach_count + 1 if self._within(target, x_limit, y_limit) else 0

    @staticmethod
    def _within(target, x_limit: float, y_limit: float) -> bool:
        return bool(
            target is not None
            and getattr(target, "found", False)
            and abs(float(target.offset_x_px)) <= x_limit
            and abs(float(target.offset_y_px)) <= y_limit
        )

    @staticmethod
    def _target_valid(target, stale: bool) -> bool:
        if target is None or stale or not bool(getattr(target, "found", False)):
            return False
        try:
            return math.isfinite(float(target.offset_x_px)) and math.isfinite(float(target.offset_y_px))
        except (TypeError, ValueError):
            return False

    def _is_new_target(self, target) -> bool:
        if target is None:
            return False
        capture_s = float(getattr(target, "capture_time_s", 0.0) or 0.0)
        if capture_s <= 0.0 or capture_s == self._last_target_capture_s:
            return False
        self._last_target_capture_s = capture_s
        return True

    def _approach_timed_out(self, now: float) -> bool:
        return bool(
            self._mission_started_s is not None
            and self.state in {
                PurpleTargetMissionState.TARGET_CLEARANCE,
                PurpleTargetMissionState.TARGET_APPROACH,
            }
            and now - self._mission_started_s >= self.config.approach_timeout_s
        )

    def _abort(self, reason: str, now: float) -> None:
        if self.payload_released:
            self._transition(PurpleTargetMissionState.ASCEND_100, reason, now)
        else:
            self._disable_target_requested = True
            self._transition(PurpleTargetMissionState.ABORT_RECOVERY, reason, now)

    def _transition(self, state: PurpleTargetMissionState, reason: str, now: float) -> None:
        if state == self.state:
            return
        self.previous_state = self.state
        self.state = state
        self.transition_reason = str(reason)
        self._state_started_s = float(now)
        if state not in {
            PurpleTargetMissionState.TARGET_APPROACH,
            PurpleTargetMissionState.LOW_CALIBRATE,
        }:
            self._reset_motion_limits()
        if state in {PurpleTargetMissionState.TARGET_APPROACH, PurpleTargetMissionState.LOW_CALIBRATE}:
            self._reach_count = 0
        if state == PurpleTargetMissionState.RELEASE_PENDING:
            self._release_requested = True
        if state in {PurpleTargetMissionState.COMPLETE, PurpleTargetMissionState.ABORTED}:
            self._disable_target_requested = True

    def _reset_motion_limits(self) -> None:
        """Start a resumed target-control segment from hover, not stale velocity."""

        self._limited_vx_cm_s = 0.0
        self._limited_vy_cm_s = 0.0
        self._planar_speed_limit_cm_s = 0.0
        self._last_control_s = None

    def _control_dt(self, now: float) -> float:
        dt = 0.1 if self._last_control_s is None else _clamp(now - self._last_control_s, 0.02, 0.5)
        self._last_control_s = now
        return dt

    def _mission_decision(self, desired: Command) -> PurpleTargetMissionDecision:
        return PurpleTargetMissionDecision(desired, self.state, True, True, False)

    def _road_decision(self, road_desired: Command) -> PurpleTargetMissionDecision:
        return PurpleTargetMissionDecision(road_desired, self.state, False, False, True)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _limit_rate(target: float, previous: float, max_rate: float, dt: float) -> float:
    limit = max(0.0, float(max_rate)) * max(0.0, float(dt))
    return float(previous) + _clamp(float(target) - float(previous), -limit, limit)


def _limit_vector_magnitude(x: float, y: float, limit: float) -> tuple[float, float]:
    magnitude = math.hypot(float(x), float(y))
    maximum = max(0.0, float(limit))
    if magnitude <= maximum or magnitude <= 1e-9:
        return float(x), float(y)
    scale = maximum / magnitude
    return float(x) * scale, float(y) * scale


def _limit_vector_rate(
    target_x: float,
    target_y: float,
    previous_x: float,
    previous_y: float,
    max_rate: float,
    dt: float,
) -> tuple[float, float]:
    delta_x = float(target_x) - float(previous_x)
    delta_y = float(target_y) - float(previous_y)
    delta = math.hypot(delta_x, delta_y)
    maximum_delta = max(0.0, float(max_rate)) * max(0.0, float(dt))
    if delta <= maximum_delta or delta <= 1e-9:
        return float(target_x), float(target_y)
    scale = maximum_delta / delta
    return (
        float(previous_x) + delta_x * scale,
        float(previous_y) + delta_y * scale,
    )


def _low_pass_limited(
    value: float,
    previous: float | None,
    tau_s: float,
    max_rate: float,
    dt_s: float,
) -> float:
    if previous is None:
        return float(value)
    dt = max(1e-6, float(dt_s))
    alpha = dt / (max(0.0, float(tau_s)) + dt)
    filtered = float(previous) + alpha * (float(value) - float(previous))
    return _limit_rate(filtered, float(previous), max_rate, dt)


__all__ = [
    "PurpleTargetMissionConfig",
    "PurpleTargetMissionController",
    "PurpleTargetMissionDecision",
    "PurpleTargetMissionState",
]
