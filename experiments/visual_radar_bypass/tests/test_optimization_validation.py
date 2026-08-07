import numpy as np

from FlightController.Solutions.Safety import (
    Command,
    FlightStatus,
    RadarFieldConfig,
    RadarObstacleField,
    SafetyArbiter,
    SafetyConfig,
)
from experiments.visual_radar_bypass.benchmark_bypass import (
    benchmark_planners,
    run_offline_validation,
)
from experiments.visual_radar_bypass.diagnostics import (
    ExperimentDiagnosticsTracker,
)
from experiments.visual_radar_bypass.parameter_registry import (
    ExperimentLoggingConfig,
    ExperimentSafetyConfig,
    ParameterSource,
    build_parameter_registry,
    safety_parameter_registry,
)
from experiments.visual_radar_bypass.smooth_sidestep import (
    SmoothSidestepConfig,
)


def test_offline_validation_has_no_threshold_oscillation():
    result = run_offline_validation()

    assert result["frames"] == 40
    assert result["state_switches"] == 0
    assert result["side_switches"] == 0
    assert result["safety_arbiter_overrides"] == 0
    assert result["max_abs_delta_vx_cm_s"] == 0.0
    assert result["max_abs_delta_vy_cm_s"] == 0.0
    assert result["max_abs_delta_yaw_rate_deg_s"] == 0.0


def test_benchmark_reports_same_candidate_statistics():
    results = benchmark_planners(iterations=30, warmup=4, seed=7)

    assert set(results) == {
        "legacy_basic",
        "legacy_forward_recovery",
        "smooth_sidestep",
        "circular_tube_bypass",
    }
    for stats in results.values():
        assert stats["samples"] == 30
        assert 0.0 < stats["p50_us"] <= stats["p95_us"]
        assert stats["mean_us"] > 0.0


def test_parameter_registry_has_provenance_and_separate_safety_rows():
    rows = build_parameter_registry(
        SmoothSidestepConfig(),
        ExperimentSafetyConfig(),
        ExperimentLoggingConfig(),
    )

    assert len(rows) == len({row["parameter"] for row in rows})
    assert {row["source"] for row in rows} <= {
        source.value for source in ParameterSource
    }
    by_name = {row["parameter"]: row for row in rows}
    assert by_name["shift_forward_speed_cm_s"]["value"] == 0.0
    assert (
        by_name["shift_forward_speed_cm_s"]["source"]
        == ParameterSource.UNVERIFIED_TUNING.value
    )
    assert by_name["obstacle_stop_distance_cm"]["value"] == 80.0
    assert (
        by_name["obstacle_stop_distance_cm"]["source"]
        == ParameterSource.EXISTING_PROJECT.value
    )
    safety_rows = safety_parameter_registry(rows)
    assert safety_rows
    assert all(row["safety_sensitive"] for row in safety_rows)


def test_safety_80cm_is_surface_x_and_preserves_lateral_command():
    field = RadarObstacleField(
        RadarFieldConfig(
            body_x_half_cm=25.0,
            body_y_half_cm=25.0,
            forward_corridor_half_width_cm=75.0,
        )
    ).update(
        np.asarray([[79.0, -40.0], [95.0, -41.0]], dtype=float),
        now_s=1.0,
    )
    arbiter = SafetyArbiter(
        SafetyConfig(
            require_fc=False,
            require_hold_pos_mode=False,
            require_radar=False,
            obstacle_stop_distance_cm=80.0,
        )
    )

    result = arbiter.filter(
        Command(8.0, 10.0, 0.0, 2.0, "planned"),
        flight=FlightStatus(),
        radar_connected=True,
        radar_age_s=0.0,
        radar_field=field,
    )

    assert result.nearest_forward_obstacle_cm == 79.0
    assert result.command.vx_cm_s == 0.0
    assert result.command.vy_cm_s == 10.0
    assert result.command.yaw_rate_deg_s == 2.0


def test_diagnostics_tracker_emits_transition_and_safety_events():
    tracker = ExperimentDiagnosticsTracker()
    desired = Command(14.0, -6.0, 0.0, 2.0, "visual")
    planned = Command(0.0, 10.0, 0.0, 2.0, "smooth")
    planner_diagnostics = {
        "state": "shift_left",
        "previous_state": "normal",
        "transition_reason": "confirmed_obstacle_encounter",
        "encounter_id": 1,
        "selected_side": "left",
        "selected_side_reason": "obstacle_right_select_left",
        "side_locked": True,
        "observation_valid": True,
        "obstacle_surface_x_cm": 79.0,
        "obstacle_center_x_cm": 80.0,
        "obstacle_center_y_cm": -40.0,
        "fallback_id": None,
        "fallback_reason": None,
    }

    payload, events = tracker.observe(
        frame_id=3,
        now_s=1.2,
        dt_s=0.1,
        planner_elapsed_us=25.0,
        desired=desired,
        planned=planned,
        safe=planned,
        final=planned,
        planner_diagnostics=planner_diagnostics,
        safety_state="OK",
        safety_reasons=[],
        decision_reason="ok",
        nearest_forward_cm=79.0,
        raw_radar_point_count=4,
        radar_point_count=4,
    )

    assert payload["desired"]["vx_cm_s"] == 14.0
    assert payload["planned"]["vx_cm_s"] == 0.0
    assert payload["final"]["vy_cm_s"] == 10.0
    assert payload["planner_elapsed_us"] == 25.0
    assert {event.event for event in events} == {"side_lock", "encounter_start"}

    stopped = Command(0.0, 0.0, 0.0, 0.0, "side_blocked")
    _, override_events = tracker.observe(
        frame_id=4,
        now_s=1.3,
        dt_s=0.1,
        planner_elapsed_us=24.0,
        desired=desired,
        planned=planned,
        safe=stopped,
        final=stopped,
        planner_diagnostics=planner_diagnostics,
        safety_state="LIMITED",
        safety_reasons=["left_side_blocked"],
        decision_reason="ok",
        nearest_forward_cm=79.0,
        raw_radar_point_count=4,
        radar_point_count=4,
    )
    assert [event.event for event in override_events] == [
        "safety_override_start"
    ]
