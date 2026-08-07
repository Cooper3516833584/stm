"""Trajectory-point road-following entry point.

This program reuses the proven camera, NPU, recording, safety, takeoff, and
landing pipeline from :mod:`road_follow_main`, but selects the controller that
flies from the camera centre toward successive road trajectory points.

Normal invocation is the production autonomous-flight entry: model validation,
FC output, automatic takeoff, and camera-only road following are enabled by
default.  Change ``TAKEOFF_HEIGHT_CM`` below to adjust the default height.  Pass
``--dry-run`` or ``--no-fc`` when a non-flight run is intended.

For a camera-free static obstacle-avoidance check, run this entry point with
``--obstacle-test``.  That mode supplies a permanent straight-road perception,
enables the two radars and bypass planner, and forcibly disables FC output.
"""

from __future__ import annotations

import sys

import road_follow_main


# User-editable production takeoff height. The FC accepts 40..500 cm.
TAKEOFF_HEIGHT_CM = 100


DEFAULT_ARGUMENTS = [
    "--road-controller",
    "trajectory-point",
    "--road-instance-selection",
    "highest-confidence",
    "--loop-hz",
    "12",
    "--max-vx-cm-s",
    "45",
    "--max-vy-cm-s",
    "12",
    "--max-yaw-rate-deg-s",
    "40",
    "--trajectory-min-forward-lookahead-px",
    "30",
    "--trajectory-max-forward-lookahead-px",
    "130",
    "--trajectory-lookahead-speed-gain-px-per-cm-s",
    "1.6",
    "--trajectory-max-latency-prediction-px",
    "32",
    "--trajectory-lateral-deadband-px",
    "24",
    "--trajectory-lateral-kp-cm-s-per-px",
    "0.10",
    "--trajectory-tangent-kp-yaw",
    "0.9",
    "--road-angle-deadband-deg",
    "5",
    "--trajectory-max-yaw-accel-deg-s2",
    "100",
    "--trajectory-tangent-filter-tau-s",
    "0.08",
    "--road-angle-filter-max-rate-deg-s",
    "180",
    "--trajectory-target-filter-tau-s",
    "0.10",
    "--road-pixel-filter-max-rate-px-s",
    "800",
    "--trajectory-max-planar-accel-cm-s2",
    "55",
    "--trajectory-degraded-speed-scale",
    "0.95",
    "--trajectory-min-curve-speed-cm-s",
    "28",
    "--trajectory-curvature-slowdown-start-deg",
    "25",
    "--trajectory-curvature-full-slowdown-deg",
    "70",
    "--trajectory-lost-grace-s",
    "0.18",
    "--require-model",
    "--no-radar",
]

DEFAULT_FLIGHT_ARGUMENTS = [
    "--enable-flight",
    "--auto-takeoff",
]

# These modes must retain their existing explicit non-flight/confirmation
# contracts instead of silently inheriting the production auto-takeoff default.
_EXPLICIT_SAFETY_MODE_OPTIONS = {
    "--dry-run",
    "--no-fc",
    "--connect-fc",
    "--obstacle-test",
    "--obstacle-flight-test",
}


def build_argv(argv: list[str] | None = None) -> list[str]:
    # User-supplied options come last, so argparse lets an explicit value
    # override the trajectory-program defaults.
    user_argv = list(sys.argv[1:] if argv is None else argv)
    flight_defaults = (
        []
        if any(option in user_argv for option in _EXPLICIT_SAFETY_MODE_OPTIONS)
        else DEFAULT_FLIGHT_ARGUMENTS
    )
    return [
        *DEFAULT_ARGUMENTS,
        "--takeoff-height-cm",
        str(TAKEOFF_HEIGHT_CM),
        *flight_defaults,
        *user_argv,
    ]


def main(argv: list[str] | None = None) -> None:
    road_follow_main.main(build_argv(argv))


if __name__ == "__main__":
    main()
