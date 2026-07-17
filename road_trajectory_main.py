"""Trajectory-point road-following entry point.

This program reuses the proven camera, NPU, recording, safety, takeoff, and
landing pipeline from :mod:`road_follow_main`, but selects the controller that
flies from the camera centre toward successive road trajectory points.
"""

from __future__ import annotations

import sys

import road_follow_main


DEFAULT_ARGUMENTS = [
    "--road-controller",
    "trajectory-point",
    "--max-vx-cm-s",
    "10",
    "--max-vy-cm-s",
    "8",
    "--max-yaw-rate-deg-s",
    "10",
]


def build_argv(argv: list[str] | None = None) -> list[str]:
    # User-supplied options come last, so argparse lets an explicit value
    # override these conservative trajectory-program defaults.
    return [*DEFAULT_ARGUMENTS, *(sys.argv[1:] if argv is None else argv)]


def main(argv: list[str] | None = None) -> None:
    road_follow_main.main(build_argv(argv))


if __name__ == "__main__":
    main()
