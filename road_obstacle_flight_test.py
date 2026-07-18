"""Guarded real-flight obstacle test with fixed straight-road perception.

This entry intentionally ignores the camera/NPU road result and continuously
supplies a centred vertical trajectory (angle=90 degrees, pixel error=0).  The
aircraft can therefore exercise only the radar bypass and safety path.

Running this file is not sufficient to fly.  Real output additionally requires
all three explicit options::

    --enable-flight --auto-takeoff --confirm-obstacle-flight-test

The shared entry validates those options before any FC or radar is opened.
"""

from __future__ import annotations

import sys

import road_follow_main


DEFAULT_ARGUMENTS = [
    "--road-controller",
    "trajectory-point",
    "--obstacle-flight-test",
    "--enable-radar",
    "--road-bypass-enable",
    "--max-vx-cm-s",
    "10",
    "--max-vy-cm-s",
    "8",
    "--max-yaw-rate-deg-s",
    "10",
    "--takeoff-height-cm",
    "100",
]


def build_argv(argv: list[str] | None = None) -> list[str]:
    return [*DEFAULT_ARGUMENTS, *(sys.argv[1:] if argv is None else argv)]


def main(argv: list[str] | None = None) -> None:
    road_follow_main.main(build_argv(argv))


if __name__ == "__main__":
    main()
