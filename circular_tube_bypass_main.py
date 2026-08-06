"""One-command real-flight entry for the circular tube bypass experiment.

Running ``python3 circular_tube_bypass_main.py`` intentionally connects the
flight controller, unlocks and takes off.  The generic experiment module keeps
its sensor-only dry-run defaults.
"""

from __future__ import annotations

import sys

from experiments.visual_radar_bypass.main import main as run_experiment


DEFAULT_FLIGHT_ARGV = (
    "--bypass-planner",
    "legacy",
    "--circular-tube-bypass",
    "--enable-flight",
    "--auto-takeoff",
    "--confirm-visual-radar-flight-test",
    "--takeoff-height-cm",
    "100",
    "--duration-s",
    "60",
)


def build_argv(overrides: list[str] | None = None) -> list[str]:
    """Return flight defaults followed by optional command-line overrides."""
    return [*DEFAULT_FLIGHT_ARGV, *(overrides or [])]


def main(argv: list[str] | None = None) -> None:
    overrides = list(sys.argv[1:] if argv is None else argv)
    run_experiment(build_argv(overrides))


if __name__ == "__main__":
    main()
