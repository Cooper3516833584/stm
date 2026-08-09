# Road contour bypass

This is an independent visual-road and physical-radar task. It does not alter
or replace `experiments.visual_radar_bypass`.

## Control ownership

```text
NORMAL       existing FrozenVisualGuidance / target mission -> final Command
ACTIVE PATH  ContourTrajectoryBypassPlanner                  -> final Command
REJOIN       smooth blend                                    -> final Command
                                                            -> direct FC send
```

The entry point does not create an environmental command arbiter and does not
use the guarded sender from the older experiment. `main.py` validates finite
values, applies only the task-local command range, rounds, and calls the FC
velocity API at most once per loop.

## Frozen-path invariant

An obstacle must be a cluster of at least three points and must satisfy the
120 cm activation condition for two consecutive frames. The confirmed cluster,
bypass side, occupancy contour, six quintic Bezier control points, and 121 path
samples are then frozen for the encounter. Active-loop radar frames may update
only obstacle bearing/FOV-exit observations; they cannot regenerate the path.

Diagnostics expose `path_frozen` and `plan_generation_count`. The latter must
remain `1` throughout a normal single-obstacle encounter.

## Real-flight command

```bash
PYTHONPATH=. /usr/local/UFC_venv/bin/python3 -u \
  -m experiments.road_contour_bypass.main \
  --runtime-mode process \
  --loop-hz 10 \
  --duration-s 100 \
  --takeoff-height-cm 100 \
  --hc14-port /dev/serial/by-id/usb-1a86_USB_Serial-if00-port0 \
  --hc14-baudrate 115200 \
  --hc14-connect-timeout-s 5 \
  --enable-flight \
  --auto-takeoff \
  --confirm-road-contour-bypass-flight-test
```

Omit the last three flight options for a live-sensor dry run.

## State machine

```text
NORMAL -> ACQUIRE -> PLAN -> FOLLOW_BYPASS -> FOV_EXIT_CONFIRM
       -> RETURN_TO_ROAD -> REJOIN_BLEND -> NORMAL
                              |              ^
                              +-> WAIT_VISUAL+

PLAN -> PLAN_FAILED (only when no collision-free sampled path is found)
```

Successful plans are written as `plans/bypass_plan_<id>.png` inside the session
recording directory. The image includes the raw cluster, 85 cm occupancy union,
external contour, P0..P5, sampled path, bypass side, and minimum clearance.
