# Testing Guide

> Status: documentation only. **No test was executed** for this change. This
> repository does not document a hardware-free automated suite; most scripts under
> `tests/` require the vehicle.

## Test classes

| Class | Purpose | Where | Safe without hardware? |
|---|---|---|---|
| Model/format generators | Build ONNX/NPZ fixtures for NPU work | `tests/gen_diag_models*.py`, `tests/gen_npz.py` | Host-side only if inputs exist |
| Hardware integration | Radar, FC connection, real-time control, optical flow | `tests/hardware/test_*.py` | **No — connects to FC/radar/motors** |
| Offline radar SLAM | `tests/hardware/test_radar_slam_offline.py` | `tests/hardware/` | Requires recorded data; review first |
| Mission entries | `road_follow_main.py`, `goal_nav_main.py` | root | **No** |

`tests/hardware/` is an on-vehicle directory. Do **not** run it on a development
machine or with propellers/motors connected.

## Static checks (safe)

```bash
python3 -m py_compile <changed files>
python3 -m compileall -q <changed small directory>
git diff --check
```

## What was not run here

- Any entry script: **NOT RUN** (would command a real aircraft).
- `tests/hardware/*`: **NOT RUN** by policy.
- No `pytest` configuration exists at the repository root; do not invent one.

## Rules when adding tests

- Pure-logic tests (frame packing, geometry, unit conversion) should avoid importing
  modules that open serial ports at import time.
- Hardware tests stay under `tests/hardware/` and must be documented as requiring
  the vehicle.
- A syntax check is not a flight test. Record any hardware run in
  `PROJECT_STATUS.md` or an experiment report, with conditions.
