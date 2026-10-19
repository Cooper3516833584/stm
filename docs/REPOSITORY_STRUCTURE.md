# Repository Structure

> Describes the existing layout. This change adds documentation only; no module,
> model or configuration file was moved.

```text
stm/
├── road_follow_main.py          segmentation road-following entry
├── goal_nav_main.py             dual-radar relative-goal entry
├── perception_pipeline.py        camera thread + inference thread + latest-frame buffers
├── safety_arbiter.py            wrapper over FlightController.Solutions.Safety
├── road_follow_mission.py       compatibility wrapper (RoadFollower)
├── goal_nav_mission.py          compatibility wrapper
├── direction_planner.py         candidate-direction selection
├── local_world_model.py         local map / world model
├── autonomy_hardware.py         dual-radar + FC connection builders
├── autonomy_command.py, autonomy_context.py
├── fleet_bus/                   air node, HC-14 transport, protocol, pose, trace
├── FlightController/            FC protocol stack, components, solutions
├── experiments/                 offline experiments (road_contour_bypass,
│                                visual_radar_bypass)
├── tests/                       hardware integration tests + model generators
├── NPU_*.md                     model conversion, ST Cloud findings, failures
├── PROJECT_OVERVIEW.md          hardware topology and design process
├── PROJECT_STATUS.md            current runtime status (newest sections first)
└── docs/                        architecture, testing, structure (this refresh)
```

## Entry points

| Purpose | Command (on target board) |
|---|---|
| Road following | `python3 road_follow_main.py ...` |
| Radar relative-goal navigation | `python3 goal_nav_main.py ...` |
| Data recording | `python3 record_data.py ...` |

## Where to add code

- Mission/control wrappers → repository root
- Reusable perception/planning → root modules or `FlightController/Solutions/`
- Flight-controller protocol → `FlightController/` (treat as protected; it is
  shared content with `FlightController_stm`)
- Ground link → `fleet_bus/`
- Experiments → `experiments/<name>/` with a report
- Model tooling → `tests/` generators + `NPU_*.md` notes

## Large assets

`FlightController/Solutions/model/` holds ONNX/NB models, calibration NPZ files and
training outputs. They are inference assets, not generated cache; do not delete
them to "clean" the repository. Several files are duplicate blobs (identical
content at different paths).

## Related documents

- `docs/SYSTEM_ARCHITECTURE.md`
- `PROJECT_OVERVIEW.md`, `PROJECT_STATUS.md`, `ALGORITHM_REFERENCE.md`
- `HARDWARE_INTERFACE.md`, `GOAL_NAV_ALGORITHM.md`
- `docs/TESTING.md`
