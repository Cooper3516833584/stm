# STM32MP257 UAV Companion Computer

Onboard perception, planning and safety-arbitration stack for a UAV whose flight
control is handled by an external Lingxiao (匿名科创) flight controller over a
binary UART link.

This repository is the **companion-computer** side: it reads cameras and two D500
LiDARs, runs segmentation and radar mapping, plans motion, arbitrates safety, and
sends high-level commands to the flight controller. It does not implement the
flight controller's inner attitude/rate loops.

## Project at a glance

| Item | Value |
|---|---|
| Compute board | MYiR MYD-LD25X — STM32MP257 (Cortex-A35), OpenSTLinux |
| Sensors | 2 × LDROBOT D500 LiDAR (upper/lower), 2 × USB cameras |
| Flight controller | Lingxiao (匿名科创), binary UART protocol at 500000 baud |
| Ground link | HC-14 radio through a `BB 33` bridge envelope (FleetBus V1) |
| Main language | Python 3 (control/perception), with NPU tooling docs |

## What it does

- **Road following** — camera frames are segmented (NPU path is experimental, CPU
  fallback exists) and a lane offset drives a heading/speed command.
- **Dual-radar relative-goal navigation** — merged D500 scans build a local polar
  map; candidate-direction tube collision checking selects a heading.
- **Safety arbitration** — a single `SafetyArbiter` gate sits between planning and
  the flight controller; stale or invalid sensor data must not pass through.
- **Ground coordination** — `fleet_bus/` exchanges high-level state and commands
  with the ground station over HC-14.
- **Recording/replay** — session recording of logs, point clouds and media for
  offline analysis.

## Architecture

See [docs/SYSTEM_ARCHITECTURE.md](docs/SYSTEM_ARCHITECTURE.md) for the system
panorama and the road-following data path, both with a component-to-source
evidence table.

Key documents (narrative sources, in order of currency):

- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) — hardware topology and design process
- [PROJECT_STATUS.md](PROJECT_STATUS.md) — latest runtime status; newer sections
  supersede older tuning notes in the same file
- [ALGORITHM_REFERENCE.md](ALGORITHM_REFERENCE.md) — algorithm-by-algorithm source map
- [GOAL_NAV_ALGORITHM.md](GOAL_NAV_ALGORITHM.md) — radar navigation details
- [HARDWARE_INTERFACE.md](HARDWARE_INTERFACE.md) — device and interface notes
- [fleet_bus/README.md](fleet_bus/README.md) — FleetBus air-node protocol

## Validation and limitations

- The NPU acceleration path is **not established**. The NPU findings document
  records that FP32 `.nb` models loaded but were not observed to use
  `/dev/galcore`, and INT8/QDQ conversion did not yet produce a usable `.nb`;
  `PROJECT_STATUS.md` also records an NPU worker RSS growth limitation.
- Performance figures in older planning documents are **targets**, not measured
  results. Do not quote them as achievements.
- `FlightController/` is a content copy of the separate `FlightController_stm`
  repository (21 identical blobs, unrelated Git history). The flight-control
  firmware is not original work of this repository.
- Hardware behaviour was developed against a specific airframe; device paths,
  mounts and calibration values are configuration, not universal constants.
- No root `LICENSE` is present. Third-party components retain their own terms;
  see the audit notes in the project documentation.

## Repository map

```text
road_follow_main.py        road-following mission entry
goal_nav_main.py           dual-radar relative-goal navigation entry
perception_pipeline.py     camera thread + inference thread + latest-frame buffers
safety_arbiter.py          safety gate wrapper (FlightController.Solutions.Safety)
direction_planner.py       candidate-direction selection
local_world_model.py       local map / world model
fleet_bus/                 air node, HC-14 transport, protocol, trace buffer
FlightController/          FC protocol stack, components and solutions
experiments/               offline experiments and optimization reports
tests/                     tests and NPU operator fixtures
NPU_*.md                   model conversion, ST Cloud findings, failure reports
```

## Running

This software is intended to run on the target board with the flight controller
and sensors connected. There is **no single hardware-free demo entry** documented
here; use the offline experiment scripts under `experiments/` and the NPU
conversion notes for host-side work. Running any mission script can command a real
aircraft — review the code and the safety notes first.

## Status

Active research/competition platform. Source of truth for current runtime behaviour:
[PROJECT_STATUS.md](PROJECT_STATUS.md).
