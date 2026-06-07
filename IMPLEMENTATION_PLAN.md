# 无人机自主飞行系统 — 实现方案与进度

**基于**: Codex 实现方案包 (00/02/05/06 号文档) + 当前仓库状态  
**硬件**: MYD-LD25X (STM32MP257) + 凌霄飞控 + 双 D500 雷达 + USB 摄像头  
**最后更新**: 2026-06-07 (新增 NPU 适配 & OS 迁移决策)
**平台变更**: 计划从 Debian 12 迁移至 OpenSTLinux v6.2 以获得 NPU 硬件加速

---

## 1. 项目总体目标

构建一套无人机机载自主飞行系统，实现两类自主任务：

| 模式 | 感知源 | 规划策略 | 输出 |
|------|--------|----------|------|
| **道路视觉循线** | USB 摄像头 → YOLO11-seg → 道路中线 | `RoadFollower` P 控制律 | vx + yaw_rate |
| **相对目标导航** | 双 D500 雷达 → 融合点云 → 障碍物分布 | 候选方向搜索 (−90°~+90°) | vx + yaw_rate |

### 硬性安全约束

1. **默认 dry-run**：所有任务入口默认只打印/记录，`--enable-flight` 才发送飞控指令
2. **不自动解锁/起飞/降落**：这阶段不实现自动起降
3. **统一安全仲裁**：所有飞控指令必须经过 `SafetyArbiter`
4. **Ctrl+C / 异常退出 → 零速**：`finally` 块必须发送零速度指令
5. **首次联调拆桨或固定机体**

### 硬件拓扑

```
MYD-LD25X (STM32MP257, Debian 12 aarch64)
  ├── UART4  (/dev/ttySTM4,  230400)  → 上雷达 D500 (正装)
  ├── UART9  (/dev/ttySTM9,  230400)  → 下雷达 D500 (倒装, Y镜像)
  ├── USB/ACM  (/dev/ttyACM0, 500000) → 凌霄飞控 (二进制协议)
  ├── /dev/video7 (USB 摄像头, 640×480) → 道路路径识别 (间歇性偏蓝/偏青)
  ├── /dev/video9 (USB 摄像头, 640×480) → 障碍物类型识别
  └── eMMC 6.9G (仅剩 930M) + 闪存卡 30G (代码+日志)
```

---

## 2. 系统分层架构

```
┌──────────────────────────────────────────────────────────────┐
│                    入口程序 (entry points)                     │
│                                                               │
│  road_follow_main.py          goal_nav_main.py                │
│  (视觉道路循线)                (雷达相对目标导航)              │
│                                                               │
│  debug/test_fc_connect.py     test_fc_command.py  test_fc_realtime.py  │
│  FlightController/tools/test_radar_avoidance.py  test_dual_radar.py    │
└────────────┬─────────────────────────────────────────────────┘
             │
┌────────────┴─────────────────────────────────────────────────┐
│                    任务层 (Mission)                            │
│                                                               │
│  road_follow_mission.py       goal_nav_mission.py             │
│  (道路跟随状态机)              (目标导航)                      │
│       │                            │                          │
│  RoadFollower.py  [待创建]      RelativeGoalNavigator.py [待] │
│  (像素/角度→yaw_rate)           (候选方向搜索→vx+yaw)         │
└────────────┬─────────────────────────────────────────────────┘
             │
┌────────────┴─────────────────────────────────────────────────┐
│                    感知层 (Perception)                         │
│                                                               │
│  road_perception.py            local_world_model.py           │
│  (YOLO11-seg ONNX → 道路中线)   (雷达点云→过滤→聚类→分类)     │
│       │                            │                          │
│       │                        obstacle_classifier.py [待实现] │
│       │                        (欧氏聚类→wall/pole/block)     │
└────────────┬─────────────────────────────────────────────────┘
             │
┌────────────┴─────────────────────────────────────────────────┐
│                    安全仲裁层 (Safety)                         │
│                                                               │
│  safety_arbiter.py                                            │
│  ┌─ 硬故障检查 ──────────────────────────────────────────┐    │
│  │ 雷达未连接/超时>0.5s | FC非定点模式 | 电压<10V | 倾角>25°│   │
│  │ → HARD_STOP (vx=vy=vz=yaw=0)                          │    │
│  ├─ 障碍物约束 ──────────────────────────────────────────┤    │
│  │ 前方<80cm  → vx=0  (OBSTACLE_STOP)                    │    │
│  │ 前方<150cm → vx≤12 (OBSTACLE_SLOW)                    │    │
│  ├─ 全局钳位 ────────────────────────────────────────────┤    │
│  │ vx≤35 vy≤35 vz≤25 yaw≤30 (可配)                       │    │
│  └───────────────────────────────────────────────────────┘    │
└────────────┬─────────────────────────────────────────────────┘
             │
┌────────────┴─────────────────────────────────────────────────┐
│                    公共类型层 (Shared Types)                   │
│                                                               │
│  autonomy_command.py    — VelocityCommand (vx/vy/vz/yaw,reason)│
│  autonomy_context.py    — SensorHealth / FlightStatus / Obstacle│
│  autonomy_hardware.py   — build_dual_radar/connect_fc/open_camera│
└────────────┬─────────────────────────────────────────────────┘
             │
┌────────────┴─────────────────────────────────────────────────┐
│                    硬件驱动层 (FlightController/)               │
│                                                               │
│  FC_Controller (Base/Protocal/Application/Serial)              │
│  LD_Radar ×2 → MultiRadar → get_obstacle_points_body_cm()    │
│  CameraSource (V4L2)                                          │
│  DeviceResolver (USB VID/PID 自动发现)                        │
│  LocalPlanner / PathPlanner / Radar_SLAM / Navigation (离线)  │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. 数据流详解

### 3.1 道路视觉循线 (Road Follow)

```
摄像头 V4L2 读取
  │  BGR frame (640×480×3, uint8)
  ▼
road_perception.get_road_perception(frame)
  │  _letterbox → _preprocess → ONNX 推理
  │  _decode_yolo_segmentation → mask 生成
  │  _clean_mask → 连通域清理
  │  _extract_centerline_and_intervals → 中线点序列
  │  _compute_pixel_error → pixel_error (px)
  │  _compute_centerline_angle → centerline_angle (deg)
  │  _compute_path_width → path_width_px
  ▼
  RoadPerceptionResult(
    is_road_found, road_state, pixel_error, centerline_angle,
    path_width_px, confidence, branches, selected_branch
  )
  │
  ▼
RoadFollower.update(perception, now_s)
  │  lost:    vx=0 or search yaw
  │  single:  yaw_rate = pixel_error * kp_pixel + angle_error * kp_angle
  │  fork:    按 preference 选 branch → 同理
  │  timeout: vx=0 停车
  ▼
  VelocityCommand(vx, vy=0, vz=0, yaw_rate, reason)

  ──→ 同时: 双雷达 → local_world_model → SafetyArbiter 健康检查
  ──→ SafetyArbiter.filter(desired, context, world)
  ──→ send_fc_command(fc, safe.command, enable_flight)
```

### 3.2 相对目标导航 (Goal Nav)

```
双雷达 1080×2 bin 点云
  │  MultiRadar.get_obstacle_points_body_cm(max_distance_cm=300)
  │  shape=(N, 2), unit=cm, body frame
  ▼
local_world_model.update_from_radar_points(points, now_s)
  │  _within_range:     距离过滤 (>300cm 丢弃)
  │  _remove_body_reflections: 机身屏蔽 (|x|<25 & |y|<25)
  │  classifier.classify_points: 欧氏聚类→wall/pole/block [待实现]
  ▼
  LocalWorldSnapshot(filtered_points, obstacles)

  ──→ world.sector_clearance_cm(angle, 12°)  扇区间隙查询
  ──→ world.nearest_forward_obstacle_cm()     前方最近障碍物
  ──→ world.radar_age_s(now_s)               雷达数据年龄
  ▼
GoalNavMission.update(world)
  │  DirectionPlanner.plan_to_body_goal(goal_x, goal_y, world)
  │    -90°~+90°, 10°步长扫描 sector_clearance
  │    代价 = |angle−goal_angle| + penalty(clearance<阈值)
  │    → 选最小代价方向
  ▼
  VelocityCommand(vx=cruise, vy=0, vz=0, yaw_rate, reason)

  ──→ SafetyArbiter.filter(desired, context, world)
  ──→ send_fc_command(fc, safe.command, enable_flight)
```

---

## 4. 关键参数基准

### 雷达

| 参数 | 值 | 说明 |
|------|-----|------|
| ACC | 3 | 角精度 bins/deg, 总 1080 bin |
| REMAP | 2 | 点扩散半径 bins |
| timeout_time | 0.15s | 点云超时清除 |
| max_distance_cm | 300 | 检测最大距离 |
| body_x_half_cm | 25 | 机身 X 屏蔽半宽 |
| body_y_half_cm | 25 | 机身 Y 屏蔽半宽 |
| forward_corridor_half_width_cm | 50 | 前方走廊半宽 |
| min_obstacle_distance_cm | 10 | 近场噪点过滤 |
| loop_hz | 30 | 主循环频率 |
| radar_timeout_s | 0.5 | 雷达超时阈值 |

### 避障

| 参数 | 值 | 说明 |
|------|-----|------|
| obstacle_stop_distance_cm | 80 | 急停距离 |
| obstacle_slow_distance_cm | 150 | 减速距离 |
| slow_speed_limit_cm_s | 12 | 减速时限速 |
| cruise_speed_cm_s | 20 | 巡航速度 |
| obstacle_clearance_cm | 120 | 候选方向通过阈值 |

### 道路循线

| 参数 | 值 | 说明 |
|------|-----|------|
| image_width | 640 | 输入图像宽度 |
| pixel_kp_yaw | 0.08 | 像素误差→yaw 增益 |
| angle_kp_yaw | 0.4 | 角度误差→yaw 增益 |
| max_yaw_rate_deg_s | 25 | yaw 速率上限 |
| search_yaw_rate_deg_s | 12 | 丢路时搜索 yaw |
| lost_timeout_s | 5.0 | 丢路超时停车 |

---

## 5. 坐标系约定

### 机体坐标系 (Body Frame)

```
        +X (前 / 机头方向)
          ↑
          |
  +Y ←───┼───→ -Y
  (左侧)  |     (右侧)
          ↓
        -X (后)
```

- 雷达 0° = +X 方向, 顺时针递增 (D500 硬件约定)
- yaw_rate 正号沿用 `send_realtime_control_data(..., yaw)` 语义, 底层 `struct.pack("<hhhh", vx, vy, vz, -yaw)`
- 下雷达倒装: Y 轴镜像 (`mount_mirror_y=True`), 先后: 翻转 → 旋转 → 平移

### 图像坐标系 (Road Perception)

```
  (0,0) ────→ x (右, 0°)
    │
    ↓
    y (下)

  角度: 右方=0°, 逆时针为正, 画面竖直向上=90°
  单路直行时 centerline_angle ≈ 90°
```

---

## 6. 实现状态总览

### 已完成 ✅

| 模块 | 状态 | 说明 |
|------|:---:|------|
| **硬件层** | | |
| 上雷达 UART4 接线 | ✅ | `/dev/ttySTM4`, 230400, 已验证 |
| 下雷达 UART9 接线 | ✅ | `/dev/ttySTM9`, 230400, 倒装 mirror_y |
| Y 轴镜像坐标变换 | ✅ | `LD_Radar.mount_mirror_y`, `get_points_body_cm()` 先翻转 |
| 双雷达安装位姿 | ✅ | 上(0,0) 下(0.96, 0.15) cm |
| 凌霄飞控 UART 协议栈 | ✅ | Base/Protocal/Application/Serial, 不可改 |
| **感知层** | | |
| D500 雷达驱动 (LDRadar_Driver) | ✅ | 串口读取线程 + Map_Circle, 优化后 CPU 12% |
| 雷达数据解析 (LDRadar_Resolver) | ✅ | CRC8 校验, Map_Circle 1080 bin |
| 双雷达融合 (MultiRadar) | ✅ | vstack 点云, mount transform, Y mirror |
| 机身自反射屏蔽 | ✅ | `_body_mask`, ±25cm |
| 障碍物聚类分类 | ❌ | 接口曾定义为 `obstacle_classifier.py`，已删除（零引用死代码）。欧氏聚类→wall/pole/block 保留为开放任务 |
| 世界模型 (local_world_model) | ✅ | 距离过滤 + 机身屏蔽 + 走廊/扇区查询 |
| YOLO 道路感知 (road_perception) | ✅ | ONNX 推理, mask 生成, 中线提取, pixel_error, angle。**NPU 加速待 OS 迁移后验证** |
| **平台** | | |
| 当前 OS | ✅ | Debian 12 Bookworm, aarch64, glibc 2.36 |
| NPU 硬件 | ✅ | `/dev/galcore` 已加载, galcore 6.4.15.6 |
| NPU 用户态库 | ✅ | libGAL, libVSC, libOpenCL_VSI 等 12 个已安装 |
| NPU ONNX Runtime | ❌ | ST 的 `onnxruntime` NPU 变体依赖 OpenSTLinux BSP, 无法在 Debian 12 上运行 |
| **迁移计划** | 🔴 | **切换至 OpenSTLinux v6.2** — 详见 `OS_MIGRATION_PLAN.md` |
| **规划层** | | |
| 反应式避障 (LocalPlanner) | ✅ | 三段式: stop/slow/cruise, 去抖 3 帧 |
| 候选方向搜索 (DirectionPlanner) | ✅ | -90°~+90°, 10°步长, 扇区间隙评估 |
| 目标导航任务 (goal_nav_mission) | ✅ | 转发目标给 DirectionPlanner |
| 道路跟随任务 (road_follow_mission) | ✅ | 状态机: follow/search/timeout |
| **安全层** | | |
| 安全仲裁 (safety_arbiter) | ✅ | 硬故障检查 + 障碍物 stop/slow + 钳位 |
| **公共层** | | |
| 统一速度指令 (autonomy_command) | ✅ | `VelocityCommand`, clamp, as_fc_tuple |
| 传感器上下文 (autonomy_context) | ✅ | `SensorHealth`, `FlightStatus`, `Obstacle`, `AutonomyContext` |
| USB 摄像头探活 | ✅ | 双摄像头 `/dev/video7` (路径) + `/dev/video9` (障碍物), 均 640×480 |
| 硬件适配 (autonomy_hardware) | ✅ | `build_dual_radar`, `connect_fc`, `open_camera` |
| 摄像头色彩诊断工具 | ✅ | `tools/diagnose_camera_color.py`, cam#7 间歇偏蓝/偏青 |
| **入口程序** | | |
| 道路循线入口 (road_follow_main) | ✅ | 双雷达+摄像头+FC 主循环, 20Hz |
| 目标导航入口 (goal_nav_main) | ✅ | 双雷达+FC 主循环, 20Hz |
| 飞控连通性测试 (debug/test_fc_connect) | ✅ | Phase A |
| 飞控指令测试 (debug/test_fc_command) | ✅ | Phase B |
| 飞控实时控制测试 (debug/test_fc_realtime) | ✅ | Phase C |
| 单雷达避障测试 (test_radar_avoidance) | ✅ | Phase D/E, 完整链路 |
| 双雷达避障测试 (test_dual_radar) | ✅ | 30Hz, --loop-hz 可配 |
| 双雷达连通性烟雾测试 (smoke_dual_radar) | ✅ | |
| **性能基线** | | |
| 单雷达: 端到端 <5ms, CPU 12% | ✅ | |
| 双雷达: 30Hz 稳定, CPU 15-25% | ✅ | |

### 待实现 ⚠️

| # | 任务 (对应 Codex 编号) | 优先级 | 预计工作量 |
|---|----------------------|:---:|--------|
| 1 | **雷达真实新鲜度 watchdog** (02) | 🔴 高 | ~30行 |
| | `LD_Radar` 加 `_last_valid_frame_host_time_s` / `is_fresh()` / `get_health_snapshot()` | | |
| | `MultiRadar` 加 `is_fresh()` / `get_health_snapshot()` | | |
| 2 | **SafetyArbiter 集成雷达 fresh 检查** (03) | 🔴 高 | ~10行 |
| | `SafetyArbiter._hard_fault_reason()` 改用 `radar.is_fresh()` 而非仅 `radar.connected` | | |
| 3 | **修复 FC 连接参数** (01) | 🔴 高 | 1行 |
| | `test_dual_radar.py:70` 中 `explicit_port` 改为正确参数名, `goal_nav_main.py` 同样检查 | | |
| 4 | **道路感知完善** (05) | 🟡 中 | ~80行 |
| | 摄像头偏移补偿: `CameraOffsetCompensationConfig` + `apply_camera_offset_compensation()` | | |
| | 岔路分支分类: `RoadBranch.label` (left/straight/right) | | |
| | 分支选择策略: `choose_branch()` 支持 preference (auto/left/right/straight) | | |
| 5 | **新增 RoadFollower 控制器** (05) | 🟡 中 | ~60行 |
| | `FlightController/Solutions/RoadFollower.py` | | |
| | P 控制律: pixel_error*kp_pixel + angle_error*kp_angle → yaw_rate | | |
| 6 | **新增 RelativeGoalNavigator** (06) | 🟡 中 | ~50行 |
| | `FlightController/Solutions/RelativeGoalNavigator.py` | | |
| | 目前根目录 `direction_planner.py` 已覆盖功能, 需评估是否重命名/移动 | | |
| 7 | **任务入口 CLI 对齐** (06) | 🟢 低 | ~30行 |
| | `road_follow_main.py` 增加偏移补偿参数 | | |
| | `goal_nav_main.py` 检查 `explicit_port` 参数名 | | |
| 8 | **新建 tests/ 目录 + 单元测试** (02) | 🟢 低 | ~40行 |
| | `tests/test_radar_freshness.py` — 无硬件单元测试 | | |
| 9 | **道路感知离线测试脚本** (05) | 🟢 低 | ~50行 |
| | CLI 模式: `road_perception.py --image xxx.jpg --model xxx.onnx --debug-out xxx.jpg` | | |
| 10 | **分阶段联调验收** (07) | 🟢 低 | 文档 |
| 11 | **障碍物类型识别** | 🔴 高 | ~100行 |
| | 原 `obstacle_classifier.py` 为空桩已删除，需重新实现: 欧氏聚类→按 size/shape 分类 wall/pole/block，并接入 `local_world_model` | |
| 12 | **操作系统迁移: Debian 12 → OpenSTLinux v6.2** | 🔴 高 | ~4h |
| | 详见 `OS_MIGRATION_PLAN.md` — 烧录镜像 → 重建 Python 环境 → NPU 验证 → 雷达/FC/摄像头全链路验证 | |

---

## 7. 关键风险与未解决问题

| 风险 | 严重度 | 说明 |
|------|:---:|------|
| **ONNX 模型文件缺失** | 🔴 高 | `FlightController/Solutions/model/road_yolo11n_seg.onnx` 是否存在未确认; 不存在则视觉链路完全无法测试 |
| **Debian 12 不支持 NPU** | 🔴 高 | 经过 2026-06-07 全天诊断, ST NPU 软件栈依赖 OpenSTLinux BSP (glibc 2.39, gcnano-driver, libArchModelSw), 无法通过替换 .so 解决; **决策: 切换至 OpenSTLinux v6.2** |
| **无测试图片集** | 🟡 中 | 无法离线验证岔路分类和偏移补偿的正确性 |
| **两套 VelocityCommand 并存** | 🟡 中 | 根目录 `autonomy_command.py` 和 `FlightController/Solutions/LocalPlanner.py` 各有定义; 字段相似但类型不兼容; 长期需统一 |
| **雷达 fresh 用主循环时间替代** | 🟡 中 | 当前 `safety_arbiter.py` 中的 `radar_age_s` 来自 `LocalWorldModel.radar_age_s()` — 那是上次 update 的时间, 不是真实帧时间; 如果串口断流但主循环继续跑, 会误判为新鲜 |
| **eMMC 仅剩 930M** | 🟢 低 | 不能追加大型 APT 包或 pip 包; ONNX 模型和额外 Python 依赖需控制大小 |
| **摄像头色彩偏色** | 🟡 中 | cam#7 间歇偏蓝/偏青，怀疑自动白平衡不稳定; 暂时假定大部分时间正常，后续可用 `cv2.xphoto.createSimpleWB()` 修正 |
| **摄像头偏移补偿符号未验证** | 🟡 中 | 补偿公式中的 `correction_sign` 需实机验证; 写反会导致纠偏方向错误 |
| **RoadFollower yaw 方向未验证** | 🟡 中 | `centerline_angle=90°` 对应正前方的假设需实机验证; 角度符号与 `send_realtime_control_data()` 的 `-yaw` 打包的交互需确认 |
| **ARM 100Hz 内核 sleep 精度** | 🟢 低 | 已知问题, 已用固定频率 `sleep(remaining)` 规避; 但视觉处理加入后 CPU 预算需重新评估 |
| **GIL 尖峰 (362ms)** | 🟢 低 | 偶发, 可能来自 eMMC I/O; 视觉推理期间主线程持 GIL 时间显著增长, 需监控 CRC 错误是否回升 |

---

## 8. 坐标系变换链路备忘

### 上雷达 (正装): 点 → 机体坐标

```
Map_Circle[bin] → 极坐标(angle, distance_mm)
  → get_points_xy_cm():
      x = distance * cos(angle) * 0.1
      y = -distance * sin(angle) * 0.1      ← D500 顺时针, 负号修正
  → get_points_body_cm():
      mirror_y: False (跳过)
      rotation: mount_yaw_deg=0  (恒等)
      translation: + (0, 0)
  → 输出 (x_cm, y_cm) 机体坐标
```

### 下雷达 (倒装): 点 → 机体坐标

```
Map_Circle[bin] → 极坐标(angle, distance_mm)
  → get_points_xy_cm():
      x = distance * cos(angle) * 0.1
      y = -distance * sin(angle) * 0.1
  → get_points_body_cm():
      mirror_y: True  →  y *= -1            ← 倒装导致左右反转
      rotation: mount_yaw_deg=0  (恒等)
      translation: + (0.96, 0.15)
  → 输出 (x_cm, y_cm) 机体坐标
```

### 融合

```python
MultiRadar.get_obstacle_points_body_cm():
  points_upper = radar[0].get_points_body_cm()
  points_lower = radar[1].get_points_body_cm()
  return np.vstack([points_upper, points_lower])
```

---

## 9. 凌霄飞控协议要点 (不可修改)

- **帧格式**: 二进制, 非 MAVLink
- **串口**: 500000 baud
- **心跳**: `AA 22` 帧, 每 250ms
- **模式**: `set_flight_mode(2)` = 定点 (HOLD_POS), 实时控制需定点模式
- **实时控制帧**: `struct.pack("<hhhh", vx, vy, vz, -yaw)` — 注意 `-yaw`
- **状态读取**: `FC_State_Struct` 16 字段 (mode, unlock, bat, pit, rol, alt_add 等)
- **禁止改动**: `Base.py`, `Protocal.py`, `Application.py`, `Serial.py` 中的协议核心

---

## 10. 验收标准

所有测试通过后才允许进入 `--enable-flight` 拆桨测试：

```bash
# 导入检查
python -m py_compile road_perception.py road_follow_main.py goal_nav_main.py

# 飞控三件套
PYTHONPATH=. python debug/test_fc_connect.py
PYTHONPATH=. python debug/test_fc_command.py --target-mode 2
PYTHONPATH=. python debug/test_fc_realtime.py --count 10 --speed 10

# 雷达 ground dry-run
PYTHONPATH=. python -u FlightController/tools/test_radar_avoidance.py --no-fc --dry-run
PYTHONPATH=. python -u FlightController/tools/test_dual_radar.py --no-fc --dry-run

# 任务入口 dry-run (无硬件)
PYTHONPATH=. python road_follow_main.py --no-fc --no-radar --dry-run --loop-hz 2
PYTHONPATH=. python goal_nav_main.py --no-fc --no-radar --dry-run --loop-hz 2

# 任务入口 dry-run (有硬件)
PYTHONPATH=. python road_follow_main.py --no-fc --dry-run --loop-hz 5
PYTHONPATH=. python goal_nav_main.py --no-fc --dry-run --loop-hz 5
```
