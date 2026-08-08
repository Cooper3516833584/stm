# 真实视觉 + 双雷达融合 static-route

该目录是当前视觉巡线与双雷达融合入口。默认使用 `ProcessRuntime` 的 spawn 多进程架构：视觉、
双雷达、记录和控制相互隔离，控制进程只读取最新快照。详细实现、验证结果和已知限制见
[OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md)。

根目录 `road_trajectory_main.py` 仍是复用 `road_follow_main` 的单视觉入口，当前没有接入这里的
双雷达进程；融合验收以本目录的 static-route 入口为准。

## 当前架构摘要

- 控制主进程与雷达进程运行在 CPU0，视觉/NPU 进程运行在 CPU1；
- 视觉图像使用 8 槽共享内存，合并雷达点云使用 2 槽共享内存；
- 元数据通道有界并采用 latest-only，不补算历史传感器帧；
- 双雷达统一批量读取、CRC 校验、时间戳回绕处理和 NumPy 地图更新；
- SessionRecorder 使用低优先级独立进程处理 JSONL、日志、视频、JPEG 和 NPZ；
- 雷达 CRC 新增立即触发不健康，连续 5 个新鲜快照无新增 CRC 后允许恢复；
- 默认 `--runtime-mode process`；`threaded` 仅作为旧路径回退。

static-route 的冻结 v1 参数和状态机行为继续保留为回归基线，详见
[STATIC_ROUTE_BYPASS.md](STATIC_ROUTE_BYPASS.md)。

## 22 cm/s 巡线转弯折中配置

冻结 `FrozenVisualConfig()` 的 v1 行为不变；仅 `static-route-22cm-experiment` 启用根目录
45 cm/s 巡线入口中已经单元验证的转弯机制。数值按当前 22 cm/s 前速、18°/s yaw 和 12 cm/s
横移上限折中缩放，没有照搬生产入口的激进幅值：

| 机制 | 22 cm/s 生效值 | 参数依据 |
|---|---:|---|
| 切线窗口 | 3 点 | 22 cm/s 实际前视约 60–82 px；v1/根目录的 5 点窗口保持不变 |
| 独立平面减速度 | 60 cm/s² | `120 × 22 / 45 ≈ 58.7`，取 60；加速度仍为 36 |
| 有符号转弯 yaw 前馈 | Kp 0.10，上限 6°/s | 根目录 0.30/18°/s 按 18/55 yaw 权限缩放 |
| 急弯前视下限 | 75 px | 同相机/路径采样几何；为3点切线窗口保留足够的前馈观测跨度 |
| 边缘恢复门 | 0.55 开始、0.90 全量 | `path_width_px` 固定，沿用无量纲门限 |
| 边缘恢复 `vy` | Kp 0.16，上限 12 cm/s | 根目录 Kp 0.22 按 12/16 横移权限缩放 |
| 边缘向内 yaw | 0.75 开始、0.95 全量、上限 3°/s | 门限沿用；8°/s 按 18/55 缩放 |
| 边缘前速限制 | 0.90 开始、0.95 全量、18.5 cm/s | `38 × 22 / 45 ≈ 18.6` |

曲率降速仍使用已经选定的 `18° → 52°`、最低 15 cm/s；总速度上限仍为
`vx=22 cm/s`、`vy=12 cm/s`、yaw=18°/s。短时丢线宽限保持关闭，丢线仍立即悬停，避免与
static-route 的 `PATH_LOST_HOLD` 和障碍 Safety 产生新的耦合。上述新增值仍属于未飞行验证参数，
必须先完成当前 22 cm/s 实验，不能据此提高巡线速度。

## 紫色目标与投放任务

默认 static-route 入口同时启动 10 Hz 紫色连通区域检测。目标连续确认后，目标状态机暂时替代巡线
期望命令，但仍经过同一个绕障规划器、Safety 和命令发送门；道路 NPU 全程保持运行。完成投放或
放弃后只停止目标检测，并立即恢复巡线。完整状态、投放安全门和参数依据见
[PURPLE_TARGET_MISSION.md](PURPLE_TARGET_MISSION.md)。

诊断时可使用 `--disable-target-mission` 关闭该任务。legacy、smooth-sidestep 和 circular 旧模式
不会启动目标任务。

## 指示灯

| 飞行状态 | 指示灯 |
|---|---|
| 初始化完成 | 绿灯；`set_digital_output(0, True)` 后等待 15 秒 |
| 即将起飞 | 红灯，等待 5 秒 |
| 正常巡线 | 绿灯 |
| 正常避障 | 红灯 |
| 紫色目标确认后至回归巡线 | 黄灯；目标任务中的正常避障暂时显示红灯，净空后恢复黄灯 |
| 雷达、视觉、道路、Safety 或规划状态异常 | 红灯亮 0.2 秒、灭 0.2 秒循环 |

指示灯只在真实飞行模式连接飞控后启用。dry-run 不连接飞控，不会改变实体灯或数字输出。

## 无飞控验证

```bash
cd /usr/local/ObstacleAvoidanceDrone
PYTHONPATH=. /usr/local/UFC_venv/bin/python3 -u \
  -m experiments.visual_radar_bypass.main \
  --runtime-mode process \
  --bypass-planner static-route \
  --loop-hz 10 \
  --duration-s 60 \
  --no-record
```

不要加入 `--enable-flight`、`--auto-takeoff` 或飞行确认参数。该命令会打开真实摄像头和双雷达，
但保持 `fc=None`，不会解锁、起飞或发送飞控命令。

## 记录输出

默认会话根目录为 `/data/stm_records`：

```text
session.json
runtime.log
commands.jsonl
frames.jsonl
radar.jsonl
camera.avi
frames/*.jpg
radar_points/*.npz
```

关键元数据与媒体任务使用不同队列；记录进程故障不会阻塞控制循环。真实飞行启动前记录器不可用
时，程序仍拒绝飞行。

## 旧模式

```bash
python3 -m experiments.visual_radar_bypass.main --runtime-mode threaded
python3 -m experiments.visual_radar_bypass.main --bypass-planner legacy
python3 -m experiments.visual_radar_bypass.main --bypass-planner smooth-sidestep
python3 -m experiments.visual_radar_bypass.main --circular-tube-bypass
```

这些入口仅用于兼容和诊断，不代表当前默认架构。

## 部署

板端源码只能通过 Git 获取已提交的版本：本地提交并推送后，板端执行 `git pull --ff-only`。
禁止通过 SCP、编辑器或 SSH 命令直接修改板端项目源码，板端也不得向远端推送。
