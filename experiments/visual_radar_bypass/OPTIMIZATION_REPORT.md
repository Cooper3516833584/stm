# 融合控制链路算力优化与运行架构报告

最后更新：2026-08-08。

本文是视觉巡线与双雷达融合入口的当前架构记录。旧的“视觉、雷达、控制和记录共享一个
Python 解释器”的描述已失效；当前默认运行方式为 `process`，控制主进程只消费最新快照。
尚未完成实飞验收的调速参数不在本文中作为结论记录。

## 1. 适用入口

- 融合避障入口：`python3 -m experiments.visual_radar_bypass.main`。
- 根目录单巡线入口：`python3 road_trajectory_main.py`。
- 本文记录的 `ProcessRuntime` 多进程架构已经接入融合避障入口；`--runtime-mode process` 是该
  入口的默认值，`--runtime-mode threaded` 仅保留为旧实现回退入口。
- `road_trajectory_main.py` 目前仍复用 `road_follow_main` 的单视觉管线，没有接入本文的双雷达
  进程；保留列出该入口是为了明确代码边界，不能据此声称两个入口已经共享运行底座。

融合实飞验收以 static-route 融合入口为准，不能用只包含视觉链路的单巡线入口代替。

## 2. 当前进程拓扑

```text
                         ┌──────────────────────────────┐
                         │ 控制主进程 / CPU0            │
                         │                              │
视觉快照 ───────────────→│ TrajectoryPointFollower      │
雷达快照/点云 ──────────→│ StaticRouteBypassPlanner     │
                         │ SafetyArbiter                │
                         │ 飞控命令 + 指示灯状态         │
                         └──────────────┬───────────────┘
                                        │ 小记录任务
                 ┌──────────────────────┴──────────────────────┐
                 │                                             │
┌────────────────▼──────────────┐             ┌────────────────▼─────────────┐
│ 视觉进程 / CPU1               │             │ 记录进程 / nice +10          │
│ 摄像头采集线程                 │             │ JSONL、runtime.log           │
│ STAI NPU 推理                  │             │ JPEG/MJPG、诊断叠图          │
│ 道路后处理与目标提取           │             │ 点云 NPZ 压缩                │
└───────────────────────────────┘             └──────────────────────────────┘

┌───────────────────────────────┐
│ 双雷达进程 / CPU0             │
│ ttySTM4 + ttySTM9 统一轮询     │
│ 批量解析 + NumPy 地图内核      │
└───────────────────────────────┘
```

所有子进程使用 Linux `spawn` 创建，不继承已经打开的 NPU、OpenCV、串口和飞控状态。

## 3. 控制主进程

控制主进程负责：

1. 非阻塞读取最新视觉快照和雷达快照；
2. 运行 `TrajectoryPointFollower`；
3. 运行 static-route 或显式选择的其他绕障规划器；
4. 运行 `SafetyArbiter`；
5. 在真实飞行模式下发送最终命令；
6. 更新状态指示灯；
7. 将小型元数据任务送入记录进程。

主循环不等待摄像头采集、NPU 推理、雷达串口读取、视频编码或点云压缩。传感器快照丢失或
过期时，控制进程继续使用既有 Safety 硬停车规则，不会等待传感器恢复后再计算本帧命令。

`LoopRateMonitor` 记录目标/实际频率、工作耗时 p50/p95/p99、最大耗时、抖动 p99 和 deadline
miss。融合入口当前控制频率保持 10 Hz，不继续以提高频率作为本阶段目标。

## 4. 视觉进程

视觉进程由 `FlightController/Runtime/VisionProcess.py` 实现：

- 在导入 NPU 和视觉管线前绑定 CPU1；
- 独占道路摄像头、采集线程、STAI session、预处理、NPU 推理和后处理；
- 使用 V5 道路语义分割 `.nb`，运行后端为 `NPU_NBGraph`；
- 摄像头采集线程持续覆盖旧帧，推理始终优先处理最新画面；
- 通过单槽队列发布紧凑 `VisionSnapshot`，控制端不会排队处理历史视觉结果；
- 快照包含序号、采集/完成时间、相机健康、道路感知、目标结果、推理分段耗时、错误数和
  发布丢弃计数；
- 原始 `640×480×3 uint8` 图像放入 8 槽共享内存环形缓冲，队列只传 `FrameRef`；
- 每个槽位带 generation，读前和读后校验 generation，若生产者已覆盖该槽则拒绝返回旧图。

视觉仍保留 NPU 计算，没有切换到 CPU 模型，也没有定期重启视觉进程。

## 5. 双雷达进程

双雷达进程由 `FlightController/Components/RadarProcess.py` 实现：

- 绑定 CPU0；
- 一个进程独占 `/dev/ttySTM4`（上雷达）和 `/dev/ttySTM9`（下雷达）；
- 按 20 ms 批处理周期统一轮询两个串口；
- 每次读取全部 `in_waiting` 数据，支持分包、粘包、错位头和错误帧后的重新同步；
- D500 帧长度固定为 47 byte，保留 CRC8 校验；
- 设备 30 秒时间戳使用 `RadarTimestampTracker` 扩展并记录回绕次数；
- 热路径直接生成结构化 NumPy 批次，不再为每个点创建 `Point_2D`、小列表和字典。

### 5.1 NumPy 原生地图内核

每个雷达维护 1080 个角度 bin（360° × 3）：

- 批量计算角度、有效点和 REMAP 索引；
- 同一帧写入同一 bin 时使用 NumPy 归并求最小距离；
- 同一批中后到帧覆盖先到帧，保持旧逐帧算法的顺序语义；
- 地图过期清理最多每 40 ms 执行一次；
- 单个 bin 的默认数据超时为 150 ms；
- 下雷达在机体坐标变换中执行 Y 镜像并应用安装偏移；
- 两个雷达的机体坐标点云合并后发布。

点云写入 2 槽共享内存环形缓冲，每槽最多 2160 个 `float32 [x,y]` 点。控制进程只复制当前
有效点，generation 不匹配时拒绝使用被覆盖的数据。

### 5.2 雷达快照和 freshness

雷达进程默认以 25 Hz 发布 `RadarSnapshot`。快照包含：

- 合并点云引用和有效点数；
- 两个雷达各自的最后帧年龄、有效帧总数、CRC 总数、解析缓冲长度和串口读取统计；
- 合并序号、发布时间、最老雷达帧时间和发布丢弃计数。

融合 Safety 使用双雷达中较老的一路作为聚合年龄。任一路超过 0.5 秒、工作进程死亡、点云
共享内存读取失败或 CRC 健康门未恢复，均会触发 `radar_not_fresh`。

CRC 判定使用“增量错误 + 自动恢复”，不再要求整个进程生命周期内累计 CRC 永远为零：

1. CRC 总数新增时立即锁定为不健康；
2. 雷达停帧或快照不新鲜时，恢复计数清零；
3. 连续 5 个新鲜新快照没有新增 CRC 后解除 CRC 锁定；
4. 累计 CRC 总数仍写入健康日志，便于区分瞬态恢复和持续串口故障。

这只修复“瞬时 CRC 后永久停车”的软件锁死。真实串口再次停帧或新增 CRC 时，Safety 仍会
重新停车。

## 6. 共享内存与 latest-only 通道

`ProcessRuntime` 使用：

| 数据 | 通道 | 容量 | 行为 |
|---|---|---:|---|
| 视觉元数据 | multiprocessing Queue | 1 | 永远只保留最新快照 |
| 雷达元数据 | multiprocessing Queue | 8 | 控制端一次排空到最新快照 |
| BGR 图像 | SharedMemory ring | 8 槽 | generation 双重校验 |
| 合并点云 | SharedMemory ring | 2 槽 | 只复制有效前缀 |

传感器生产者不会因控制进程暂时繁忙而无限堆积数据。丢弃旧快照是设计行为，并通过
`publish_drops` 统计；控制链路不补算旧帧。

## 7. 独立记录进程

`SessionRecorder` 的公开调用方式和会话目录兼容旧实现，但内部已经改为 `spawn` 子进程：

- 记录进程允许使用两个核心，并在 Linux 上执行 `nice +10`；
- 关键队列默认 512 项，保存命令、安全裁决、运行日志和 JSONL；
- 媒体队列默认 8 项，保存帧、视频和点云压缩任务；
- JSON flush、诊断叠图、JPEG、MJPG 和 `np.savez_compressed` 均不在控制主进程执行；
- 图像任务可直接引用视觉共享内存中的 `FrameRef`；
- 媒体拥塞先丢弃媒体任务，关键元数据不与媒体共用队列；
- 记录采样支持固定时间频率，不会因控制循环频率提高而成倍增加负载。

目录和回放合同保持为：

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

真实飞行开始前记录器不可用时仍拒绝飞行；运行中记录进程故障只记录告警，不阻塞控制线程。

## 8. CPU 调度与生命周期

- 视觉进程：CPU1；
- 控制主进程：CPU0；
- 双雷达进程：CPU0；
- 记录进程：不固定核心，`nice +10`；
- 非 Linux 或单核环境：亲和性设置自动跳过。

主进程只在两个 worker 创建后绑定 CPU0，防止视觉子进程继承错误的单核亲和性。关闭顺序为：

1. 停止视觉和雷达生产者；
2. 让记录进程排空已引用的共享内存任务；
3. 关闭队列；
4. 关闭并 unlink 共享内存。

`ProcessRuntime.stop()` 和 `SessionRecorder.close()` 均按幂等方式设计，避免重复清理造成异常。

## 9. 安全链路

控制顺序保持为：

```text
视觉期望命令 → 绕障规划命令 → SafetyArbiter → 飞控最终命令
```

优化没有绕过 Safety：

- 雷达 freshness 门限保持 0.5 秒；
- 前向硬停车保持 80 cm；
- 前向慢速区域保持 150 cm；
- 侧向停车保持 45 cm；
- 机体按 50 cm × 50 cm 矩形近似；
- 侧向扫掠只检查 `|x|≤25 cm`；
- 飞控连接、模式、解锁、姿态、非法数值和可选电池门均继续参与硬停车。

控制日志同时保存 `desired`、`planned`、`safe`、`final` 和 Safety 原因，便于区分视觉/规划器
仍有输出但被安全层清零的情况。

## 10. 指示灯与数字输出

指示灯只在真实融合飞行路径连接飞控后工作；dry-run 不连接飞控，也不会操作灯或数字输出。

| 状态 | 指示 |
|---|---|
| 初始化完成、起飞准备 | 绿灯；执行 `set_digital_output(0, True)` 后等待 15 秒 |
| 即将起飞 | 红灯常亮，等待 5 秒 |
| 起飞完成、正常巡线 | 绿灯常亮 |
| 正常巡线仅受 Safety 限速（`LIMITED`） | 绿灯常亮 |
| 紫色目标连续确认后至任务完成/放弃恢复完成 | 黄灯常亮 |
| 正常避障或 Safety 障碍停车（`OBSTACLE_STOP`） | 红灯常亮 |
| 雷达/视觉失效、道路丢失、Safety 硬停车、路径/跟踪丢失、failsafe、timeout | 红灯亮 0.2 秒、灭 0.2 秒循环 |

LED 只在颜色或闪烁相位变化时发送命令，且飞行控制命令先于本帧 LED 更新发送。数字输出 0
置为 True 后不会由该指示模块自动复位。灯光优先级为异常闪红、正常避障/障碍停车红、目标任务
黄、巡线绿；`LIMITED` 本身不改变灯色。因此目标阶段发生绕障时暂时显示红灯，净空后恢复黄灯，
交还巡线后才恢复绿色。

## 11. NPU 内存现状

板端定位确认持续内存增长来自 STAI MPU 6.0.1 的 NBG `set_input()` 路径，而不是控制、雷达、
记录队列或共享内存泄漏。重复 `set_input()` 即使复用同一 ndarray，RSS 仍约按每次 384 KiB
增长；`run()` 和 `get_output()` 本身没有同量级持续增长。

已经验证但未采用的方案：

- 定期重启视觉 worker 可以回收部分内存，但会造成约 6.7 秒视觉中断；
- CPU INT8 模型内存稳定且较快，但与当前 NPU V5 输出差异过大；
- VSINPU 直接加载替代模型会发生段错误。

当前决定是保留 NPU 推理，不做周期性 worker 重启。因此 STAI 内存增长仍是已知限制，长时间
连续运行需要监控 RSS；该问题不能被描述为已经解决。

## 12. 已完成验证

- 雷达分包、粘包、错位头、CRC、时间戳回绕和 NumPy 地图语义均有专项测试；
- `ProcessRadarClient` 聚合年龄已通过 Safety 离线回放，不再因接口缺失产生假
  `radar_not_fresh`；
- 最近实飞 CRC/年龄序列重放证明：CRC 新增立即停车，连续健康快照后可恢复，后续真实故障
  会再次停车；
- 板端 CRC 恢复断言通过；
- 2026-08-08 板端真实相机、NPU、双雷达 dry-run 达到约 9.97–9.98 Hz；
- dry-run 明确没有飞控连接、没有解锁、没有起飞；
- 状态灯调用顺序和颜色状态在本地及板端假飞控对象上通过；
- dry-run 结束后没有残留视觉或雷达进程。

板端 `UFC_venv` 当前没有安装 `pytest`，因此板端使用直接断言和 `compileall` 验证；完整 pytest
仍应在具备测试依赖的开发环境运行。

## 13. 当前事实对旧文档的替换

以下旧描述不再代表融合程序当前默认架构：

- “视觉、双雷达和控制共享一个 Python 解释器”已由三个进程替换；
- “融合入口直接管理视觉/雷达线程”已由 `ProcessRuntime` 快照接口替换；
- “雷达使用 `MultiRadar/LDRadar_Driver` 作为默认热路径”仅适用于 `threaded` 旧路径；
- “逐点 Python 地图更新”已由批量 NumPy 内核替换；
- “压缩、绘图和 JSON flush 在控制线程执行”已由独立记录进程替换；
- “任意历史 CRC 导致整次运行永久不健康”已由 CRC 增量门和连续健康快照恢复替换；
- “按每 N 个控制循环增加全部记录负载”已由固定时间频率调度替换。

## 14. 关键实现文件

| 功能 | 文件 |
|---|---|
| 进程运行底座、共享内存和快照客户端 | `FlightController/Runtime/ProcessRuntime.py` |
| 视觉进程 | `FlightController/Runtime/VisionProcess.py` |
| 双雷达批量解析和 NumPy 地图 | `FlightController/Components/RadarProcess.py` |
| 独立记录进程 | `FlightController/Solutions/SessionRecorder.py` |
| Safety | `FlightController/Solutions/Safety.py` |
| 融合入口 | `experiments/visual_radar_bypass/main.py` |
| static-route 状态机 | `experiments/visual_radar_bypass/static_route_bypass.py` |
| 指示灯状态机 | `experiments/visual_radar_bypass/flight_indicator.py` |

## 15. 部署约束

- 所有源码修改先在本地 Git 提交并推送；
- 开发板只能执行 `git pull --ff-only` 获取提交，不允许通过 SSH 直接编辑或复制项目源码；
- 板端不得向远端推送；
- benchmark/dry-run 命令不得包含 `--enable-flight`、`--auto-takeoff` 或飞行确认参数；
- 未经明确授权不得连接飞控、解锁或起飞。

## 16. 紫色目标任务接入

默认 static-route 入口现已将低成本紫色连通区域检测放在既有视觉子进程中，并限制为 10 Hz。
检测与道路 NPU 共用相机帧；道路、雷达和控制进程拓扑不变。控制主进程只在连续 3 个目标结果
确认后运行目标方位和高度状态机，目标期望仍依次经过 static-route、Safety 和发送门。

投放完成或任务放弃后，process 模式通过独立事件只停止目标线程，threaded 模式调用相同的目标
生命周期接口；相机采集和道路 NPU 不停止。任务状态、原始/滤波偏移、方位、ALT_ADD、道路与目标
期望、规划/Safety/最终命令及投放事件均进入现有独立记录进程。具体状态和参数见
[PURPLE_TARGET_MISSION.md](PURPLE_TARGET_MISSION.md)。

## 17. 22 cm/s 巡线转弯逻辑的选择性融合

根目录 `road_trajectory_main.py` 与融合入口共用 `TrajectoryPointFollower`，但前者的目标速度为
45 cm/s，不能直接复制参数。冻结 v1 配置继续作为已验证基线；22 cm/s实验只打开以下已有控制
分支：有符号转弯 yaw 前馈、曲率触发的前视缩短、基于固定 `path_width_px` 的边缘恢复、边缘
降速以及独立减速度。

缩放结果为：3点切线窗口，减速度60 cm/s²，yaw前馈Kp 0.10/上限6°/s，急弯前视下限75 px，边缘恢复
Kp 0.16/最大12 cm/s，边缘yaw最大3°/s，边缘前速下限18.5 cm/s。无量纲道路边缘门限沿用
0.55/0.90和0.75/0.95；曲率降速仍为18°到52°、最低15 cm/s。根目录的0.18秒丢线继续飞行
未接入，避免影响static-route路径丢失停车和Safety语义。

所有值均进入会话参数注册表，来源标记为 `UNVERIFIED_TUNING`；在22 cm/s飞行实验通过前，
不得提升速度或把这些数值标记为飞行验证。

## 18. 22 cm/s 绕障退出简化

冻结 `static-route-flight-v1` 仍保留预测圆心、管半径、机后余量和全前向走廊净空的完成条件。
当前 `static-route-22cm-experiment` 改为 `clearance_run_s=1.5`：进入 `CLEARANCE_RUN` 后清除旧管体
预测，不再对其做50 cm关联，也不再以 `front_corridor_clear` 作为门槛。规划器改用 NORMAL 的
10～180 cm、横向±75 cm和连续2帧规则发现新路径障碍；确认后开启新的 encounter。没有新障碍时，
只有Safety之后实际执行的正向帧才累计时间，停车或拒绝执行会把连续时间清零。累计1.5秒后进入
`WAIT_VISUAL`，再按原有2秒 `BLEND_BACK` 回到巡线。该参数属于 `UNVERIFIED_TUNING`。
