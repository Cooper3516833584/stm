# 独立视觉巡线 + 实体雷达避障测试

该目录不修改现有 `road_follow_main.py`、`road_trajectory_main.py`、视觉后处理或生产
绕障规划器。视觉层冻结当前 NPU `fast-main` + `TrajectoryPointFollower` 的配置；后续
雷达绕障调试只修改本实验目录。生产视觉巡线、生产绕障规划器和
`SafetyArbiter` 均保持不变。

测试条件：

- 障碍物是可移动的真实管状体，不生成、不注入虚拟点云；
- 不预设障碍物在道路左侧还是右侧，也不配置固定横向位置；
- 障碍物周围已人工确认没有其他真实障碍；
- 规划器在 `x=40..180cm、|y|<=75cm` 内提取最密集的实体雷达点簇；
- 障碍簇在左侧时向右绕，在右侧时向左绕，接近中心时默认向右绕；
- 一次绕行过程中锁定方向，避免雷达噪声导致左右反复切换；
- 双雷达完整点云始终进入全局安全仲裁器。

默认规划器现为 `smooth-sidestep`：连续两帧确认后，每个 encounter 只选择一次
绕行侧；短时雷达丢点和融合期重现不会重新选边。活动绕障目标为纯横移
`vx=0, |vy|=10cm/s`，用 smoothstep 渐入/渐出，从而避免前向障碍距离在 80 cm
附近抖动时由安全仲裁器反复开关 `vx`。原有方案仍可显式运行：

```bash
# 原基础方案及 forward recovery
PYTHONPATH=. python3 -u -m experiments.visual_radar_bypass.main \
  --bypass-planner legacy --bypass-forward-transition-s 2.0 \
  --no-record --duration-s 60

# 原圆拟合方案
PYTHONPATH=. python3 -u -m experiments.visual_radar_bypass.main \
  --circular-tube-bypass --no-record --duration-s 60
```

因此，管状体只在部分道路右侧不会影响其他道路：没有检测到有效障碍簇时，规划器
保持 `normal`，视觉巡线命令不被修改。

无飞控实物传感器测试：

```bash
PYTHONPATH=. python3 -u -m experiments.visual_radar_bypass.main \
  --no-record --duration-s 60
```

真实飞行测试（会实际解锁和起飞）：

```bash
PYTHONPATH=. python3 -u -m experiments.visual_radar_bypass.main \
  --enable-flight \
  --auto-takeoff \
  --confirm-visual-radar-flight-test \
  --takeoff-height-cm 100 \
  --duration-s 60
```

真实飞行前会依次验证模型文件、记录目录、两只实体雷达的新鲜数据、连续三帧真实
道路识别以及飞控电池/锁定状态。测试到时或按 `Ctrl+C` 后调用飞控原生降落流程。

无摄像头、无实体雷达、无飞控的离线稳定性与性能验证：

```bash
PYTHONPATH=. python3 -u -m experiments.visual_radar_bypass.benchmark_bypass \
  --iterations 3000 --warmup 200
```

输出包括各方案 planner update 的 mean/p50/p95，以及 75/80 cm 阈值噪声下的
state switch、side switch、SafetyArbiter override 和最大命令增量。运行时结构化
命令日志默认每 2 帧采样，雷达点快照默认每 5 帧采样，可分别用
`--tuning-log-every-n` 和 `--radar-snapshot-every-n` 调整；这两个参数尚未经过实飞
调优。
