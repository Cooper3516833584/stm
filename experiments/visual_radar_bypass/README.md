# 独立视觉巡线 + 实体雷达避障测试

该目录不修改现有 `road_follow_main.py`、`road_trajectory_main.py`、视觉后处理或生产
绕障规划器。视觉层冻结当前 NPU `fast-main` + `TrajectoryPointFollower` 的配置；后续
雷达绕障调试只修改 `radar_bypass.py`。

测试假设：

- 障碍物是真实树木，不生成、不注入虚拟点云；
- 树木只在机体左侧，约位于道路中心 `Y=+40cm`；
- 右侧已人工确认无其他障碍；
- 双雷达完整点云仍进入全局安全仲裁器，规划器只忽略已确认空侧的杂散回波。

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
