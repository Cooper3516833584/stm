# 独立视觉巡线 + 实体雷达避障测试

该目录不修改现有 `road_follow_main.py`、`road_trajectory_main.py`、视觉后处理或生产
绕障规划器。视觉层冻结当前 NPU `fast-main` + `TrajectoryPointFollower` 的配置；后续
雷达绕障调试只修改 `radar_bypass.py`。

测试条件：

- 障碍物是可移动的真实管状体，不生成、不注入虚拟点云；
- 不预设障碍物在道路左侧还是右侧，也不配置固定横向位置；
- 障碍物周围已人工确认没有其他真实障碍；
- 规划器在 `x=40..180cm、|y|<=75cm` 内提取最密集的实体雷达点簇；
- 障碍簇在左侧时向右绕，在右侧时向左绕，接近中心时默认向右绕；
- 一次绕行过程中锁定方向，避免雷达噪声导致左右反复切换；
- 双雷达完整点云始终进入全局安全仲裁器。

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
