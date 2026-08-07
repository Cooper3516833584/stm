# 锁向单次横移 + 平滑融合测试

该方案是当前独立实验的默认规划器，不修改 `radar_bypass.py`；原 `legacy`
规划器仍可通过 `--bypass-planner legacy` 显式运行。

主要行为：

- 对前方矩形区域内的物理雷达点直接取中位数，不进行网格聚类或候选路径搜索。
- 连续两帧确认后锁定与障碍物相反的横移方向。
- 用 smoothstep 权重在 1 秒内渐入横移指令。
- 活动目标为 `vx=0, |vy|=10cm/s`，不额外生成 yaw；视觉 yaw rate 保持连续。
- 障碍点消失后继续保持 2 秒，再用 2.5 秒平滑交还视觉巡线。
- 融合结束前障碍再次出现时沿用原方向，不重新选择左右侧。
- 规划器连续横移超过 9 秒时停车，避免超出约 ±90 cm 的实验活动范围。

无飞控实物传感器测试：

```bash
PYTHONPATH=. python3 -u -m experiments.visual_radar_bypass.main \
  --bypass-planner smooth-sidestep \
  --no-record --duration-s 60
```

真实飞行测试仍需原有的显式确认参数。上述速度和时间参数均为尚待实飞确认的
`UNVERIFIED_TUNING`；生产 SafetyArbiter 的 80 cm 前向停止阈值没有修改。
