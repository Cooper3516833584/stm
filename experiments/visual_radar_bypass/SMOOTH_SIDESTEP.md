# 锁向单次横移 + 平滑融合测试

该方案是独立的可选规划器，不修改 `radar_bypass.py`，也不改变现有入口的
`legacy` 默认行为。

主要行为：

- 对前方矩形区域内的物理雷达点直接取中位数，不进行网格聚类或候选路径搜索。
- 连续两帧确认后锁定与障碍物相反的横移方向。
- 用 smoothstep 权重在 1 秒内渐入横移指令。
- 障碍点消失后继续保持 2 秒，再用 2.5 秒平滑交还视觉巡线。
- 融合结束前障碍再次出现时沿用原方向，不重新选择左右侧。
- 规划器连续横移超过 9 秒时停车，避免超出约 ±90 cm 的实验活动范围。

无飞控实物传感器测试：

```bash
PYTHONPATH=. python3 -u -m experiments.visual_radar_bypass.main \
  --bypass-planner smooth-sidestep \
  --no-record --duration-s 60
```

真实飞行测试仍需原有的显式确认参数；未指定 `--bypass-planner` 时继续使用
原有规划器。
