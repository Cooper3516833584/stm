# 雷达圆拟合绕管实验

该模式由独立开关启用，不改变 `legacy`、`smooth-sidestep` 或右半平面实验：

```bash
--circular-tube-bypass
```

每帧只进行一次侵入区域向量筛选和一次代数最小二乘圆拟合，直接从管状物
雷达弧点得到圆心与半径。绕行轨迹半径为：

```text
雷达拟合管半径 + 安全外拓半径
```

默认安全外拓半径为 75 cm。拟合点不足、秩不足、半径超限或 RMS 残差超过
4 cm 时，使用 15 cm 备用管半径，因此备用绕行半径为 90 cm。参数可通过
`--tube-radius-cm` 和 `--tube-safety-radius-cm` 调整。

控制器只计算圆周切向速度与径向误差修正。轨迹达到默认 90° 圆弧、障碍物
离开观测区或最长 12 秒时进入视觉回归。视觉误差必须先离开 ±50 px，随后
重新回到 `abs(err) < 50`，才可提前结束圆弧，避免道路原本居中时立即退出
避障。视觉回归使用 1.5 秒平滑融合，偏航角速度限制为 ±7°/s，整数化后仍
严格小于 8°/s。

真实飞行参数示例：

```bash
PYTHONPATH=. /usr/local/UFC_venv/bin/python3 -u \
  -m experiments.visual_radar_bypass.main \
  --bypass-planner legacy \
  --circular-tube-bypass \
  --enable-flight \
  --auto-takeoff \
  --confirm-visual-radar-flight-test \
  --takeoff-height-cm 100 \
  --duration-s 60
```

该开关不能和 `--right-half-radar-then-visual` 或 `smooth-sidestep` 同时使用。
