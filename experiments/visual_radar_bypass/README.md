# 独立真实视觉 + 实体雷达静态障碍绕行实验

本目录不会修改生产视觉巡线或主目录 `SafetyArbiter`。默认
`static-route` 模式用于绕开指定路线中的单个、静止、孤立管状障碍物。

## 默认行为

- 雷达规划范围为机头前方 180°（机体系 `-90°..+90°`）。
- 视觉路径切线继续控制 yaw，使机头跟随局部路径方向。
- 避障完成前屏蔽视觉横向命令；`vy` 只由障碍侧别和实体表面净空决定。
- 避障前向目标为 `14 cm/s × 60% = 8.4 cm/s`，Safety 可以降低或清零。
- 障碍到达 80°～90°侧方并离开前半平面后，使用成功下发的最终
  `vx/vy/yaw_rate` 做保守二维传播；整根管体位于机后 20 cm 后才恢复视觉横移。
- 双雷达完整点云始终进入 Safety，80 cm 前向停车和 45 cm 侧向停车不变。

详细状态与参数见 [STATIC_ROUTE_BYPASS.md](STATIC_ROUTE_BYPASS.md)，验证结果见
[OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md)。

## 无飞控实体传感器验证

```bash
PYTHONPATH=. /usr/local/UFC_venv/bin/python3 -u \
  -m experiments.visual_radar_bypass.main \
  --duration-s 60
```

不要传入 `--enable-flight`、`--auto-takeoff` 或飞行确认参数。dry-run 会启动真实
摄像头和双雷达，但保持 `fc=None`，也不会把规划命令计入已执行里程。

## 离线闭环断言与基准

```bash
PYTHONPATH=. /usr/local/UFC_venv/bin/python3 -u \
  -m experiments.visual_radar_bypass.benchmark_bypass --assert-only

PYTHONPATH=. /usr/local/UFC_venv/bin/python3 -u \
  -m experiments.visual_radar_bypass.benchmark_bypass --iterations 2000

PYTHONPATH=. /usr/local/UFC_venv/bin/python3 -u \
  -m experiments.visual_radar_bypass.replay_radar_session \
  /data/stm_records/<session>
```

## 旧模式兼容入口

```bash
# Legacy，0 仍表示禁用原 forward recovery
python3 -m experiments.visual_radar_bypass.main \
  --bypass-planner legacy --bypass-forward-transition-s 0

# 原 smooth sidestep
python3 -m experiments.visual_radar_bypass.main \
  --bypass-planner smooth-sidestep

# 原 circular 和 right-half 专用标志仍自动选择 legacy
python3 -m experiments.visual_radar_bypass.main --circular-tube-bypass
python3 -m experiments.visual_radar_bypass.main --right-half-radar-then-visual
```

真实飞行入口仍保留原有多重确认门，但本次开发板验证严格禁止连接飞控或实飞。
