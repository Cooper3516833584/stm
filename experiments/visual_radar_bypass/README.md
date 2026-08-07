# 独立真实视觉 + 实体雷达静态障碍绕行实验

生产视觉巡线逻辑保持不变。默认 `static-route` 模式用于绕开指定路线中的单个、静止、
孤立管状障碍物。2026-08-07 根据真实飞行记录修正了主目录 `SafetyArbiter` 的侧向几何和
组合仲裁，但没有降低原有安全阈值。

## 当前稳定版本

- 默认参数档为 `static-route-flight-v1`，状态为 `FROZEN_FLIGHT_VALIDATED`。
- 不传 `--bypass-planner` 时即使用该方案；旧方案仍需显式选择且保持可运行。
- v1 已通过实际飞行验证，绕行过程平滑，无左右摇摆，并能完成侧缘越过确认。
- Git 冻结标签为 `static-route-flight-v1`，对应提交 `4c84333`。
- 冻结的是 v1 默认参数，不是封死规划器；`StaticRouteBypassConfig` 仍可用于后续新增提速
  v2，且不得静默覆盖 v1。

## 默认行为

- 雷达规划范围为机头前方 180°（机体系 `-90°..+90°`）。
- 视觉路径切线继续控制 yaw，使机头跟随局部路径方向。
- 避障完成前屏蔽视觉横向命令；`vy` 只由障碍侧别和实体表面净空决定。
- 避障前向目标为 `14 cm/s × 60% = 8.4 cm/s`，Safety 可以降低或清零。
- 障碍到达 80°～90°侧方并离开前半平面后，使用成功下发的最终
  `vx/vy/yaw_rate` 做保守二维传播；整根管体位于机后 20 cm 后才恢复视觉横移。
- 双雷达完整点云始终进入 Safety，80 cm 前向停车和 45 cm 侧向停车不变。
- 50 cm × 50 cm 机身按半尺寸 25 cm 建模；侧向 Safety 只检查 `|x|≤25 cm` 的实际机体
  扫掠走廊，避免侧后方近中心线杂点误报。

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

真实飞行入口仍保留原有多重确认门。所有板端源码更新必须先提交到 Git，再由开发板
`git fetch`/`git pull --ff-only` 获取；禁止 SCP 或直接编辑板端源码。
