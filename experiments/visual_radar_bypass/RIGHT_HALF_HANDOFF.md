# 右半平面雷达后切换纯视觉

该模式必须显式增加：

```bash
--right-half-radar-then-visual
```

启用后，雷达阶段只处理机体坐标系 `y <= 0` 的点，即以前向为 0°、顺时针
0° 到 180° 的右半平面。原 `legacy` 避障和三段式 `forward_recovery`
保持不变。

一次 `forward_recovery` 正常结束后开始计时。如果规划器连续处于 `normal`
满 5 秒且没有重新进入雷达避障，本次会话将停止双雷达线程、清空雷达场，
并永久切换为轨迹视觉控制。重新进入左右避障会取消本次计时；仅处于普通
`normal`、但此前没有完成 `forward_recovery` 时不会关闭雷达。为避免两帧
激活门槛的边界竞争，只要出现第一帧待确认侵入点，也会立即取消计时。

真实飞行参数示例：

```bash
PYTHONPATH=. /usr/local/UFC_venv/bin/python3 -u \
  -m experiments.visual_radar_bypass.main \
  --bypass-planner legacy \
  --bypass-forward-transition-s 2.0 \
  --right-half-radar-then-visual \
  --enable-flight \
  --auto-takeoff \
  --confirm-visual-radar-flight-test \
  --takeoff-height-cm 100 \
  --duration-s 60
```

不输入 `--right-half-radar-then-visual` 时，双侧雷达范围和持续雷达安全处理
均保持原行为。
