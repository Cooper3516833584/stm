# 雷达避障到视觉巡线的前向过渡态

原有 `legacy` 雷达避障规划器默认在障碍消失后进入 2 秒
`forward_recovery`，而不是立即把完整控制权交给视觉巡线。

过渡过程：

- 前 0.4 秒平滑衰减原横移速度；
- 中间阶段保持约 10 cm/s 的前向速度，只采用 15% 的视觉横移和偏航修正；
- 最后 0.5 秒用 smoothstep 连续融合到完整视觉指令；
- 前方障碍重新进入 80 cm 或重新形成侵入点簇时，立即恢复本次锁定方向的雷达避障；
- 视觉道路丢失或控制指令进入 road-lost 状态时停止前进。

默认时限：

```bash
--bypass-forward-transition-s 2.0
```

例如改为 3 秒：

```bash
PYTHONPATH=. python3 -u -m experiments.visual_radar_bypass.main \
  --bypass-planner legacy \
  --bypass-forward-transition-s 3.0 \
  --no-record --duration-s 60
```

设置为 `0` 可以恢复原先从雷达避障直接切换到视觉巡线的行为。
