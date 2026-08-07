# 独立真实视觉 + 实体雷达管状物避障优化报告

## 1. Root Cause

`legacy` 在确认障碍前直接返回视觉命令，确认后立即把横移改成固定 `±8cm/s`；
障碍释放后又进入另一套 forward-recovery 命令。完整 encounter 结束后锁向被清除，
下一次 encounter 会根据带噪点簇中位数重新选边，因此视觉横移、雷达横移和重新选边
可以形成左前/右前交替。`SafetyArbiter` 的 80 cm 阈值采用前向走廊内过滤雷达点的
最小 `x`（障碍表面前向距离），只会把正 `vx` 清零，不会反转 `vy`，但会放大阈值
附近的前进速度阶跃。

## 2. Candidate Comparison

| 方案 | 计算量/FPS | 平滑度 | 摇摆风险 | 巡线恢复 | 参数/调试成本 | 改动量 |
| --- | --- | --- | --- | --- | --- | --- |
| basic radar bypass | 较慢，含网格聚类 | 低，硬切换 | 中高 | 直接返回 | 中 | 无 |
| forward recovery | 与 legacy 接近 | 回归较平滑，进入仍硬 | 中 | 较好 | 高，多阶段参数 | 无 |
| smooth sidestep | 最快，单 mask+median | 最好，固定目标+smoothstep | 最低 | clear-hold 后融合 | 最低 | 小 |
| circular tube bypass | 中等，每帧圆拟合 | 理论轨迹平滑，受圆心噪声影响 | 中 | 平滑融合 | 高 | 无 |
| 85/75/65 | 未实现；预计最高 | 取决于多层边界调参 | 低至中 | 复杂 | 最高 | 大 |
| right-half handoff | 低 | 非通用 | 地图相关 | 最终视觉独占 | 中 | 无 |

## 3. Fastest Solution

`smooth_sidestep`。本地相同 synthetic 序列 3000 次结果为 mean 27.33 µs、
p50 25.50 µs、p95 70.00 µs。

## 4. Smoothest Solution

`smooth_sidestep`。绕障侧和目标速度在 encounter 内固定，仅融合权重连续变化；
圆绕行虽然几何上连续，但实时圆心/半径噪声会直接进入切向和径向命令。

## 5. Easiest-to-Tune Solution

`smooth_sidestep`。关键动态参数只有横移目标、渐入、clear-hold、渐出和总超时，
没有拟合半径、RMS、径向增益、圆弧完成角及 emergency 多层阈值耦合。

## 6. Selected Solution

选择锁向 smooth sidestep。活动目标改为 `vx=0, |vy|=10cm/s`，不额外生成 yaw；
它在安全、不摇摆、计算量、命令连续性和调参成本之间均优于当前候选。降低活动
`vx` 是实验安全收紧，不是降低 SafetyArbiter 底线。

## 7. Rejected Alternatives

- basic/forward recovery：保留可运行，但雷达进入仍存在命令阶跃，聚类成本也更高。
- circular：保留可运行；圆拟合失败时使用预设管半径是既有 fallback，且拟合/备用
  路径切换可能产生控制差异。
- 85/75/65：不采用。它以拟合圆心距离为分层依据，而生产 80 cm 安全门使用障碍
  表面 `x`，直接组合会形成定义错位；完整方案还需要新 emergency、切线异常处理和
  多个未经实飞验证的增益。
- right-half handoff：仅适用于已验证地图区域，不作为通用管状物绕障。

## 8. Architecture / State Machine

`NORMAL` 连续两帧确认障碍后只选一次绕行侧，进入 `SHIFT_LEFT/RIGHT`；目标命令
通过 smoothstep 渐入。障碍短时消失由 clear-hold 吸收，随后进入 `BLEND_BACK`；
融合期间障碍重现时恢复原锁定侧。仅在融合权重归零后结束 encounter 并解锁。
持续横移达到上限进入锁存 `TIMEOUT_STOP`，障碍清除后才允许融合返回。

## 9. Files Changed

- `smooth_sidestep.py`：选中状态机、纯横移目标与诊断。
- `main.py`：默认方案、集中参数、结构化事件/调参/性能日志。
- `parameter_registry.py`、`diagnostics.py`：参数来源和低开销诊断。
- `benchmark_bypass.py`：无摄像头、无雷达设备、无飞控的基准与稳定性验证。
- 实验目录 tests、README、SMOOTH_SIDESTEP.md 和本报告。

## 10. Tests & Results

本地算法隔离测试：56 passed。覆盖旧 legacy、forward recovery、circular、
right-half、SafetyArbiter，以及新锁向、融合、相反侧噪声、80 cm 阈值、诊断和参数
登记。板端离线及真实视觉/雷达 dry-run 结果在完成 Git 拉取验证后补充。

## 11. Performance

Windows/Anaconda，同一固定种子 synthetic 序列，warm-up 200，采样 3000：

| 方案 | mean µs | p50 µs | p95 µs |
| --- | ---: | ---: | ---: |
| legacy basic | 260.04 | 239.90 | 507.93 |
| legacy forward recovery | 242.01 | 196.35 | 506.10 |
| smooth sidestep | 27.33 | 25.50 | 70.00 |
| circular tube bypass | 140.79 | 152.60 | 241.90 |

这些数字只表示 planner update，不代表摄像头/NPU 端到端 FPS。

## 12. Smoothness / Stability

完成渐入后对障碍表面 `x=79/81cm` 交替 40 帧：state switch=0、side switch=0、
SafetyArbiter override=0、max `|Δvx|=0`、max `|Δvy|=0`、max
`|Δyaw_rate|=0`。相反侧点簇噪声测试保持同一 encounter 和原锁定侧。

## 13. Agent-added Fallback Registry

| ID | 触发条件 | 行为 | 原因 | 副作用 | 相关参数 |
| --- | --- | --- | --- | --- | --- |
| 无 | 无 | 无 | 本次未新增 fallback | 无 | 无 |

`TIMEOUT_STOP` 是显式安全状态，不是静默 fallback。circular 的预设半径 fallback 是
任务前既有行为，未修改。

## 14. Parameter Registry

| 参数 | 当前值 | 来源 | 作用 | 增大效果 | 减小效果 | 实飞调节 |
| --- | ---: | --- | --- | --- | --- | --- |
| road_half_width_cm | 25 | EXISTING_PROJECT | 道路几何 | 扩大门限 | 缩小门限 | 否 |
| intrusion_half_width_cm | 75 | EXISTING_PROJECT | 横向触发范围 | 更早触发 | 更晚触发 | 否 |
| clearance_cm | 75 | EXISTING_PROJECT | 安全包络 | 更保守 | 更激进 | 否 |
| activity_half_width_cm | 90 | EXISTING_PROJECT | 活动横移范围 | 允许更远 | 更早受限 | 否 |
| min_x_cm/lookahead_cm | 10/180 | EXISTING_PROJECT | 前向门限 | 缩小/扩大观测 | 扩大/缩小观测 | 否 |
| min_points | 3 | EXISTING_PROJECT | 障碍证据 | 少误报多漏检 | 更敏感 | 否 |
| side_deadband_cm | 5 | EXISTING_PROJECT | 中心死区 | 更多默认侧 | 更易受噪声影响 | 否 |
| center default side | right | EXISTING_PROJECT | 中心障碍确定性 | 不适用 | 不适用 | 否 |
| shift_forward_speed_cm_s | 0 | UNVERIFIED_TUNING | 活动前进目标 | 更快但接触 80cm 门 | 更保守 | 是 |
| shift_lateral_speed_cm_s | 10 | UNVERIFIED_TUNING | 横移目标 | 更快/阶跃更大 | 更慢 | 是 |
| ramp_in_s | 1.0 | UNVERIFIED_TUNING | 渐入 | 更平滑但更慢 | 响应更快 | 是 |
| clear_hold_s | 2.0 | UNVERIFIED_TUNING | 掉点迟滞 | 更稳但横移更久 | 更早返回 | 是 |
| blend_back_s | 2.5 | UNVERIFIED_TUNING | 视觉交还 | 更平滑 | 更快 | 是 |
| max_sidestep_s | 9.0 | UNVERIFIED_TUNING | 超时停车 | 允许更远 | 更早停车 | 是 |
| activate_frames | 2 | UNVERIFIED_TUNING | 连续确认 | 更稳更慢 | 更敏感 | 是 |
| min_confidence | 0.4 | EXISTING_PROJECT | 启动视觉条件 | 更严格 | 更宽松 | 否 |
| nominal_dt_s | 0.1 | UNVERIFIED_TUNING | 首帧积分 | 首步更大 | 首步更小 | 是 |
| tuning_log_every_n | 2 | UNVERIFIED_TUNING | 命令日志采样 | 更少 I/O | 更多细节 | 是 |
| radar_snapshot_every_n | 5 | UNVERIFIED_TUNING | 点云快照采样 | 更少 I/O | 更多细节 | 是 |

运行时 manifest 会写出逐项完整 registry，包括未在表中合并显示的雷达场参数。

## 15. Safety-sensitive Parameters

保持不变：前向停止 80 cm、减速 150 cm、减速上限 10 cm/s、侧向停止 45 cm、
雷达超时 0.5 s、最大 `vx/vy/yaw_rate=14/10/10`。活动 `vx` 从 8 降至 0；活动
`vy=10` 未超过既有上限。所有安全项在运行时 registry 中单独标记。

## 16. Logging / Diagnostics

EVENT LOG 查看 state transition、side lock/unlock、encounter start/end、timeout 和
Safety override。调参日志查看 obstacle/nearest distance、desired/planned/safe/final、
blend alpha、选边原因和命令增量。性能字段为 `planner_elapsed_us`。命令默认 5 Hz、
雷达快照默认 2 Hz，避免原先每帧重复同步 flush。

## 17. Recommended Tuning Order

1. 先保持 Safety 参数不变，检查雷达距离/点数和 side-blocked 日志。
2. 调 `shift_lateral_speed`：过小表现为超时仍未横移出包络，过大表现为 Δvy 和侧向
   仲裁增多。
3. 调 `ramp_in`：过小入口阶跃大，过大则接近障碍仍未达到横移目标。
4. 调 `clear_hold`：过小会对掉点敏感，过大会无障碍后继续横移。
5. 调 `blend_back`：过小视觉交还阶跃大，过大则恢复巡线迟缓。
6. 最后调整 `max_sidestep` 和日志采样；不得用降低 80/45 cm 阈值解决行为问题。

## 18. Remaining Risks

- 尚未真实飞行验证这些 `UNVERIFIED_TUNING` 参数。
- 单矩形中位数依赖实验场景“管状物附近无相邻障碍”的前提。
- 纯横移可能比圆绕行慢，但能避免前向安全门振荡。
- SessionRecorder 仍同步 flush；降低采样后需在板端确认端到端 FPS。

## 19. Git Diff Review

- 修改限定在 `experiments/visual_radar_bypass/`，生产视觉和 Safety 文件未修改。
- 无 Agent 新增 fallback；无静默安全阈值修改。
- 经验参数均登记为 `UNVERIFIED_TUNING`，关键数值集中配置。
- 用户原有四个未提交文档不会暂存或提交。
- 最终提交前再次执行 whitespace、diff、测试和板端无飞控验证审计。
