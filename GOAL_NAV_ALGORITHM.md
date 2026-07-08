# 目标导航算法详解

> 本文档详细解析 `goal_nav_main.py` 中的步进状态机（`_next_step_command`）和 `RelativeGoalNavigator.py` 中的核心避障规划算法。

---

## 1. 系统分层概览

```
主循环 (goal_nav_main.py)
  │
  ├── 雷达点云采集 → RadarObstacleField 过滤
  │
  ├── _next_step_command()  ← 步进状态机 (plan → execute → hold)
  │     └── RelativeGoalNavigator.update()  ← 候选方向搜索
  │
  ├── SafetyArbiter.filter()  ← 安全仲裁
  │     ├── 硬故障检查 (FC断连 / 非HOLD_POS / 雷达超时 / 倾角过大)
  │     ├── 前方障碍 <80cm → vx=0  (OBSTACLE_STOP)
  │     └── 前方障碍 <150cm → vx≤12 (OBSTACLE_SLOW)
  │
  └── send_command_safely() → FC 实时控制指令
```

## 2. 核心数据结构

### 2.1 `StepRuntime` — 步进状态机的运行状态

```python
@dataclass
class StepRuntime:
    phase: str = "plan"              # "plan" | "execute" | "hold"
    phase_until_s: float = 0.0       # 当前 phase 的截止时间 (perf_counter)
    active_command: object | None    # 当前执行中的 Command
    mission_done: bool = False       # 任务终止标志
    done_reason: str = ""            # 终止原因 (如 "blocked_no_path_dir_0_clear_72")
```

### 2.2 `Command` — 速度指令

```python
@dataclass(frozen=True)
class Command:
    vx_cm_s: float         # 前向速度，单位 cm/s
    vy_cm_s: float         # 横向速度（本导航器恒为 0）
    vz_cm_s: float         # 垂向速度（恒为 0）
    yaw_rate_deg_s: float  # 偏航角速率，单位 deg/s
    reason: str            # 指令来源标识字符串
```

### 2.3 `_DirectionEval` — 单个候选方向的评估结果

```python
@dataclass(frozen=True)
class _DirectionEval:
    angle_deg: float         # 候选方向角度 (deg, 机体坐标系)
    allowed: bool            # 该方向是否可通过
    tube_clearance_cm: float # 沿该方向的管状区域内的最近障碍物距离 (cm)
    cost: float              # 综合代价 (heading + clearance + switch)
```

---

## 3. 步进状态机 `_next_step_command()`

### 3.1 状态转移图

```
                         ┌────────────────────────────────────┐
                         │                                    │
                         ▼                                    │
                    ┌──────────┐                               │
          ┌────────→│   PLAN   │                               │
          │         │          │                               │
          │         │ navigator.update()                       │
          │         │  ──→ Command(vx, 0, 0, 0/yaw)           │
          │         │                                         │
          │         │ 无路径? ──→ mission_done = True ──→ 退出 │
          │         └─────┬────┘                              │
          │               │ 有路径                             │
          │               ▼                                    │
          │         ┌──────────┐                               │
          │         │ EXECUTE  │                               │
          │         │          │──────────────────────────────→│
          │         │ 每循环重新评估:                           │
          │         │   navigator.update() → live_plan         │
          │         │                                         │
          │         │ ① live_plan 无路径?                      │
          │         │    → mission_done = True → 退出          │
          │         │                                         │
          │         │ ② _should_interrupt_step()?             │
          │         │    原为前进但新计划要求停/转              │
          │         │    → 打断，切 HOLD                        │
          │         │                                         │
          │         │ ③ now_s < phase_until_s?                │
          │         │    → 继续执行原指令                       │
          │         │                                         │
          │         │ ④ 超时完成                                │
          │         │    → 切 HOLD                             │
          │         └───┬───┬───────┘                         │
          │             │   │                                  │
          │  打断后      │   │ 超时后（等待 hold_after_step_s）  │
          │  切 HOLD     │   │                                  │
          │             ▼   ▼                                  │
          │         ┌──────────┐                               │
          └─────────│   HOLD   │                               │
                    │          │                               │
                    │ 零速等待 hold_after_step_s               │
                    │ 超时后 → PLAN                            │
                    └──────────┘                               │
```

### 3.2 各阶段详解

#### **PLAN 阶段**（第 442-461 行）

```python
planned = navigator.update(radar_field, now_s=now_s)
```

- 调用 `RelativeGoalNavigator.update()` 获取规划结果
- **无路径** → `runtime.mission_done = True`，任务终止
- **有路径** → 计算步进时长 → 进入 EXECUTE 阶段

#### **EXECUTE 阶段**（第 420-440 行）

每循环都用最新雷达数据 **重新评估** 当前执行中的指令是否仍然安全：

```
live_plan = navigator.update(radar_field)
```

- **无路径** → 任务终止
- **_should_interrupt_step()** → 打断条件：
  - 当前正在前进（`active_vx > 0`），但新计划要求停止或转向 → 打断
  - 核心判断：`live_vx <= 1e-6 or live_yaw > 1e-6`
- **超时完成** → 正常结束 → 切 HOLD

#### **HOLD 阶段**（第 414-418 行）

```
Command.zero("step_hold")  →  零速等待
```

- 等待 `hold_after_step_s`（默认 0.35s）
- 超时后自动回到 PLAN 阶段

#### _should_interrupt_step() 详解（第 475-485 行）

```
打断条件（三个全满足时打断）:
  ① active_command ≠ None
  ② 当前正在前进 (active_vx > 0)
  ③ 新计划要求: 停止前进 (live_vx ≤ 0) 或 转向 (live_yaw > 0)
```

**设计意图**：在 EXECUTE 阶段，无人机正在按旧计划前进。但障碍物是动态变化的——可能新出现的障碍物让原路径不再安全。此时不能等到步进超时才反应，必须立即打断。

### 3.3 步进时长计算 `_step_duration_for_command()`

```python
if vx > 0:
    duration = forward_step_cm / vx       # 例: 40cm / 10cm/s = 4.0s
    duration = clip(duration, 0.35s, 2.0s)  # 钳位到 [min_step_s, max_step_s]
elif yaw > 0:
    duration = turn_step_s (0.45s)        # 转向固定时长
else:
    duration = hold_after_step_s (0.35s)  # 零速等待
```

**默认参数下的行为**：前进速度 10 cm/s，步长 40 cm → 理论时长 4.0s → 被钳位到 **2.0s**。意味着每次前进最多持续 2 秒就必须停下来重新评估——这是一种**保守的步进策略**，适合障碍物密集的未知环境。

### 3.4 安全仲裁打断 `_step_was_safety_interrupted()`

```python
return active_vx > 0.0 and safe_vx <= 0.0 and "front_obstacle_stop" in reason
```

发生在主循环第 310 行（不在状态机内部）：
- navigator 输出前进指令
- SafetyArbiter 检测前方 < 80cm → 强制 vx=0
- 主循环检测到这个矛盾 → 强制切 HOLD → 重新 PLAN

**这是 SafetyArbiter 与 Navigator 之间的最后一道防线**。Navigator 可能因为方向投影的管状区域与 SafetyArbiter 的前方走廊定义不同而"乐观"，此时 SafetyArbiter 直接否决。

---

## 4. `RelativeGoalNavigator` 核心算法

### 4.1 总体流程图

```
update(radar_field)
  │
  ├── forward_test 模式? → 仅评估 0° 方向，直接返回
  │
  ├── continuous_forward? 否 + 目标距离 < arrive_distance_cm?
  │     → 返回 "relative_goal_reached" (零速)
  │
  ├── 计算 goal_angle = atan2(goal_y, goal_x)，钳位到 ±75°
  │
  ├── [阻塞释放检查] 之前 blocked 且 front 0° clearance ≥ 90cm?
  │     → 解除阻塞，恢复前进
  │
  ├── _select_direction(goal_angle)  ← 核心方向选择
  │     │
  │     ├── 对所有候选角度 (-75°~+75°, 2°步长) 调用 _evaluate_direction()
  │     │
  │     ├── 无 allowed 方向 → 选 clearance 最大的
  │     │     stop_when_no_path? → "blocked_no_path_*" (零速)
  │     │     否则 → "blocked_turn_*" (原地偏航搜索)
  │     │
  │     ├── 有 allowed 方向 → 选 min(cost) 的
  │     │
  │     └── 方向切换迟滞 → 如上次方向与最优方向成本差 ≤ 5.0，保持上次方向
  │
  ├── selected 不可行 → 零速 (任务终止)
  │
  ├── 需要转向 → yaw_rate 指令 (零前进速度)
  │
  ├── 已对准 → 重新评估 0° 方向
  │     front 0° 被阻塞? → 零速 hold 或重新转向
  │
  └── front 0° 通畅 → 速度指令 (vx, 0, 0, 0)
```

### 4.2 候选方向生成

```python
_candidate_angles_deg():
    half_fov = (scan_fov_deg / 2) - candidate_edge_margin_deg
            = 75° - 0° = 75°                 # 默认 scan_fov=150, margin=0
    step = candidate_step_deg = 2.0°
    
    生成: -75°, -73°, -71°, ..., -1°, 1°, ..., 73°, 75°
    确保包含 0°
    总共: 75 个候选方向
```

### 4.3 方向评估 `_evaluate_direction()` — 管状区域碰撞检测

这是算法的**数学核心**：

#### Step 1: 点云过滤 `_front_scan_points()`

```
输入: RadarObstacleField 过滤后的机体坐标点云
过滤条件:
  ① 距离 > min_obstacle_distance_cm (10cm)  ← 滤除雷达壳体反射
  ② 距离 ≤ sqrt(lookahead² + clearance²)    ← 只关心前方区域
  ③ |角度| ≤ scan_half_fov (75°)            ← 仅前向 150°
```

#### Step 2: 管状区域投影

```
以候选方向 angle_deg 为轴，建立管状区域:
  
  unit = [cos(θ), sin(θ)]       ← 方向单位向量
  normal = [-sin(θ), cos(θ)]    ← 法线方向
  
  对每个点 p:
    along  = p · unit           ← 沿方向的投影距离
    lateral = |p · normal|      ← 垂直偏离距离
  
  点在管内的条件:
    along ∈ (min_obstacle_distance, lookahead_cm]
    lateral ≤ obstacle_clearance_cm (80cm)
  
  tube_clearance = min(管内的 along)   ← 最近障碍物沿方向的深度
                  = lookahead_cm      ← 无物时返回最大探测距离
```

```
      机体
       │
       ├──── 管状区域 (半宽 80cm) ────→ 方向 θ
       │         ┌──────────┐
       │         │          │
       │         │ 沿方向    │      × 障碍物 (tube_clearance = 120cm)
       │         │ 0 → 220cm│
       │         │          │
       │         └──────────┘
       │
```

#### Step 3: 通过判定

```python
allowed = tube_clearance > obstacle_clearance_cm (80cm)
```

即：该方向的管状区域内，**至少前 80cm 是畅通的**。

#### Step 4: 代价计算

```python
cost = heading_cost + clearance_cost + switch_cost

其中:
  heading_cost  = |angle - goal_angle|           # 偏离目标方向惩罚
  clearance_cost = linear(tube_clearance)         # 越近越危险 (0 at ≥150cm, 120 at 80cm)
  switch_cost   = |angle - last_angle| × 0.15    # 方向切换惩罚
```

### 4.4 方向选择 `_select_direction()` — 两层策略

#### 第一层：无可通过方向

```
选 clearance 最大的方向，用于偏航搜索:
  key = (clearance, -|angle-goal_angle|, -tiebreak)
```

#### 第二层：有可通过方向

```
选 min(cost) 的方向:
  key = (cost, |angle-goal_angle|, tiebreak)

方向切换迟滞 (Hysteresis):
  如果上次选择的方向与最佳方向的 cost 差 ≤ switch_cost_margin (5.0)
  → 保持上次方向 (避免频繁切换)
```

**设计意图**：5.0 的迟滞带很小（相当于 ~5° 的 heading 差或 ~4cm 的 clearance 差），在轻微扰动下不会切换方向，但遇到显著变化时能及时响应。

### 4.5 转向判定 `_should_turn()` — 迟滞比较器

```
进入转向:  |selected_angle| > align_start_deg (10°)
退出转向:  |selected_angle| ≤ align_stop_deg (3°)
```

施密特触发器式迟滞，**防止在阈值边缘抖动**。3° 的退出阈值比 10° 的进入阈值严格得多——确保飞机真正对准了才停止转向。

### 4.6 阻塞与释放机制

Navigator 维护一个内部状态 `_blocked`：

```
触发阻塞:
  _can_move_forward() → front.tube_clearance ≤ 80cm → self._blocked = True
  或 selected direction 不可行 → self._blocked = True

解除阻塞:
  front 0° clearance ≥ clearance_release_cm (90cm) 且目标方向已对准
  → self._blocked = False → 输出 "forward_release_*"

阻塞期间:
  _can_move_forward() → front.clearance < 90cm → 返回 False
  → 即使 clearance 在 80~90cm 之间，也不前进 (hysteresis)
```

这就是之前飞行日志中看到的现象：
```
forward_clear_114  →  SafetyArbiter: front_obstacle_stop
→ _blocked = True → clearance 需 ≥ 90 才能恢复
→ 下次 plan: 114 ≥ 90 → _blocked 解除 → 又被 safety 打断
```

### 4.7 速度整定 `_speed_from_clearance()`

速度随前方 clearance 线性缩放：

```
clearance ≤ 80cm   → vx = 0
clearance ≥ 150cm  → vx = cruise_speed_cm_s (20cm/s)
80cm < clearance < 150cm:
  ratio = (clearance - 80) / (150 - 80)
  vx = min_forward_speed + ratio * (cruise - min_forward)
     = 8 + ratio * (20 - 8)
     = 8 + ratio * 12

例: clearance = 100cm → ratio = 0.286 → vx ≈ 11.4cm/s
例: clearance = 120cm → ratio = 0.571 → vx ≈ 14.9cm/s
```

### 4.8 偏航控制 `_yaw_command()`

```
yaw = clip(angle * yaw_kp, -yaw_rate_limit, +yaw_rate_limit)
    = clip(angle * 0.5, -25, +25)

最小值保护:
  if 0 < |yaw| < min_turn_yaw_rate (6°/s):
    yaw = ±6°/s  ← 确保转向有可感知的速度

对准窗口:
  if |angle| ≤ align_stop_deg (3°): yaw = 0
```

---

## 5. 信息通路全链路

```
 ┌──────────────────────────────────────────────────────────┐
 │  双雷达 (D500 ×2)                                        │
 │    upper: UART4 /dev/ttySTM4 @230400 → Map_Circle 1080bin │
 │    lower: UART9 /dev/ttySTM9 @230400 → Map_Circle 1080bin │
 └────────────────────┬─────────────────────────────────────┘
                      │
                      ▼
 ┌──────────────────────────────────────────────────────────┐
 │  MultiRadar.get_obstacle_points_body_cm(max_distance=300) │
 │    vstack 融合 → 机身屏蔽 (±25cm) → 距离裁剪 (≤300cm)    │
 │    输出: (N, 2) 机体坐标系点云, 单位 cm                   │
 └────────────────────┬─────────────────────────────────────┘
                      │
                      ▼
 ┌──────────────────────────────────────────────────────────┐
 │  RadarObstacleField.update(points)                        │
 │    _normalize_points → 距离过滤 (>300cm 丢弃)              │
 │    mask_body_reflection: 滤除 |x|<25 & |y|<25             │
 │    输出: filtered_points_body_cm                          │
 └────────────────────┬─────────────────────────────────────┘
                      │
                      ▼
 ┌──────────────────────────────────────────────────────────┐
 │  _next_step_command() — 步进状态机                        │
 │    PLAN: navigator.update(radar_field) → Command           │
 │    EXECUTE: 每循环 re-evaluate → 打断/完成/继续            │
 │    HOLD: 零速等待 → PLAN                                  │
 │    mission_done: blocked_no_path → 任务终止               │
 └────────────────────┬─────────────────────────────────────┘
                      │  desired = Command(vx, vy, yaw, reason)
                      ▼
 ┌──────────────────────────────────────────────────────────┐
 │  SafetyArbiter.filter(desired, flight, radar_field)        │
 │                                                           │
 │  层级1 — 硬故障 (Hard Stop):                               │
 │    FC断连 | 非HOLD_POS模式 | 雷达超时(>0.5s) | 倾角>25°    │
 │    → Command.zero("safety_stop:*")                        │
 │                                                           │
 │  层级2 — 全局钳位:                                         │
 │    vx clipped to ±35, vy to ±25, yaw to ±30               │
 │                                                           │
 │  层级3 — 前方障碍:                                         │
 │    nearest_forward < obstacle_stop (80cm)                  │
 │      → vx=0 (OBSTACLE_STOP)                               │
 │    nearest_forward < obstacle_slow (150cm)                 │
 │      → vx ≤ slow_speed_limit (12cm/s) (OBSTACLE_SLOW)     │
 └────────────────────┬─────────────────────────────────────┘
                      │  safe = SafetyResult(command, state, reasons)
                      ▼
 ┌──────────────────────────────────────────────────────────┐
 │  send_command_safely(fc, command, arbiter, health)        │
 │    非 dry_run: fc.send_realtime_control_data(vx,vy,vz,yaw) │
 │    → 凌霄飞控 (二进制协议, 500000 baud)                     │
 └──────────────────────────────────────────────────────────┘
```

---

## 6. 关键参数速查表

| 参数 | 默认值 | 所在位置 | 说明 |
|------|--------|----------|------|
| **扫描/探测** | | | |
| `scan_fov_deg` | 150° | RelNav | 前向扫描视场角（±75°） |
| `lookahead_cm` | 220 | RelNav | 管状区域最大探测深度 |
| `candidate_step_deg` | 2.0° | RelNav | 候选方向角步长（共75个方向） |
| `max_distance_cm` | 300 | RadarField | 雷达点云最大距离 |
| **安全阈值** | | | |
| `obstacle_clearance_cm` | 80 | RelNav | 管状区域半宽 / 前方安全距离 |
| `clearance_release_cm` | 90 | RelNav | 阻塞解除所需的前方 clearance |
| `avoid_begin_distance_cm` | 150 | RelNav | 开始减速 / 计入 clearance cost |
| `obstacle_stop_distance_cm` | 80 | Safety | SafetyArbiter 强制停止距离 |
| `obstacle_slow_distance_cm` | 150 | Safety | SafetyArbiter 强制减速距离 |
| **速度/控制** | | | |
| `cruise_speed_cm_s` | 20 | RelNav | 巡航速度 |
| `min_forward_speed_cm_s` | 8 | RelNav | 接近障碍时的最低前进速度 |
| `yaw_rate_limit_deg_s` | 25 | RelNav | 偏航角速率上限 |
| `yaw_kp` | 0.5 | RelNav | 偏航 P 增益 |
| `min_turn_yaw_rate_deg_s` | 6.0 | RelNav | 偏航最小速率（防死区） |
| **转向迟滞** | | | |
| `align_start_deg` | 10° | RelNav | 进入转向阈值 |
| `align_stop_deg` | 3° | RelNav | 退出转向阈值 |
| **代价权重** | | | |
| `clearance_cost_weight` | 120 | RelNav | clearance 不足的代价乘数 |
| `switch_cost_weight` | 0.15 | RelNav | 方向切换的单位角度代价 |
| `switch_cost_margin` | 5.0 | RelNav | 方向保持迟滞带 |
| **步进** | | | |
| `forward_step_cm` | 40 | goal_nav_main | 每次步进的前进距离 |
| `min_step_s` | 0.35s | goal_nav_main | 最短步进时长 |
| `max_step_s` | 2.0s | goal_nav_main | 最长步进时长 |
| `turn_step_s` | 0.45s | goal_nav_main | 转向步进时长 |
| `hold_after_step_s` | 0.35s | goal_nav_main | 步间停顿 |

---

## 7. 已知问题与设计决策

### 7.1 Navigator 与 SafetyArbiter 判据不一致

- **Navigator**：管状投影 → `tube_clearance`（方向 θ 的走廊内最近障碍物）
- **SafetyArbiter**：`nearest_forward_obstacle_cm()` → 纯前方走廊 `x > min_x & |y| < half_width`

当障碍物偏斜时，Navigator 可能判定某方向可通过（114cm），但 SafetyArbiter 从纯前方走廊看可能 < 80cm，导致 step 被安全层打断。

### 7.2 方向迟滞的利弊

`switch_cost_margin=5.0` 的迟滞带在大多数情况下是好策略（防止来回摆动），但在窄通道中可能让 Navigator 坚持一个次优方向，反复被 SafetyArbiter 打断。

### 7.3 保守的前进策略

每次前进最多 2 秒（`max_step_s`），之后必须停下来重新 PLAN。这适合障碍物密集的环境，但在开阔地带会导致"走走停停"的行为。如果要更流畅的长距离飞行，可调大 `--max-step-s`。
