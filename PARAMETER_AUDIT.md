# 默认参数审计报告

**项目**: Cooper_drone  
**审计日期**: 2026-06-07  
**范围**: 全部 .py 文件 (~55 个)，对照 `HARDWARE_INTERFACE.md` / `PROJECT_STATUS.md` / `IMPLEMENTATION_PLAN.md` / `FC_INTEGRATION_TEST_PLAN.md`  
**最新进展**: 所有 4 个占位值已修复; NPU 适配因 Debian/OpenSTLinux 不兼容，决策切换 OS

---

## 1. 审计分类标准

| 标记 | 含义 |
|------|------|
| 🔴 **占位值** | 明显的临时值，或与实际硬件不符合的值，需填入真实值 |
| 🟡 **可疑值** | 与文档不一致、不同文件矛盾、或无文档记录的调优参数 |
| 🟢 **有据可查** | 在文档中有明确记录，或代码间一致可解释 |

---

## 2. 🔴 占位值 / 与实际硬件不符

### 2.1 `obstacle_classifier.py` — 整个模块为空桩 ✅ 已删除

**文件**: ~~obstacle_classifier.py~~（已于 2026-05-24 删除）  
**严重度**: 🔴 高 → 已解决

模块内容（删除前）:

```python
"""Compatibility placeholder for future obstacle classification."""

class ObstacleClassifier:
    def classify_points(self, points_body_cm, now_s):
        return []  # ← 永远返回空列表

class ObstacleClassifierConfig:
    pass  # ← 无配置字段
```

**追踪结论**: 全项目零引用、零导入、零调用。`local_world_model.py` 是 `Safety.RadarObstacleField` 的薄封装，不包含 classifier 引用。IMPLEMENTATION_PLAN.md 架构图中描述的 `classifier.classify_points: 欧氏聚类→wall/pole/block` 数据流从未实现。

**处理**: 直接删除该模块。障碍物类型识别（wall/pole/block 分类）保留为开放任务目标，详见 IMPLEMENTATION_PLAN.md §6。

---

### 2.2 `road_follow_main.py:22` — 摄像头 index 默认值错误 ✅ 已修复

**文件**: [road_follow_main.py:22](road_follow_main.py#L22)

```python
parser.add_argument("--camera-index", type=int, default=7)  # 原 default=0
```

**实际硬件**: 路径识别摄像头 = `/dev/video7` (cv2 index 7)，障碍物摄像头 = `/dev/video9` (cv2 index 9)。

**处理**: 已将默认值从 `0` 改为 `7`。

---

### 2.3 `goal_nav_main.py:19` — 目标坐标默认值为演示值 ✅ 已标注

**文件**: [goal_nav_main.py:19](goal_nav_main.py#L19)

```python
parser.add_argument("--goal-x-cm", type=float, default=200.0,
                    help="目标在机体前方距离 (cm)。默认 200 仅供 dry-run 测试，实飞须显式指定")
```

**处理**: 保留 200cm 默认值（dry-run 测试用），通过 `help` 字符串明确标注实飞须显式指定。

---

### 2.4 `get_road_perception(flight_height_m=1.5)` — 参数未使用 ✅ 已修复

**文件**: [road_perception.py:1311](road_perception.py#L1311)

**原问题**:
```python
def get_road_perception(
    ...
    flight_height_m: float = 1.5,   # ← 被显式丢弃 (_ = flight_height_m)
    ...
):
    ...
    _ = flight_height_m              # ← 未进入任何计算路径
```

两个调用方均未传入该参数，`compute_meters_per_pixel()` 已有但未接入。

**处理**:
1. `flight_height_m` 默认值改为 **2.0m**（典型飞行高度）
2. 函数内 `_ = flight_height_m` 替换为：当 `cfg.meters_per_pixel_x is None` 时，调用 `compute_meters_per_pixel(row_from_bottom=120, height_m=flight_height_m)` 自动计算
3. `compute_meters_per_pixel()` 新增 `height_m` 覆盖参数，支持飞行时高度与标定高度不同
4. `road_follow_main.py` 新增 `--flight-height-m` (default=2.0) 参数并传入 `get_road_perception()`

---

## 3. 🟡 可疑值 / 无文档记录

### 3.1 道路感知阈值常量（共 18 个）

**文件**: [road_perception.py:143-164](road_perception.py#L143-L164)

| 常量 | 值 | 说明 |
|------|:--:|------|
| `INP_SIZE` | 320 | YOLO 输入尺寸，来源于模型元数据，非实测 |
| `CONF_THRESH` | 0.4 | 检测置信度阈值 |
| `IOU_THRESH` | 0.45 | NMS IoU 阈值 |
| `MASK_THRESH` | 0.5 | 分割掩码二值化阈值 |
| `MIN_AREA_RATIO` | 0.02 | 最小道路面积比例 |
| `MIN_ROAD_PX_PER_ROW` | 12 | 每行最少道路像素 |
| `BOTTOM_RATIO` | 0.10 | 底部区域比例 |
| `BOTTOM_IGNORE_RATIO` | 0.03 | 底部忽略比例 |
| `BOTTOM_ERROR_Y_MIN_RATIO` | 0.82 | pixel_error 计算窗口下界 |
| `BOTTOM_ERROR_Y_MAX_RATIO` | 0.96 | pixel_error 计算窗口上界 |
| `ANGLE_Y_MIN_RATIO` | 0.60 | 角度计算窗口下界 |
| `CONTROL_ANGLE_Y_MIN_RATIO` | 0.72 | 控制角度窗口下界 |
| `CONTROL_ANGLE_Y_MAX_RATIO` | 0.98 | 控制角度窗口上界 |
| `CENTERLINE_SCAN_Y_MAX_RATIO` | 0.97 | 中线扫描上限 |
| `MIN_FIT_PTS` | 5 | 线拟合最小点数 |
| `FORK_INTERVAL_ROWS_MIN` | 8 | 岔路检测最小行数 |
| `FORK_WIDTH_GROWTH_RATIO` | 1.6 | 岔路宽度增长比 |
| `WIDE_INTERVAL_RATIO` | 0.45 | 宽路段判定比 |
| `WIDTH_JUMP_LOCK_RATIO` | 1.45 | 宽度跳变锁定比 |
| `CENTER_SMOOTH_ALPHA` | 0.65 | 中线平滑系数 |

**状态**: 均无文档记录来源，无离线测试图片集验证。这些值可能是从 YOLO11-seg 示例代码继承或经验调参得出，但缺少调参记录。

**建议**: 不要求逐个文档化（这是视觉领域算法的常规超参），但应在首次实飞路测后固化并记录关键参数的调优日志。

---

### 3.2 偏移补偿相关参数

**文件**: [road_perception.py:65-67](road_perception.py#L65-L67)

| 参数 | 默认值 | 问题 |
|------|:------:|------|
| `correction_sign` | 1.0 | 符号未实机验证，写反会导致纠偏方向错误（IMPLEMENTATION_PLAN §7 已标注风险） |
| `max_correction_px` | 120.0 | 120px 相当于 640 画面的 18.75%，来源不明 |
| `pipeline_latency_s` | 0.0 | 零意味着不做延迟预测补偿，实际 ONNX 推理到飞控输出有非零延迟 |

---

### 3.3 YOLO 模型路径 — 文件是否存在未确认

**多处引用**: `road_perception.py:141`, `road_follow_main.py:30`

```python
MODEL_PATH = "FlightController/Solutions/model/road_yolo11n_seg.onnx"
```

IMPLEMENTATION_PLAN §7 标记为 🔴 高风险——模型文件未确认存在。若不存在，整个视觉链路无法测试。

---

### 3.4 `LocalPlanner.camera_fov_deg=70.0` — 与实测不一致

**文件**: [LocalPlanner.py:17](FlightController/Solutions/LocalPlanner.py#L17)

实测标定 HFOV = **68°**，代码默认 **70°**。2° 误差在实际使用中影响微小（相对 70° 仅 3% 偏差），但说明该值非来源于实测。

---

### 3.5 `goal_nav_main.py:37` — `yaw_kp=0.5` 无调参记录

```python
parser.add_argument("--yaw-kp", type=float, default=0.5)
```

IMPLEMENTATION_PLAN §4 仅记录了道路循线的 `pixel_kp_yaw=0.08` 和 `angle_kp_yaw=0.4`，目标导航的 yaw P 增益 0.5 无调参来源。

---

### 3.6 主循环频率不一致

| 文件 | 默认 loop_hz | 备注 |
|------|:-----------:|------|
| `road_follow_main.py` | **10** | 视觉推理路径 |
| `goal_nav_main.py` | **10** | 雷达导航路径 |
| `test_dual_radar.py` | **30** | 纯雷达避障 |
| IMPLEMENTATION_PLAN §4 | **30** | 参数基准表 |

10Hz vs 30Hz 各有道理（视觉 ONNX 推理慢、雷达快），但差异未在任何文档中解释。

---

### 3.7 Map_Circle 参数 — 小部分无文档

**文件**: [LDRadar_Resolver.py:268-277](FlightController/Components/LDRadar_Resolver.py#L268-L277)

| 常量 | 值 | 状态 |
|------|:--:|------|
| `ACC = 3` | 1080 bins (360×3) | 🟢 文档记录 |
| `REMAP = 2` | 点扩散 2 bins | 🟢 文档记录 |
| `timeout_time = 0.15` | 150ms | 🟢 文档记录 (PROJECT_STATUS §6.3) |
| `confidence_threshold = 0` | 不过滤 | 🟡 无解释（D500 不输出置信度字段，设为 0 等价于关闭） |
| `distance_threshold = 10` | 10mm | 🟡 无解释（同一 bin 内新数据覆盖旧数据的最小距离差阈值） |

---

## 4. 🟢 已确认有据可查的关键参数

以下参数在代码和文档间一致，仅列代表性条目：

| 参数 | 值 | 文档来源 |
|------|:--:|------|
| 上雷达串口 | `/dev/ttySTM4` | HARDWARE §4 |
| 下雷达串口 | `/dev/ttySTM9` | HARDWARE §4 |
| 雷达波特率 | 230400 | HARDWARE §4 |
| FC 串口波特率 | 500000 | STATUS §9, PLAN §9 |
| FC VID:PID | `66CC:2233` | STATUS §9 |
| D500 VID:PID | `10C4:EA60` | HARDWARE §4 |
| 上雷达安装位姿 | (0,0)cm, yaw=0°, mirror=N | HARDWARE §3.4 |
| 下雷达安装位姿 | (0.96,0.15)cm, yaw=0°, mirror=Y | HARDWARE §3.4 |
| 摄像头分辨率 | 640×480 | HARDWARE §5.2 |
| 前向偏移 | 10cm | HARDWARE §5.4 |
| 光轴倾角 α | 30.27° | HARDWARE §5.4 |
| VFOV/2 (β) | 27.54° | HARDWARE §5.4 |
| HFOV | 68° | HARDWARE §5.4 |
| 摄像头安装高度 | 17cm | HARDWARE §5.4 |
| obstacle_stop_distance | 80cm | PLAN §4, Safety.py |
| obstacle_slow_distance | 150cm | PLAN §4, Safety.py |
| body_x/y_half | 25cm | PLAN §4 |
| forward_corridor_half_width | 50cm | PLAN §4 |
| radar_timeout | 0.5s | PLAN §4 |
| max_distance | 300cm | PLAN §4 |
| 机身屏蔽 | ±25cm | PLAN §4 |
| FC mode HOLD_POS | 2 | STATUS §9.1 |
| 双雷达融合频率 | 30Hz | PLAN §4 |
| 凌霄协议帧头 | `AA 22` | STATUS §4 |
| 心跳间隔 | 250ms | STATUS §9.3 |

---

## 5. 统计摘要

| 分类 | 数量 | 说明 |
|------|:--:|------|
| 🔴 占位值 | **0** | 全部已修复 |
| 🟡 可疑值 | **30+** | 18 个视觉阈值常量 + 偏移补偿 3 个 + 模型路径 + FOV 偏差 + 频率不一致 + yaw kp + 若干 Map_Circle 参数 |
| 🟢 有据可查 | **50+** | 雷达接线、摄像头参数、机身尺寸、避障距离、协议参数 |

**总体评价**: 代码库整体质量较高，没有到处散布的 `// TODO` 或 `magic_number = 999` 之类典型占位。主要问题集中在：

1. **视觉算法超参**（18 个阈值）未文档化，但这属于算法调优范畴，偏低优先级
2. **偏移补偿**的 `correction_sign` 和 `pipeline_latency_s` 需实飞后标定
3. **摄像头 index 默认值已修正为 7**（原为 0），但障碍物识别摄像头 `/dev/video9` (index=9) 无对应入口参数
