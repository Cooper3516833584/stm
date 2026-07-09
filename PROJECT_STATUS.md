# 🚁 Cooper_drone 伴随计算机系统配置与开发进度纪要

**项目名称**: Cooper_drone (基于 MYiR 开发板的无人机机载/伴随计算机开发)
**当前阶段**: M7 阶段 — OS 迁移完成 (OpenSTLinux v6.0) + 全子系统验证通过 + Yaw 符号 Bug 修复 + NPU INT8 `.nb` 转换链路排查中
**最后更新**: 2026年7月9日 (ST Cloud/STM32MP257 NPU 模型转换实测记录补充；FP32 `.nb` 可运行但未证明 NPU 生效，INT8/QDQ/QOperator Optimize 仍失败)

---

## 0. 2026-07-09 NPU/ST Cloud 当前结论

完整记录见 [NPU_ST_CLOUD_20260709_FINDINGS.md](NPU_ST_CLOUD_20260709_FINDINGS.md)，旧转换计划修正见 [NPU_MODEL_CONVERSION_PLAN.md](NPU_MODEL_CONVERSION_PLAN.md)，`.nb` 板端诊断见 [NPU_MODEL_NB_DIAGNOSIS.md](NPU_MODEL_NB_DIAGNOSIS.md)。

当前已确认：`road_yolo11n_seg.onnx` 和全改写 FP32 模型均可在 ST Cloud Optimize 并生成 `.nb`，但板端输出为 float16、推理约 600ms，`strace` 未观察到 `/dev/galcore` ioctl，因此不能作为真实 VIP9000 NPU 加速结果。ST Cloud QDQ、本地 QDQ、强制 int8 I/O 和 QOperator 量化模型 Optimize 均失败，典型错误为 `Generation does not contain any output`。下一步重点不是继续扩大校准集，而是解决“量化 ONNX 到实际 NPU INT8 `.nb`”的编译链路。

---

## 1. 项目概述与硬件底盘状态
本项目旨在配置和开发一套适配于无人机的机载计算机系统，负责上层逻辑处理、激光雷达避障算法以及与底层匿名飞控（灵霄）的二进制串口协议通信（非 MAVLink）。

* **核心板**: MYiR MYD-LD25X (STM32MP257, Cortex-A35 双核架构)。
* **操作系统**: ~~Debian 12 (Bookworm)~~ → **OpenSTLinux v6.0** (Yocto Scarthgap) aarch64, Linux 内核 6.6.48, gcnano 6.4.19.4。
* **飞控通信**: 匿名飞控（灵霄）二进制协议，UART 串口 500000 baud。协议栈位于 `FlightController/Base.py` → `Protocal.py` → `Application.py`。
* **存储与内存**:
  * 板载 RAM **2GB**，严禁将日志或大文件写入 `/tmp`（tmpfs 占用 RAM），否则触发 OOM 系统卡死。
  * SD 卡 (30G) 现为系统盘 (rootfs + userfs)，eMMC (7.3G) 已同步烧录可独立启动。
  * 代码仓库位于 `/usr/local/ObstacleAvoidanceDrone` (userfs 分区, 1.3G)，虚拟环境位于 `/usr/local/UFC_venv`。
  * WiFi 已配置开机自启 (`wpa_supplicant@wlan0.service`)，IP 固定 192.168.31.199。

## 2. 混合架构 Python 隔离舱 (UFC_venv) 状态
为了在羸弱的 ARM64 算力下规避现场编译导致的 OOM (内存爆满) 宕机，本项目采用**"APT底层预编译 + PIP上层轻量包"的半透膜沙盒架构**。

* **隔离舱机制**: 使用 `virtualenv --system-site-packages /usr/local/UFC_venv` 建立，允许虚拟环境向下继承操作系统的底层 C++ 依赖。所有终端自动激活 (`.bashrc` + `.bash_profile`)。
* **已就绪的核心生态 (100% 探活通过)**:
  * **[APT 注入层]** `numpy` (1.26.4), `opencv` (4.9.0), `matplotlib` (3.7.2), `onnxruntime` (1.19.2 含 VSINPUExecutionProvider)
  * **[PIP 注入层]** `scipy` (1.17.1), `loguru` (0.7.3), `pyserial` (3.5), `simple-pid` (2.0.1), `attrs` (26.1.0)

## 3. 史诗级避坑指南 (Troubleshooting Archive)
1. **GitHub SNI 阻断与 SSH 降维打击**: 强制将 Git 全局配置由 `https://` 篡改为 `git@github.com:`，穿透 GnuTLS -110 超时阻断。
2. **Debian APT 版本锁死突围**: 当拉取底层库遇 `held broken packages` 时，放弃全系统 `upgrade` 避免爆盘，直接在 `venv` 中用 `pip install --only-binary :all:` 拉取预编译 Wheel 包。
3. **NumPy 2.0 世纪大断层 (ABI Break) 防御**:
   * **症状**: `AttributeError: _ARRAY_API not found`
   * **根源**: Debian 12 的底层 C++ 库基于 NumPy 1.x 编译，而 `pip` 擅自拉取了 NumPy 2.x，导致内存指针寻址崩溃。
   * **防御锁**: 在 `requirements.txt` 顶部强制注入 `numpy<2.0.0`，永久镇压所有下级依赖的越权升级。
4. **PYTHONPATH 导包幽灵**: 运行测试脚本时抛出 `ModuleNotFoundError`，通过在终端指令前置 `PYTHONPATH=.` 强行将项目根目录注入 Python 搜索树解决。
5. **RealSense T265 的生态抛弃**: 由于 ARM64 架构缺乏官方预编译的 `pyrealsense2` 且现场编译必定宕机，已在测试代码中将其彻底跳过 (`--skip-pyrealsense2`)。
6. **ARM 100Hz 内核与 Python sleep 精度陷阱**:
   * **症状**: `time.sleep(0.001)` 在 CONFIG_HZ=100 的 ARM 内核上实际睡眠 ~10ms（内核调度粒度 = 1/HZ = 10ms）。
   * **影响**: 串口读取线程每次等待新数据时 sleep 1ms 实际变为 10ms，期间 D500 以 ~300 包/秒持续输出，导致每个 sleep 间隙积压 ~3 个数据包。累积效应下串口缓冲区逐渐填满，处理延迟随时间线性增长（运行 3 分钟后延迟可超过 60 秒）。
   * **修复**: 将 `time.sleep(0.001)` 改为 `time.sleep(0)` — 仅让出 GIL 而不实际睡眠，线程以 busy-wait 方式轮询串口。批处理模式下数据到达即刻读取，消除积压。
7. **loguru 同步文件写入导致 GIL 争用** ✅ 已确认修复:
   * **症状**: `PROFILE` 日志行显示个别循环迭代耗时 88.9ms（正常 < 5ms），伴随 `Map_Circle` 有效点云数从 ~850 暴跌至 ~390，雷达吞吐降至 ~51%。
   * **根源**: `loguru.add(..., enqueue=False)` 同步写 eMMC 文件。当文件系统触发 journal commit 或写回缓存 flush 时，`write()` 调用阻塞主线程 80-90ms。Python GIL 在此期间被持有，串口读取线程无法执行 `Map_Circle.update()`，雷达帧积压在 `buf` 中无法处理，`timeout_clear` (0.15s) 清除过期点云导致覆盖率下降。
   * **修复**: `enqueue=False` → `enqueue=True`（异步队列写入），配合 `sleep(0.001)` + batch read 串口策略，最终方案经实物验证：GIL 阻塞峰值从 88.9ms 降至不可测（所有 PROFILE total < 12ms），零 CRC 错误，详见 [§6.6](#66-mapupdate-性能优化攻坚)。
8. **RelativeGoalNavigator Yaw 符号 Bug — 避障方向左右颠倒** ✅ 已修复 (2026-07-08):
   * **症状**: 雷达避障表现"左右镜像"——障碍物在左侧时飞机右转撞向障碍物，在右侧时左转撞向障碍物。地面推车测试 (§6.8) 仅验证了 1D 前向 stop/slow/cruise（不涉及 yaw），未暴露此问题。
   * **根源**: `_yaw_command()` 缺少符号反转。Navigator 内部角度以机体坐标系表示（+Y = 左），但飞控 API `send_realtime_control_data(yaw)` 的约定是 `yaw>0 = 顺时针 = 右转`。
     * 场景：障碍物在左前方 → Navigator 选择右侧方向(-30°)避开 → `_yaw_command(-30°) = -15 °/s` → 飞控解读为左转 → 飞机**转向障碍物**而非远离。
     * 正确行为：`_yaw_command(-30°)` 应输出 `+15 °/s`（右转）。
   * **修复** (`RelativeGoalNavigator.py:605`, 1行):
     * `angle_deg * cfg.yaw_kp` → `-angle_deg * cfg.yaw_kp`
     * `RoadFollower` 已有 `yaw_sign` 参数应对同类问题，`RelativeGoalNavigator` 此前漏掉了。
   * **验证**: 阶段 0 测试套件 (`tests/test_radar_coordinates.py` + `tests/test_yaw_sign_consistency.py`) 确认雷达坐标转换正确、yaw 符号反了；阶段 1 合成数据管道测试 (`tests/stage1_synthetic_radar_pipeline.py`) 10/10 通过；实物雷达方向监控脚本 (`tests/stage1_hardware_radar_dir.py`) 待运行。

## 4. 激光雷达 (LDROBOT D500) 硬件拓扑

### 4.1 上雷达 (index=0)
* **物理连线**: TX 接入 J13 Pin 11 (`PB6_UART4_RX`)。PWM 引脚强制拉高至 3.3V (J13 Pin 1) 防止高频电磁干扰触发 MCU 安全锁。
* **系统映射**: 挂载于 `/dev/ttySTM4`，波特率 230400。
* **信道净化**: 强制执行 `stty raw` 指令打通原始透传模式，斩断内核对 `0x0A / 0x0D` 的截断干扰。
* **扫描性能**: 转速 590-600 RPM (~10Hz)，每帧 47 字节（2 头 + 44 数据 + 1 CRC），~300 包/秒，12 点/包，360° 覆盖。

### 4.2 下雷达 (index=1) — 2026-05-17 新增
* **物理连线**: TX 接入 J13 Pin 10 (`PI5_UART9_RX`)。PWM 拉高至 J13 Pin 17 (3.3V)。VCC 取 J13 Pin 4 (5V)，GND 取 J13 Pin 14。
* **系统映射**: 挂载于 `/dev/ttySTM9`，波特率 230400。
* **安装布局**: 上下双雷达。上雷达水平 360° 扫描中高层避障，下雷达下倾覆盖前下方盲区（具体下倾角待实测确定）。
* **设备发现**: 两颗 D500 同 VID/PID `10C4:EA60`，`resolve_radar_port(index)` 按顺序分配。环境变量 `RADAR0_PORT` / `RADAR1_PORT` 可直接指定设备路径。
* **详细接线**: 参见 `HARDWARE_INTERFACE.md`。

### 4.3 USB 摄像头 — 2026-05-17 新增

两个 USB 摄像头通过 USB Host 接入，OpenCV V4L2 读取，均 640×480 分辨率。

| 项目 | 路径识别摄像头 (上/道路) | 障碍物识别摄像头 (下/障碍物) |
|------|--------------|-----------------|
| 设备路径 | `/dev/video9` | `/dev/video7` |
| USB 位置 | `usb-1.1` | `usb-1.2.3` |
| cv2 index | **9** (OpenSTLinux) / 7 (Debian) | **7** (OpenSTLinux) / 9 (Debian) |
| 型号 | `USB 2.0 Camera` | `USB Camera` |

> ⚠️ **摄像头 index 在 OpenSTLinux 上互换了**。Debian 12 上道路摄像头为 cam#7，OpenSTLinux 上为 cam#9。`road_follow_main.py` 默认值已改为 9。

**已知问题**: 路径识别摄像头 (cam#9) 存在偏青色彩问题 (R/G=0.36, B/G=0.79)，V4L2 手动白平衡控制力有限。已实现软件白平衡修正 (`road_perception.py` `_apply_white_balance()`)，默认系数 R×2.78 / G×1.00 / B×1.26，通过 `road_follow_main.py --wb-enable` 启用。

## 5. M3 阶段：雷达点火与离线算法验证 (Milestone Reached)
底层物理链路与上层 SLAM 算法已成功打通，完成两级探活：

✅ **级别一：底层数据泵探活 (Data Pump)**
* 成功接管后台守护线程。雷达物理转速稳定在 590~600 RPM（约 10Hz）。
* `Map_Circle` 内存池高频刷新正常，能稳定输出 2 米内的有效点云坐标矩阵。

✅ **级别二：离线 SLAM 算法验证 (Hough Line Transform)**
* 规避了 Headless 环境下的 `cv2.imshow` 崩溃陷阱，采用离线渲染图像 `debug_save_img=True` 的策略。
* **纸盒封闭房间实验**：利用矩形纸盒模拟局部正交坐标系，霍夫直线检测算法成功在伴随高频噪声的锯齿状极坐标点云中，精准提取出相互垂直的"墙壁"直线特征。
* **位姿解算成功**：成功算出实时的局部姿态 `(X=0.2, Y=0.0, Yaw=4.1°)`。证明算法已具备将混沌的物理点云降维成飞控可用坐标系的能力。

## 6. M4 阶段：单雷达在线避障链路联调 (Current)

### 6.1 串口读取引擎优化

原串口读取线程 `LDRadar_Driver._read_serial_task()` 存在两个严重性能问题，在 ARM 100Hz 内核环境下互相叠加导致数据积压：

| 问题 | 原实现 | 影响 | 修复 |
|---|---|---|---|
| 逐字节读取 | `self._serial.read(1)` 每次只读 1 字节 | ~300+ 系统调用/秒，CPU 开销大 | 改为 `self._serial.read(self._serial.in_waiting)` 一次性读取所有可用数据 |
| sleep 精度 | `time.sleep(0.001)` | ARM 100Hz 内核实际 sleep ~10ms，期间 ~3 帧积压 | 改为 `time.sleep(0)` — 让出 GIL 但不睡眠 |

**修复后效果**: 串口吞吐稳定在 298-332 包/秒（理论的 100-106%），零 CRC 错误，零丢帧。

### 6.2 LocalPlanner 避障规划器增强

`FlightController/Solutions/LocalPlanner.py` 新增以下功能：

| 功能 | 配置参数 | 说明 |
|---|---|---|
| **无目标自由飞行** | `enable_free_flight=True` | 不依赖摄像头/T265，纯雷达避障巡航 |
| **前方走廊可配置** | `forward_corridor_half_width_cm=50` | 室外场景走廊加宽至 ±50cm |
| **近场噪点过滤** | `min_obstacle_distance_cm=10` | 滤除雷达壳体反射等 < 10cm 噪点 |
| **去抖框架** | `debounce_frames=3` | 障碍物需连续 3 帧确认才生效/解除，消除单帧抖动 |

**避障决策逻辑**:
- 前方障碍物距离 < `obstacle_stop_distance_cm` (80cm) → 急停 (vx=0)
- 前方障碍物距离 < `obstacle_slow_distance_cm` (150cm) → 减速至 50%
- 前方无障碍物 → 巡航速度 `free_flight_speed_cm_s` (30cm/s)

### 6.3 Map_Circle 点云缓存调整

| 参数 | 原值 | 新值 | 说明 |
|---|---|---|---|
| `timeout_time` | 0.4s | 0.15s | 雷达周期 100ms，150ms 超时足够覆盖一次全周扫描，同时让障碍物消失后更快反映 |

### 6.4 飞控通信适配

确认飞控协议为匿名飞控（灵霄）二进制协议，非 MAVLink。协议栈架构:

```
FlightController/
├── Base.py          ← UART 串口通信层 (500000 baud, 心跳, 断线重连)
├── Protocal.py       ← 飞控协议层 (解锁/起飞/降落/实时速度指令)
├── Application.py    ← 应用层 (安全起飞, 实时控制线程)
├── Serial.py         ← 通用串口读取器 (状态机/缓冲)
├── Components/
│   ├── LDRadar_Driver.py    ← D500 雷达驱动 + 实时位姿解算
│   ├── LDRadar_Resolver.py  ← 雷达数据包解析 + Map_Circle
│   ├── MultiRadar.py        ← 双雷达障碍物融合 (前+后)
│   ├── DeviceResolver.py    ← 串口设备自动发现 (VID/PID)
│   ├── RealSense.py         ← T265 VIO 视觉里程计 (可选)
│   └── CameraSource.py      ← USB 摄像头 V4L2 采集 (可选)
└── Solutions/
    ├── LocalPlanner.py      ← 反应式避障 + 目标追踪 + 自由飞行
    ├── AutonomousNavigator.py ← 感知-规划-控制主循环 (支持无摄像头)
    ├── PathPlanner.py       ← 势场法路径规划 (PFBPP)
    ├── Navigation.py        ← 闭环导航 (PID×4, 轨迹跟踪, 起降)
    ├── Radar_SLAM.py        ← 霍夫直线检测 SLAM + ICP 匹配
    └── Vision_Net.py        ← ONNX 神经网络推理 (可选)
```

### 6.5 终版端到端性能验证

经 `map.update()` 热路径优化后（详见 [§6.6](#66-mapupdate-性能优化攻坚)），在 STM32MP257 真机上持续运行验证。最终配置: `sleep(0.001)` + batch read + `map.update()` Python builtins + `setdefault` + `timeout_clear` 降频。

| 指标 | 实测值 | 状态 |
|---|---|---|
| 设备钟速 | **100.01-100.05%**，持续收敛 | ✅ |
| 串口吞吐 | 416-418 包/秒 (理论的 140%) | ✅ |
| 雷达帧年龄 | **0-3ms** | ✅ |
| CRC 错误 | **0 次** | ✅ |
| 解析buf | **0B** 常驻 (仅一次瞬闪 7B) | ✅ |
| 串口buf峰值 | 141-1222B (远未满 4095B) | ✅ |
| 点云稳定性 | 804-827 点，无波动 | ✅ |
| 数据年龄 (最旧) | 93-125ms (正常雷达 10Hz 旋转周期) | ✅ |
| get_points_body_cm 耗时 | 1.0-1.2ms | ✅ |
| planner 耗时 | 0.1-0.2ms | ✅ |
| 单次循环总耗时 | 4.4-12.1ms | ✅ |
| DELAY_TREND | →稳定 (最旧 ~125ms 为 Map_Circle 自然老化) | ✅ |
| 端到端延迟 | **< 5ms** | ✅ |

### 6.6 map.update() 性能优化攻坚

#### 问题发现

串口 I/O 三种方案（`sleep(0)` busy-wait / `sleep(0.001)` ~10ms 盲等 / `read()` timeout）均无法同时达成零积压和零 CRC 错误。进一步诊断发现瓶颈不在 I/O，而在 **`Map_Circle.update()` 每帧处理时间过长**。

#### 微基准测试揭示真相

`bench_map_update.py` 在真机上模拟 300 帧测量各步骤耗时:

| Step | p50(ms) | @300fps | 占比 |
|---|---|---|---|
| dict_apply | **3.381** | 1127ms | **87%** ← 真凶 |
| dict_build | 0.523 | 169ms | 13% |
| timeout_clear | 0.048 | 18ms | 1% |
| avail_points | 0.033 | 11ms | 1% |
| **Total** | **3.905** | **1295ms** | **单核 130% 过载** |

`timeout_clear`（1080 元素布尔掩码，之前猜测的瓶颈）仅占 1%。**87% 的 CPU 消耗在 `np.min([1500])` 对 1-3 元素小列表的 numpy C 层调度开销**。

#### 6 变体增量优化

| 变体 | dict_build p50 | dict_apply p50 | total p50 | CPU |
|---|---|---|---|---|
| BASELINE | 0.523ms | 3.381ms | 3.905ms | 130% |
| APPLY_BUILTINS (`np.min`→`min`) | 0.518ms | **0.216ms** | 0.734ms | 40% |
| APPLY_ONE_TS (`perf_counter` 提升) | 0.524ms | 3.307ms | 3.832ms | 118% |
| APPLY_BOTH (builtins + 单次 TS) | 0.520ms | 0.174ms | 0.695ms | 22% |
| BUILD_SETDEFAULT (`try/except`→`setdefault`) | **0.147ms** | 3.383ms | 3.532ms | 118% |
| **FULL_OPT** (全部优化) | **0.139ms** | **0.174ms** | **0.314ms** | **12%** |

#### 核心修改

三行代码改动（`LDRadar_Resolver.py:305-322`）:

1. **`np.min(values)` → `min(values)`** (dict_apply): 消除 numpy C 层调度，**16 倍加速** (3.381→0.216ms)
2. **`try/except KeyError` → `dict.setdefault()`** (dict_build): 消灭异常对象创建，消除 max 尖峰 (14.5→1.6ms)
3. **`time.perf_counter()` 提升到循环外**: 每帧 1 次调用替代 15-30 次

#### 最终效果

```
每帧 p50:  3.905ms → 0.314ms  (↓ 92%)
CPU 占比:   130%  → 12.3%     (headroom 88%)
@300fps:   1295ms/s → 123ms/s
```

### 6.7 诊断工具

**在线避障链路测试** `FlightController/tools/test_radar_avoidance.py`:

```bash
# 基础用法
PYTHONPATH=. python -u FlightController/tools/test_radar_avoidance.py --no-fc --dry-run

# 性能分析 + 异步日志文件 (⚠️ 日志务必写入闪存卡，/tmp 为 tmpfs 会吃 RAM)
PYTHONPATH=. python -u FlightController/tools/test_radar_avoidance.py \
    --no-fc --dry-run --profile \
    --raw-latency --raw-latency-stdout \
    --log-file /media/sdcard/radar_verify.log

# 实时监控 (另一 SSH 窗口)
tail -f /media/sdcard/radar_verify.log | grep PROFILE
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--raw-latency` | False | 输出雷达帧时间戳延迟、设备钟速、串口buf/解析buf |
| `--raw-latency-stdout` | False | 将 RAW_LATENCY 行用 print(flush=True) 直接输出，绕过 loguru/vfat |

**裸串口诊断** `FlightController/tools/diagnose_radar_latency.py`:

```bash
# 绕过所有业务逻辑，仅测串口原始延迟
PYTHONPATH=. python -u FlightController/tools/diagnose_radar_latency.py \
    --port /dev/ttySTM4 --duration 180 --report-interval 1
```

**map.update() 微基准测试** `FlightController/tools/bench_map_update.py`:

```bash
PYTHONPATH=. python -u FlightController/tools/bench_map_update.py
```

### 6.8 实物延迟验证

**测试方法**:

```bash
# 终端 1: 启动测试
PYTHONPATH=. python -u FlightController/tools/test_radar_avoidance.py \
    --no-fc --dry-run --profile \
    --raw-latency --raw-latency-stdout --debug-dump \
    --log-file /media/sdcard/latency_verify.log

# 终端 2: 监控延迟事件
tail -f /media/sdcard/latency_verify.log | grep LATENCY
```

手持纸板在雷达正前方做急停/减速/侧移测试，观察 `[LATENCY]` 行记录的障碍物距离变化。实测响应在可接受范围内，帧号差 × 100ms ≤ 端到端延迟上限，`[RAW_LATENCY]` 行的 `雷达帧年龄: 当前=1ms` 确认雷达数据从采集到处理完成仅需 ~1ms。

## 7. 代码仓库迁移与存储架构重构

### 7.1 仓库迁移

原开发板仅有 `FlightController/` 目录连接旧 GitHub 仓库，现将整个 `ObstacleAvoidanceDrone/` 迁移至独立仓库 `github.com:Cooper3516833584/stm.git`，实现完整项目管理。

| 项目 | 迁移前 | 迁移后 |
|---|---|---|
| 仓库地址 | 旧仓库（仅 FlightController） | `git@github.com:Cooper3516833584/stm.git`（全项目） |
| 代码物理位置 | eMMC `/home/stm/Desktop/ObstacleAvoidanceDrone` | 闪存卡 `/media/sdcard/ObstacleAvoidanceDrone` |
| 访问路径 | 真实目录 | 软链接 `~/Desktop/ObstacleAvoidanceDrone → /media/sdcard/ObstacleAvoidanceDrone` |
| 虚拟环境集成 | 旧路径 `pip install -e .` | 重新安装，`Editable project location: /media/sdcard/...` |

**迁移步骤摘要**:
1. VS Code Server 离线手动安装（绕过微软下载服务器不可达问题）
2. 清理 eMMC 残留旧目录，删除混乱嵌套结构
3. 闪存卡重新挂载为 vfat，指定 `uid=stm,gid=stm` 解决权限问题
4. SSH clone 新仓库至闪存卡，软链接保持原路径兼容

### 7.2 ⚠️ eMMC 存储空间危机 (CRITICAL)

**这是当前系统最严峻的硬件约束，必须在后续所有操作中持续关注。**

| 存储介质 | 总容量 | 已用 | 可用 | 占用率 | 用途 |
|---|---|---|---|---|---|
| **eMMC** (`mmcblk1`) | **6.9G** | **5.6G** | **930M** | **87%** | 系统 + 虚拟环境 |
| 闪存卡 (`mmcblk0`) | 30G | 2.9M | 30G | 1% | 代码仓库 + 日志 |

**eMMC 空间占用分解 (清理后)**:

| 目录 | 大小 | 可否清理 |
|---|---|---|
| `/usr/lib` | 2.2G | 否（系统库） |
| `/home/stm/.vscode-server` | **1.6G** | 否（Remote SSH 必需，单版本已最小化） |
| `/usr/share` | 1.1G | 部分（已清 doc/locale，剩余 fonts/icons 可压） |
| `/home/stm/UFC_venv` | 365M | 否（Python 虚拟环境核心） |
| `/var` | 303M | 受限（已清日志） |
| `/home/stm/.cache` | ~144M | 已清理 |

**已执行的清理措施**:
- APT 缓存全清 (`apt clean`)
- `/usr/share/doc/*` 删除（134M）
- `/usr/share/locale` 去除非英文 locale（~200M）
- `/home/stm/.cache/*` 清空（144M）
- `.dotnet` 目录删除（252K）
- `~/npu_runtime` 15MB (OpenSTLinux .so, 待OS迁移后清理)

**⚠️ 硬性约束与风险**:
- eMMC 仅剩 **930M**，禁止执行 `apt upgrade`（会拉取大量 deb 包导致爆盘）
- 禁止安装任何新的大型 APT 包
- 新 Python 包通过 `pip install` 安装到 `UFC_venv`（eMMC），需控制总大小
- 日志文件、调试输出务必写入 `/tmp`（tmpfs，内存）或闪存卡路径
- VS Code Server 更新时需手动删除旧版本再安装新版本，防止同时存在两个版本（×2 = 3.2G 直接爆盘）
- `/usr/share/fonts`（126M）和 `/usr/share/icons`（43M）可作为紧急情况下的最后清理目标

**闪存卡注意事项**:
- 格式为 **vfat**，不支持 Unix 权限、符号链接等 ext4 特性
- 自动挂载配置已写入 `/etc/fstab`: `vfat rw,uid=stm,gid=stm,noatime`
- 闪存卡仅存代码和 git 数据，不存放运行时依赖或虚拟环境

## 8. 后续工作 (M5 阶段规划)

当前状态: 单雷达在线避障链路已 100% 验证（设备钟速 ≥100%、零积压、零 CRC 错误、端到端延迟 < 5ms、CPU 占用 12%）。以下为雷达相关后续工作，按优先级排序:

### 8.1 飞控联调 (阶段 B/C) — 优先级 ⭐⭐⭐ ✅ 已完成 (2026-05-17)
- [x] Phase A: `debug/test_fc_connect.py` — FC 基础连通性，mode/bat/姿态/速度/位置 16 字段回传正常
- [x] Phase B: `debug/test_fc_command.py` — `set_flight_mode(2)` ACK 确认，mode 切换到 HOLD_POS 成功
- [x] Phase C: `debug/test_fc_realtime.py` — `send_realtime_control_data()` 10/10 成功，零异常零断连
- [x] Phase D: `test_radar_avoidance.py --dry-run` — FC+雷达并行运行，设备钟速 100%，FC 心跳对雷达链路零干扰
- [x] Phase E: `test_radar_avoidance.py` (去除 --dry-run) — 真实飞控指令下发，stop/slow/cruise 三段避障完整验证，Ctrl+C 安全退出零速发送
- [x] 地面推车测试：手持障碍物靠近雷达，通过 FC 状态回传验证速度指令变化（stop→slow→cruise 三段切换）

### 8.2 单雷达盲区评估 — 优先级 ⭐⭐⭐ → ✅ 已决策跳过

**决策**: 不做完整盲区测绘实验，直接采用上下双雷达布局。理由：

- 飞行速度慢 + 小倾角 → 俯仰诱导盲区有限，下雷达低装进一步压缩盲区
- 已明确只做前向路径规划，后方不在需求范围内
- 上下双雷达已定，决策前提（"是否需要第二颗"）已不存在
- 唯一需快速验证的点：**下雷达扫描面内 4 个支撑脚的遮挡**——确认遮挡扇区不覆盖前向走廊即可

### 8.3 双雷达融合 — 优先级 ⭐⭐⭐ → ✅ 已完成 (2026-05-17)

- [x] 第二颗 D500 已接入 UART9 (`/dev/ttySTM9`)，接线记录至 `HARDWARE_INTERFACE.md`
- [x] 下雷达倒装 Y 轴镜像：`LD_Radar` 新增 `mount_mirror_y` 参数，`get_points_body_cm()` 先翻转 Y 再旋转/平移
- [x] 双雷达点云融合坐标系验证：上 1080 + 下 1080 点全覆盖，零 CRC 错误
- [x] 双雷达安装位姿测量完成：上雷达原点 (0,0)，下雷达 (0.96, 0.15) cm
- [x] 机身自反射屏蔽：机体范围 ±25cm 内点云自动过滤
- [x] `test_dual_radar.py` 双雷达统一测试脚本，默认 30Hz 固定频率，CPU ~15%

**双雷达链路 KPI**:

| 指标 | 实测值 |
|------|--------|
| 上雷达点云 | 1080/1080 全覆盖 |
| 下雷达点云 | 1080/1080 全覆盖 |
| CRC 错误 | 0 次 |
| 机身屏蔽 | 80-85 点（稳定基线） |
| 主循环频率 | 30Hz（可配 `--loop-hz`） |
| 主循环 CPU | 14-25% |
| 帧率稳定性 | 27-29Hz @30Hz 目标 |

### 8.4 SLAM 在线化 — 优先级 ⭐⭐
- [ ] 将离线 Hough SLAM (`Radar_SLAM.radar_resolve_rt_pose`) 接入 `LDRadar_Driver._map_resolve_task()`（当前 `subtask_event` 每 4 帧已触发，但 `rt_pose` 未启用）
- [ ] 验证实时位姿估计精度（与 T265 VIO 对比，如 T265 不可用则使用 ICP 匹配雷达帧间位姿）
- [ ] 配置 `Navigation` 以 `mode="radar"` 运行，实现无 GPS/VIO 条件下的闭环悬停

### 8.5 避障策略升级 — 优先级 ⭐
- [ ] 当前为纯前向走廊检测 (1D stop/slow/cruise)，升级为基于势场法 (PFBPP) 的方向性避障
- [ ] 集成 `PathPlanner.py` 到在线 loop 中，实现 360° 障碍物排斥 + 目标吸引的路径规划
- [ ] 评估 PFBPP vs VFH 在 D500 点云密度下的适用性（D500 单帧 12 点，远低于传统 2D LiDAR）

---

## 9. M5 阶段：飞控联调代码范式 (2026-05-17 完成)

### 9.1 飞控连接最小范式

Phases A-C 独立测试脚本位于 `debug/` 目录，Phase D-E 复用 `test_radar_avoidance.py`。以下为已验证的 FC 连接代码范式：

```python
from FlightController import FC_Controller

fc = FC_Controller()

# 自动探测 VID/PID 66CC:2233 端口, block 模式持续重试直到连接
fc.start_listen_serial(block_until_connected=True)
fc.wait_for_connection(timeout_s=5)

# 读取状态 (FC_State_Struct, 16 字段)
s = fc.state
print(s.mode.value)     # 1=定高 2=定点 3=程控
print(s.unlock.value)   # bool, 电机解锁状态
print(s.bat.value)      # float, 电池电压 (USB供电=0.0V)
print(s.rol.value)      # float, roll 角度
# ... 其余 12 字段见 FC_State_Struct.RECV_ORDER

# 模式切换 (实时控制需 HOLD_POS)
fc.set_flight_mode(2)           # 切定点模式
fc.wait_for_last_command_done() # 等待 ACK (timeout 10s)

# 实时控制 (需先切 mode=2)
fc.send_realtime_control_data(vel_x=30, vel_y=0, vel_z=0, yaw=0)

# 安全退出: 发送零速 → 关闭
fc.send_realtime_control_data(0, 0, 0, 0)  # 先归零
fc.close()                                  # 再断开
```

### 9.2 避障链路集成范式 (test_radar_avoidance.py 核心循环)

```python
# --- 初始化 ---
fc = FC_Controller()
fc.start_listen_serial(block_until_connected=True)
fc.wait_for_connection()
fc.set_flight_mode(2)
fc.wait_for_last_command_done()

radar = LDRadar_Driver(port="/dev/ttySTM4")
planner = LocalPlanner(enable_free_flight=True, forward_corridor_half_width_cm=50)

# --- 主循环 (100Hz) ---
while running:
    points_body = radar.map.get_points_body_cm(max_distance=200)
    obstacle_cm = planner.update(points_body)
    vx, vy, vz, yaw = planner.get_velocity_command()
    
    if not dry_run:
        fc.send_realtime_control_data(vel_x=vx, vel_y=vy, vel_z=vz, yaw=yaw)
    
    # 日志 (每10帧)
    logger.info(f"FC[mode={fc.state.mode.value} ...] | "
                f"点云={len(points_body)}点 | 前方={obstacle_cm}cm | "
                f"指令=(vx={vx}, vy={vy}, vz={vz}, yaw={yaw})")

# --- 安全退出 (finally 块) ---
fc.send_realtime_control_data(0, 0, 0, 0)
radar.stop()
fc.close()
```

### 9.3 FC 与雷达并行性能验证

| 指标 | Phase E 实测 | M4 纯雷达基准 | 结论 |
|---|---|---|---|
| 设备钟速 | 100.00-100.01% | 100.01% | 零干扰 |
| 串口吞吐 | 415-417 包/秒 (139-140%) | 416 包/秒 (140%) | 零衰减 |
| CRC 错误 | 0 次 | 0 次 | — |
| 解析buf | 0B | 0B | — |
| 单次循环 | 5.3-11.7ms | 4.4-12.1ms | 无差异 |
| 点云数 | 835-955 | 804-827 | FC 线程无干扰 |

FC 心跳线程 (每 250ms 发送 AA 22 帧) 对雷达串口读取线程无 GIL 争用影响，设备钟速保持 100% 收敛。

### 9.4 调试脚本目录

| 脚本 | 用途 | 关键参数 |
|---|---|---|
| `debug/test_fc_connect.py` | Phase A: FC 连通性 + 全状态打印 | `--port` 指定串口 |
| `debug/test_fc_command.py` | Phase B: 模式切换 + ACK 验证 | `--target-mode` 1/2/3 |
| `debug/test_fc_realtime.py` | Phase C: 实时控制协议层测试 | `--count` `--speed` `--interval` |
| `FlightController/tools/test_radar_avoidance.py` | Phase D/E: 避障链路完整测试 | `--dry-run` `--profile` `--raw-latency` |
| `FlightController/tools/test_dual_radar.py` | 双雷达融合避障链路测试 | `--loop-hz` `--debug-dump` `--dry-run` |
| `FlightController/tools/smoke_dual_radar.py` | 双雷达连通性烟雾测试 | `--upper-port` `--lower-port` |

运行方式: `cd ~/Desktop/ObstacleAvoidanceDrone && PYTHONPATH=. python debug/<script>.py`

### 9.5 USB 供电注意事项

FC 通过 USB 供电时 (无电池)，`state.bat.value` 回传 0.0V。此非故障——电池检测引脚悬空导致 ADC 读数为 0。Phase A 测试已适配此场景（bat=0 → WARN 非 FAIL）。实际飞行前需接入电池并验证 bat > 10V。

### 9.6 双雷达融合代码范式 (2026-05-17 完成)

**硬件接线**:

| | 上雷达 (index=0) | 下雷达 (index=1) |
|---|---|---|
| 串口 | UART4 `/dev/ttySTM4` | UART9 `/dev/ttySTM9` |
| 波特率 | 230400 | 230400 |
| 安装方式 | 正装 | 倒装 (左右反转) |

详见 `HARDWARE_INTERFACE.md`。

**软件架构**:

```
MultiRadar(configs)
  ├── LD_Radar("upper", index=0, mount_mirror_y=False)
  │     └── UART4 串口读取线程 ─→ Map_Circle (1080 bins)
  └── LD_Radar("lower", index=1, mount_mirror_y=True)
        └── UART9 串口读取线程 ─→ Map_Circle (1080 bins)

主循环 (30Hz):
  multi_radar.get_obstacle_points_body_cm()  ─→ vstack 融合点云
  _body_mask(points, ±25cm)                  ─→ 滤除机身自反射
  planner.plan(obstacles, target=None)       ─→ 避障决策
  fc.send_realtime_control_data()            ─→ 飞控指令
```

**最小初始化范式**:

```python
from FlightController.Components import MultiRadar, RadarConfig

configs = [
    RadarConfig(
        name="upper", index=0,
        mount_xy_cm=(0.0, 0.0), mount_yaw_deg=0.0,
        port="/dev/ttySTM4",
    ),
    RadarConfig(
        name="lower", index=1,
        mount_xy_cm=(0.96, 0.15), mount_yaw_deg=0.0,
        mount_mirror_y=True,           # ← 倒装 Y 轴镜像
        port="/dev/ttySTM9",
    ),
]
multi_radar = MultiRadar(configs)
multi_radar.start()

# 等待连接
while not multi_radar.connected:
    time.sleep(0.1)

# 主循环
while True:
    points = multi_radar.get_obstacle_points_body_cm(max_distance_cm=300)
    # 机身屏蔽: |x| < 25cm AND |y| < 25cm
    filtered = points[~((abs(points[:, 0]) < 25) & (abs(points[:, 1]) < 25))]
    # 避障决策 → 飞控
    ...

multi_radar.stop()
```

**坐标变换链路** (下雷达为例):

```
① get_points_xy_cm()  →  雷达本地帧 (x, y)
② Y 轴镜像           →  points[:, 1] *= -1.0    ← mount_mirror_y=True
③ yaw 旋转           →  points @ rotation.T     ← mount_yaw_deg=0 (恒等)
④ XY 平移            →  + mount_xy_cm           ← (0.96, 0.15) cm
⑤ np.vstack          →  与上雷达点云融合
```

**关键参数**:

| 参数 | 值 | 说明 |
|------|-----|------|
| 目标频率 | 30Hz | `--loop-hz 30`，稳定 27-29Hz |
| CPU 占用 | 14-25% | 留足 ~60% 给视觉处理 |
| 机身屏蔽 | ±25cm | `--body-x-half-cm` / `--body-y-half-cm` |
| 前方走廊 | ±50cm | `--corridor-half-width-cm` |
| 最大检测距离 | 300cm | `--max-distance-cm` |

**运行方式**:

```bash
# 仅雷达测试
PYTHONPATH=. python -u FlightController/tools/test_dual_radar.py --no-fc

# 带 debug dump
PYTHONPATH=. python -u FlightController/tools/test_dual_radar.py --no-fc --debug-dump

# 完整链路 (雷达 + 飞控)
PYTHONPATH=. python -u FlightController/tools/test_dual_radar.py

# 调频
PYTHONPATH=. python -u FlightController/tools/test_dual_radar.py --no-fc --loop-hz 50
```

**主循环频率选择逻辑** (ARM CONFIG_HZ=100):

| 方式 | 效果 | 问题 |
|------|------|------|
| `sleep(0)` 自由竞争 | 帧率不稳定，CPU 饱和 | 三线程抢 GIL，主线程仅抢到 50-75% |
| 固定频率 `sleep(remaining)` | 帧率精确，GIL 不争抢 | 需选择合适频率 |

选定 30Hz (周期 33ms)：每帧工作 ~5ms 后 sleep ~28ms，主线程主动让出 GIL，双串口线程不受干扰。

---

## 附录：关键性能调优决策记录

| 决策 | 理由 |
|---|---|
| Python 继续使用（不换 C++） | numpy / opencv / pyserial 底层均为 C 实现，Python 仅作胶水代码。`map.update()` 优化后 pipeline 耗时 < 1ms，语言切换无收益 |
| 跳过盲区测绘，直接上双雷达 | 飞行速度慢+小倾角使俯仰诱导盲区有限；已明确只做前向路径规划；上下双雷达已定（而非"是否需要"的决策） |
| 下雷达倒装 + Y 轴镜像 | 下雷达倒装导致左右反转，`mount_mirror_y=True` 在坐标变换中先翻转 Y 再平移，5 行代码解决，无需改硬件安装方向 |
| 主循环固定 30Hz | `sleep(0)` 导致主线程+双串口线程三线程自由争抢 GIL，帧率波动 25-74Hz 不可控。固定频率 `sleep(remaining)` 让主线程主动让出 GIL，帧率稳定在 27-29Hz，CPU 仅 15%，留 60% 算力给视觉 |
| 机身范围屏蔽 ±25cm | 无人机自反射 + 4 支撑脚在近距离产生固定噪点，`_body_mask()` 在融合后统一过滤，避免被误判为障碍物触发急停 |
| 反应式避障优先 SLAM 建图 | 紧急避障需要 < 200ms 响应，SLAM 位姿估计用于导航漂移校正，两者分层异步运行 |
| 串口读取: `sleep(0.001)` + batch read | `sleep(0)` busy-wait 导致 CRC 错误递增（与 loguru writer 抢 CPU）；`read()` timeout 导致发散（每次 10ms 盲等）。`sleep(0.001)` 虽受 100Hz 内核限制实际 ~10ms，但配合 `map.update()` 优化（CPU 仅占 12%）后无积压风险 |
| `np.min([1500])` → `min([1500])` | **全项目最大单行优化**。numpy 对 1-3 元素小列表的 C 层调度开销是 Python 内置 `min()` 的 ~16 倍。真机基准: dict_apply 3.381ms → 0.216ms。CPU 从 130% 降至 12% |
| `try/except KeyError` → `dict.setdefault()` | 消除冷键路径的 KeyError 异常对象创建，dict_build 从 0.523ms 降至 0.147ms，max 尖峰从 14.5ms 降至 1.6ms |
| `timeout_clear` 每 12 帧执行 | 1080 元素全量扫描由 300 次/秒降至 25 次/秒。效果微小（仅占 1%）但无代价 |
| 日志文件异步写入 (`enqueue=True`) | 消除 eMMC I/O 阻塞主线程导致的 GIL 争用，避免周期性点云覆盖率骤降。日志写入闪存卡 vfat，`tail -f` 约 1 分钟延迟为 cosmetic 问题 |


---

## 10. M6 阶段：NPU 适配 & 视觉管线就绪 (2026-06-07)

### 10.1 视觉推理性能基线

**`bench_vision_fps.py`** 在 Debian 12 + STM32MP257 上的实测结果：

| 指标 | 实测值 | 说明 |
|------|--------|------|
| 单帧感知耗时 | **~1800ms** | 瓶颈: `onnxruntime` CPU 推理 YOLO11-seg |
| 实际 FPS | **0.6** | 纯 CPU (Cortex-A35 x2), 无硬件加速 |
| ONNX 推理 | `session.run()` 1800ms | 占单帧 99% 耗时 |
| 捕获 | 3-8ms | V4L2 摄像头抓帧 |
| 后处理 | 1-3ms | mask 解码/中线提取/偏移补偿 |

**视觉管道当前无法实时使用。** 0.6 FPS 意味着每秒仅判断一次道路方向。

### 10.2 NPU 硬件栈验证

| 层 | 状态 | 说明 |
|------|:--:|------|
| `/dev/galcore` | Yes | NPU 内核驱动已加载 (galcore 6.4.15.6) |
| `libGAL.so`, `libVSC.so` | Yes | 从 vendorfs 提取，已安装到 /usr/lib/aarch64-linux-gnu/ |
| `libOpenCL_VSI.so` | Yes | NPU OpenCL 后端就绪 |
| `libovxlib.so`, `libOpenVX.so`, `libtim-vx.so` | Yes | 从 OpenSTLinux rootfs 提取 |
| `libArchModelSw.so` | No | 不存在于 Debian 镜像，是 gcnano 驱动栈组件 |
| `libopenvx-gcnano` | No | 仅 OpenSTLinux 提供 |

### 10.3 NPU ONNX Runtime 确认存在

从 ST APT 仓库下载的 `onnxruntime_1.19.2_arm64.deb`：
```bash
strings /usr/lib/libonnxruntime.so.1.19.2 | grep -i vsinpu
# -> VSINPUExecutionProvider
# -> OrtSessionOptionsAppendExecutionProvider_VSINPU
```
**VsiNpuExecutionProvider 已编译在 ST 的 ONNX Runtime 版本中。** 但无法在 Debian 12 上加载——缺少 glibc 2.38 和 libArchModelSw.so。

### 10.4 尝试过的不兼容路径

| 方案 | 结果 | 失败原因 |
|------|:--:|------|
| `pip install onnxruntime` (标准版) | No | 只含 CPUExecutionProvider |
| 从 SDK 提取 onnxruntime (CPU variant) | No | AISDK-Y-MP2 不含 NPU variant |
| `dpkg -i` ST 的 NPU deb 包 | No | 依赖 libc6>=2.39, libopenvx-gcnano |
| `LD_LIBRARY_PATH` 加载 OpenSTLinux glibc | No | libc.so.6: undefined symbol |
| `LD_PRELOAD` 加载 libovxlib.so | No | libArchModelSw.so 缺失 |

### 10.5 代码侧更新汇总 (2026-06-07)

| 文件 | 改动 |
|------|------|
| `road_perception.py` | `_select_providers()` NPU>XNNPACK>CPU, `compute_meters_per_pixel()` height_m 覆盖, `get_road_perception()` flight_height_m 接入 |
| `road_follow_main.py` | --camera-index 0->7, --flight-height-m, --cam-forward-offset-m 0.15->0.10 |
| `goal_nav_main.py` | --goal-x-cm 加 help 标注, 新增安全参数 |
| `obstacle_classifier.py` | 删除 (零引用死代码) |
| `HARDWARE_INTERFACE.md` | 新增 section 5.4 (摄像头几何标定: alpha=30.27, VFOV=55.08, HFOV=68) |
| `PARAMETER_AUDIT.md` | 新增 — 全代码默认参数审计报告 |
| `NPU_REQUIREMENTS.md` | 新增 — NPU 软件需求分析 |
| `OS_MIGRATION_PLAN.md` | 新增 — Debian->OpenSTLinux 迁移方案 |
| tools: `diagnose_npu.py` | NPU 诊断工具 |
| tools: `bench_vision_fps.py` | 视觉 FPS 基准测试 |
| tools: `visualize_correction_boundary.py` | 偏移补偿边界可视化 |

### 10.6 决策: 切换至 OpenSTLinux v6.2

**原因**: ST NPU 软件栈深度绑定 OpenSTLinux BSP。Debian 12 (glibc 2.36) 与 ST 目标 (glibc 2.39) 不兼容。

**迁移方案**: 详见 `OS_MIGRATION_PLAN.md` — SD 卡烧录, Python 3.12 venv 重建, NPU->雷达->FC->摄像头 全链路验证, ~4h, Debian eMMC 保留为回退。

---

## 11. M7 阶段：OS 迁移 & 全子系统验证 (2026-06-11)

### 11.1 迁移执行摘要

**实际安装**: OpenSTLinux **v6.0** (Yocto Scarthgap)，v6.2 镜像未找到。使用 `myir-image-full` 预构建 `.raw` 镜像 (5.7GB) 通过 balenaEtcher 写入 SD 卡，首次启动自动同步至 eMMC。

| 对比维度 | Debian 12 (旧) | OpenSTLinux v6.0 (新) |
|------|------|------|
| Kernel | 6.1.x (CONFIG_HZ=100) | 6.6.48 |
| glibc | 2.36 | 2.39 |
| gcnano driver | 6.4.15.6 (手工安装) | **6.4.19.4** (预装) |
| Python | 3.11 | 3.12 |
| onnxruntime | 1.25.1 (CPU) | 1.19.2 (NPU) |
| 串口设备映射 | `/dev/ttySTM4`, `/dev/ttySTM9` | 完全一致 ✅ |
| 飞控 | `/dev/ttyACM0` | `/dev/ttyACM0` ✅ |
| 摄像头 | cam#7(道路), cam#9(障碍) | **cam#9(道路), cam#7(障碍)** ⚠️ 互换 |

### 11.2 NPU 状态

| 组件 | 状态 | 详情 |
|------|:--:|------|
| `/dev/galcore` | ✅ | VIP9000, model=0x8000, 800MHz |
| gcnano-userland (6.4.19) | ✅ | APT 预装 |
| libopenvx-gcnano + libovxkernels | ✅ | APT 预装 |
| onnxruntime 1.19.2 + VSINPU EP | ✅ | APT 安装 (AINPU 6.0 仓库) |
| `_select_providers()` 大小写修复 | ✅ | `VsiNpu` → `VSINPU` |
| **YOLO11n-seg NPU 推理** | 🔴 | VSINPU EP 注册成功 (348/353 nodes)，但 `ConvTranspose` 回退 CPU 时 segfault |
| **解决方案** | 🔄 | ST Edge AI Cloud 模型转换 — 详见 `NPU_MODEL_CONVERSION_PLAN.md` |

### 11.3 全子系统验证结果

| 子系统 | 工具 | 结果 |
|------|------|:--:|
| 双雷达 (上+下) | `smoke_dual_radar.py` | ✅ 2150点, 零CRC |
| 飞控 Phase A (连通性) | `test_fc_connect.py` | ✅ 16字段正常, bat=0V(USB供电) |
| 飞控 Phase B (模式切换) | `test_fc_command.py --target-mode 2` | ✅ mode=1→2 ACK |
| 摄像头探活 | OpenCV V4L2 | ✅ cam7/cam9 640×480 OK |
| 白平衡修正 | `road_follow_main.py --wb-enable` | ✅ 已编码, 待道路场景验证 |

### 11.4 代码更新 (2026-06-11)

| 文件 | 改动 |
|------|------|
| `road_perception.py` | `_select_providers()` 大小写修正 `VsiNpu`→`VSINPU`; 新增 `CameraWhiteBalanceConfig` + `_apply_white_balance()` |
| `road_follow_main.py` | `--camera-index` 默认值 7→9; 新增 `--wb-enable/--wb-r/--wb-g/--wb-b` 参数; 注入 `wb_config` |
| `OS_MIGRATION_PLAN.md` | 补充 SD 卡烧录详细步骤 (§4)、硬件清单、balenaEtcher 教程、常见问题排查 |
| `NPU_MODEL_CONVERSION_PLAN.md` | 新增 — ST Edge AI Cloud 转换方案 + 验收 Checklist |

### 11.5 待办

| 项目 | 优先级 | 说明 |
|------|:---:|------|
| ST Edge AI Cloud 模型转换 | 🔴 | 将 YOLO11n-seg 转为 NPU 兼容 ONNX |
| 道路场景实测 | 🟡 | 白平衡效果 + 视觉推理正确性 |
| 校准图片采集 | 🟡 | 用于 INT8 量化 (需道路场景) |
| Weston 桌面管理 | 🟢 | 当前未关闭, ~100MB RAM; 可 `systemctl disable weston` |
| eMMC 空间利用 | 🟢 | 7.3G eMMC 空闲, 可作备份/数据盘 |

---

## 12. M8 阶段：坐标系全链路验证 & Yaw 符号 Bug 修复 (2026-07-08)

### 12.1 问题描述

用户报告"雷达平面内数据左右镜像"——无人机的避障转向方向与预期相反。

### 12.2 坐标系全链路审计

经完整追踪从 D500 雷达原始数据到飞控指令的坐标转换链：

| 环节 | 约定 | 验证结果 |
|------|------|:--:|
| D500 雷达物理约定 | 0°=前, 顺时针递增（俯视） | ✅ |
| `get_points_xy_cm()` 极坐标→笛卡尔 | `x=d·cos(θ), y=-d·sin(θ)` | ✅ |
| 机体坐标系 | +X=前, +Y=左, -Y=右 | ✅ |
| `get_points_body_cm()` 安装位姿变换 | 旋转 + 平移 + Y镜像 (下雷达) | ✅ |
| `select_forward_corridor()` SafetyArbiter 前方走廊 | `x>min_x & |y|<half_w` | ✅ (仅 1D) |
| 可视化渲染 (`visualize_radar_data.py`) | 前=上, 左=左 | ✅ |
| 飞控 API `send_realtime_control_data(yaw)` | yaw>0 = 顺时针 = 右转 | ✅ (在线取反 `-yaw`) |
| **`_yaw_command()` 输出→飞控 yaw** | **缺少符号反转** | 🔴 **BUG** |

**结论：雷达坐标映射完全正确，问题出在 `RelativeGoalNavigator._yaw_command()` 的 yaw 输出符号。**

### 12.3 Bug 详解

Navigator 内部候选方向以机体坐标系表示（+ = 左, - = 右），但飞控 API 的 yaw 约定是 `yaw>0 = 顺时针 = 右转`。二者之间存在一个负号关系，而 `_yaw_command()` 没有处理：

```
障碍物在左前方 (body angle=+45°) 
  → Navigator 选右侧方向避开 (selected=-30°)
  → 当前 _yaw_command(-30°) = -30×0.5 = -15 °/s  → FC: 左转 ❌ 撞向障碍物!
  → 修复 _yaw_command(-30°) = +30×0.5 = +15 °/s  → FC: 右转 ✅ 避开障碍物!
```

| 场景 | 修复前 | 修复后 |
|------|:------:|:------:|
| 左前有障碍，选右侧方向(-30°) | yaw=-15 → 左转 ❌ | yaw=+15 → 右转 ✅ |
| 右前有障碍，选左侧方向(+30°) | yaw=+15 → 右转 ❌ | yaw=-15 → 左转 ✅ |

**为什么之前没发现**：`LocalPlanner` 的 `_plan_free_flight()` 只做前向 stop/slow/cruise，不输出 yaw；地面推车测试（§6.8）只验证了 1D 前向障碍物检测。

### 12.4 修复

**文件**: `FlightController/Solutions/RelativeGoalNavigator.py:605`，**1 行改动**:

```python
# 修复前
yaw = angle_deg * cfg.yaw_kp

# 修复后
yaw = -angle_deg * cfg.yaw_kp
```

注：`RoadFollower` 已有 `yaw_sign` 配置参数（默认 1.0）应对同类问题，`RelativeGoalNavigator` 此前缺失了此防护。

### 12.5 新增测试套件

| 测试脚本 | 类型 | 说明 |
|----------|------|------|
| `tests/test_radar_coordinates.py` | PC 单元测试 | 雷达极坐标→笛卡尔坐标转换验证 (10/10) |
| `tests/test_yaw_sign_consistency.py` | PC 单元测试 | 完整控制链 yaw 符号一致性验证 (6/6) |
| `tests/stage1_synthetic_radar_pipeline.py` | PC 合成数据测试 | Map_Circle → body frame → 可视化全管道 (10/10) |
| `tests/stage1_hardware_radar_dir.py` | 开发板实物测试 | 雷达物理方向监控脚本 (按扇区显示最近距离) |

### 12.6 虚拟环境自动激活修复

开发板默认 shell 为 `/bin/sh`（非 `/bin/bash`），`.bashrc` 不会被加载。在 `~/.profile` 中添加：

```bash
source /usr/local/UFC_venv/bin/activate
```

### 12.7 外设诊断命令

```bash
# 串口设备列表
ls -la /dev/tty* 2>/dev/null | grep -E 'tty(USB|ACM|STM|AMA)'

# USB 设备树
lsusb

# 雷达/飞控 VID:PID 识别
# D500 雷达: 10C4:EA60 (CP210x)
# 凌霄飞控: 66CC:2233

# 摄像头
ls -la /dev/video*

# 内核日志 (串口/USB 信息)
dmesg | grep -iE 'tty|cp210|usb|radar|uart' | tail -30
```

### 12.8 雷达数据录制与可视化

```bash
# 录制 (仅雷达, 不上摄像头)
PYTHONPATH=. python record_data.py --no-camera --output-dir /media/sdcard/recordings

# PC 端渲染
python visualize_radar_data.py <record_dir> --video --video-fps 10
