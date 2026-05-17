# 🚁 Cooper_drone 伴随计算机系统配置与开发进度纪要

**项目名称**: Cooper_drone (基于 MYiR 开发板的无人机机载/伴随计算机开发)
**当前阶段**: M5 阶段 — 飞控联调完成 (Phase A-E 全部通过) + 单雷达在线避障完整链路验证（雷达→避障→飞控指令下发，端到端 < 12ms）
**最后更新**: 2026年5月17日 (飞控联调 Phase A-E 全部通过，FC+雷达完整链路验证完成)

---

## 1. 项目概述与硬件底盘状态
本项目旨在配置和开发一套适配于无人机的机载计算机系统，负责上层逻辑处理、激光雷达避障算法以及与底层匿名飞控（灵霄）的二进制串口协议通信（非 MAVLink）。

* **核心板**: MYiR MYD-LD25X (STM32MP257, Cortex-A35 双核架构)。
* **操作系统**: Debian 12 (Bookworm) aarch64, Linux 内核 (CONFIG_HZ=100)。
* **飞控通信**: 匿名飞控（灵霄）二进制协议，UART 串口 500000 baud。协议栈位于 `FlightController/Base.py` → `Protocal.py` → `Application.py`。
* **存储与内存极限管控 (Survival Mode)**:
  * 板载 RAM 仅 **2GB**，严禁将日志或大文件写入 `/tmp`（tmpfs 占用 RAM），否则触发 OOM 系统卡死。
  * 已彻底斩杀图形桌面系统，进入纯 Headless (无头) 命令行模式以释放极其有限的 RAM。
  * 物理抹除 1GB Swap 分区，清空 APT 缓存，为 eMMC 腾出极限空间保障大型 C++ 库的安装。
  * 代码仓库已迁移至 30G 外置闪存卡（vfat, `/media/sdcard`），eMMC 仅保留系统与虚拟环境，详见 [§7.2](#72-%E2%9A%A0%EF%B8%8F-emmc-存储空间危机-critical)。
  * 强制切断 Wi-Fi 休眠机制 (`wifi.powersave = 2`)，配合物理冷启动，防止 SDIO 总线假死。

## 2. 混合架构 Python 隔离舱 (UFC_venv) 状态
为了在羸弱的 ARM64 算力下规避现场编译导致的 OOM (内存爆满) 宕机，本项目采用**"APT底层预编译 + PIP上层轻量包"的半透膜沙盒架构**。

* **隔离舱机制**: 使用 `virtualenv --system-site-packages ~/UFC_venv` 建立，允许虚拟环境向下继承操作系统的底层 C++ 依赖。
* **已就绪的核心生态 (100% 探活通过)**:
  * **[APT 注入层]** `scipy` (1.17.1), `matplotlib` (3.6.3), `opencv-python-headless` (4.13.0) - *注：必须为 Headless 无头版本以防 cv2.imshow 崩溃*。
  * **[PIP 注入层]** `numpy` (1.26.4), `loguru` (0.7.3), `pyserial` (3.5), `simple-pid` (2.0.1), `onnxruntime` (1.25.1 纯 CPU 推理模式)。

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

## 4. 激光雷达 (LDROBOT D500) 硬件拓扑
* **物理连线**: TX 接入 J13 Pin 11 (`PB6_UART4_RX`)。PWM 引脚强制拉高至 3.3V 防止高频电磁干扰触发 MCU 安全锁。
* **系统映射**: 挂载于 `/dev/ttySTM4`，波特率 230400。
* **信道净化**: 强制执行 `stty raw` 指令打通原始透传模式，斩断内核对 `0x0A / 0x0D` 的截断干扰。
* **扫描性能**: 转速 590-600 RPM (~10Hz)，每帧 47 字节（2 头 + 44 数据 + 1 CRC），~300 包/秒，12 点/包，360° 覆盖。

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

### 8.2 单雷达盲区评估 — 优先级 ⭐⭐⭐
- [ ] 测绘 D500 单线雷达在无人机平台上的实际视野覆盖（水平 360° 全覆盖，俯仰为单平面）
- [ ] 评估前方下视盲区（无人机前倾时雷达视场抬升，地面障碍物可能漏检）
- [ ] 评估后方盲区（当前仅前方走廊检测，后方碰撞无感知）
- [ ] 结论: **是否需要第二颗 D500 雷达**（推荐上下布局: 上雷达水平扫描 360° 避障，下雷达下倾覆盖前下方盲区）

### 8.3 双雷达融合 (取决于 8.2 结论)
- [ ] 接入第二颗 D500 雷达（UART 端口待定），复用 `LDRadar_Driver` 实例
- [ ] 启用 `MultiRadar.py` 双雷达障碍物融合（前+后/上+下）
- [ ] 双雷达时间同步与点云坐标系统一到机体坐标系

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

运行方式: `cd ~/Desktop/ObstacleAvoidanceDrone && PYTHONPATH=. python debug/<script>.py`

### 9.5 USB 供电注意事项

FC 通过 USB 供电时 (无电池)，`state.bat.value` 回传 0.0V。此非故障——电池检测引脚悬空导致 ADC 读数为 0。Phase A 测试已适配此场景（bat=0 → WARN 非 FAIL）。实际飞行前需接入电池并验证 bat > 10V。

---

## 附录：关键性能调优决策记录

| 决策 | 理由 |
|---|---|
| Python 继续使用（不换 C++） | numpy / opencv / pyserial 底层均为 C 实现，Python 仅作胶水代码。`map.update()` 优化后 pipeline 耗时 < 1ms，语言切换无收益 |
| 单雷达优先双雷达 (当前阶段) | 最小闭环原则：先端到端跑通单雷达避障，验证链路正确性后再加双雷达。M4 已竣工，M5 开启双雷达评估 |
| 反应式避障优先 SLAM 建图 | 紧急避障需要 < 200ms 响应，SLAM 位姿估计用于导航漂移校正，两者分层异步运行 |
| 串口读取: `sleep(0.001)` + batch read | `sleep(0)` busy-wait 导致 CRC 错误递增（与 loguru writer 抢 CPU）；`read()` timeout 导致发散（每次 10ms 盲等）。`sleep(0.001)` 虽受 100Hz 内核限制实际 ~10ms，但配合 `map.update()` 优化（CPU 仅占 12%）后无积压风险 |
| `np.min([1500])` → `min([1500])` | **全项目最大单行优化**。numpy 对 1-3 元素小列表的 C 层调度开销是 Python 内置 `min()` 的 ~16 倍。真机基准: dict_apply 3.381ms → 0.216ms。CPU 从 130% 降至 12% |
| `try/except KeyError` → `dict.setdefault()` | 消除冷键路径的 KeyError 异常对象创建，dict_build 从 0.523ms 降至 0.147ms，max 尖峰从 14.5ms 降至 1.6ms |
| `timeout_clear` 每 12 帧执行 | 1080 元素全量扫描由 300 次/秒降至 25 次/秒。效果微小（仅占 1%）但无代价 |
| 日志文件异步写入 (`enqueue=True`) | 消除 eMMC I/O 阻塞主线程导致的 GIL 争用，避免周期性点云覆盖率骤降。日志写入闪存卡 vfat，`tail -f` 约 1 分钟延迟为 cosmetic 问题 |
