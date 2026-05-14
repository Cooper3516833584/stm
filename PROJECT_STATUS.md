# 🚁 Cooper_drone 伴随计算机系统配置与开发进度纪要

**项目名称**: Cooper_drone (基于 MYiR 开发板的无人机机载/伴随计算机开发)
**当前阶段**: M4 阶段 (单雷达在线避障链路联调竣工，端到端延迟 < 500ms) + 基础设施迁移完成
**最后更新**: 2026年5月14日 (仓库迁移、闪存卡挂载、eMMC 空间清理)

---

## 1. 项目概述与硬件底盘状态
本项目旨在配置和开发一套适配于无人机的机载计算机系统，负责上层逻辑处理、激光雷达避障算法以及与底层匿名飞控（灵霄）的二进制串口协议通信（非 MAVLink）。

* **核心板**: MYiR MYD-LD25X (STM32MP257, Cortex-A35 双核架构)。
* **操作系统**: Debian 12 (Bookworm) aarch64, Linux 内核 (CONFIG_HZ=100)。
* **飞控通信**: 匿名飞控（灵霄）二进制协议，UART 串口 500000 baud。协议栈位于 `FlightController/Base.py` → `Protocal.py` → `Application.py`。
* **存储与内存极限管控 (Survival Mode)**:
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
7. **loguru 同步文件写入导致 GIL 争用**:
   * **症状**: `PROFILE` 日志行显示个别循环迭代耗时 88.9ms（正常 < 5ms），伴随 `Map_Circle` 有效点云数从 ~850 暴跌至 ~390，雷达吞吐降至 ~51%。
   * **根源**: `loguru.add(..., enqueue=False)` 同步写 eMMC 文件。当文件系统触发 journal commit 或写回缓存 flush 时，`write()` 调用阻塞主线程 80-90ms。Python GIL 在此期间被持有，串口读取线程无法执行 `Map_Circle.update()`，雷达帧积压在 `buf` 中无法处理，`timeout_clear` (0.15s) 清除过期点云导致覆盖率下降。
   * **修复**: 将 `enqueue=False` 改为 `enqueue=True`。日志消息放入内存队列（微秒级），后台独立线程异步写文件。主线程不再受 eMMC I/O 影响，GIL 及时释放，串口线程始终保持实时。

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

### 6.5 端到端性能验证

在 STM32MP257 真机上持续运行测试脚本验证:

| 指标 | 实测值 | 状态 |
|---|---|---|
| 串口吞吐 | 298-332 包/秒 (理论的 100-106%) | ✅ |
| CRC 错误 | 0 次 | ✅ |
| 数据年龄 (最新) | 2-4ms | ✅ |
| 数据年龄 (最旧) | 120-155ms | ✅ |
| get_points_body_cm 耗时 | 1.0-3.5ms | ✅ |
| planner 耗时 | 0.1-0.3ms | ✅ |
| 单次循环总耗时 | 2.5-7.0ms | ✅ |
| 端到端延迟 (障碍物出现→日志输出) | < 500ms | ✅ |
| 延迟增长趋势 (DELAY_TREND) | →稳定 (无增长) | ✅ |

### 6.6 诊断工具

`FlightController/tools/test_radar_avoidance.py` — 单雷达在线避障链路测试脚本，支持:

```bash
# 基础用法
PYTHONPATH=. python -u FlightController/tools/test_radar_avoidance.py --no-fc --dry-run

# 性能分析 + 异步日志文件
PYTHONPATH=. python -u FlightController/tools/test_radar_avoidance.py \
    --no-fc --dry-run --profile \
    --log-file /tmp/radar.log

# 实时监控日志 (另一 SSH 窗口)
tail -f /tmp/radar.log
```

**命令行参数一览**:

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--port` | `/dev/ttySTM4` | 雷达串口路径 |
| `--no-fc` | False | 不连接飞控，仅测试雷达端 |
| `--dry-run` | False | 不发送实际飞控指令 |
| `--max-distance-cm` | 300 | 障碍物检测最大距离/cm |
| `--stop-distance-cm` | 80 | 急停距离/cm |
| `--slow-distance-cm` | 150 | 减速距离/cm |
| `--cruise-speed-cm-s` | 30 | 巡航速度/cm/s |
| `--corridor-half-width-cm` | 50 | 前方走廊半宽/cm |
| `--min-distance-cm` | 10 | 最小检测距离/cm (过滤噪点) |
| `--loop-hz` | 10 | 主循环频率/Hz |
| `--debug-dump` | False | 打印前方走廊原始点云 |
| `--profile` | False | 性能分析模式 |
| `--log-file` | None | 异步日志文件路径 (绕过 SSH 缓冲) |

**Profile 模式输出指标**:
- 串口吞吐 (实际/理论 包/秒 + 百分比)
- CRC 错误累计 + 速率
- 数据年龄 (最新 + 最旧)
- Pipeline 各级耗时 (get / plan / total)
- 点云覆盖率 (有效点 vs 理论 1080 仓)
- DELAY_TREND (运行时间 vs 延迟趋势，每 ~30s 输出)

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

### 7.1 飞控联调 (阶段 B/C)
- [ ] `test_radar_avoidance.py --dry-run` → 连接飞控，确认状态回传正常
- [ ] `test_radar_avoidance.py` → 完整链路，确认 `send_realtime_control_data()` 正确下发
- [ ] 地面推车测试：手持障碍物靠近雷达，验证飞控收到的速度指令变化

### 7.2 室外实测
- [ ] 室外 10m 直径小范围低速 (<2 m/s) 飞行测试
- [ ] 调整避障参数适配室外环境 (走廊宽度、停止/减速距离)
- [ ] 评估单雷达盲区，确定是否需要双雷达（上下布局）

### 7.3 SLAM 在线化
- [ ] 将离线 Hough SLAM (`Radar_SLAM.radar_resolve_rt_pose`) 接入 `LDRadar_Driver._map_resolve_task()`
- [ ] 配置 `Navigation` 以 `mode="radar"` 运行，实现闭环悬停

### 7.4 避障策略升级
- [ ] 当前为纯前向走廊检测 (1D)，升级为基于势场法 (PFBPP) 或 VFH 的方向性避障
- [ ] 集成 `PathPlanner.py` 到在线 loop 中，替代简单的 stop/slow/cruise 三段逻辑

---

## 附录：关键性能调优决策记录

| 决策 | 理由 |
|---|---|
| Python 继续使用（不换 C++） | numpy / opencv / pyserial 底层均为 C 实现，Python 仅作胶水代码。串口修复后 pipeline 耗时 < 10ms，语言切换无收益 |
| 单雷达优先双雷达 (当前阶段) | 最小闭环原则：先端到端跑通单雷达避障，验证链路正确性后再加双雷达 |
| 反应式避障优先 SLAM 建图 | 紧急避障需要 < 200ms 响应，SLAM 位姿估计用于导航漂移校正，两者分层异步运行 |
| 测试脚本使用 busy-wait (`sleep(0)`) | 嵌入式专用设备，CPU 开销可接受 (~2% 单核)，换取消灭 sleep 精度带来的数据积压 |
| 日志文件异步写入 (`enqueue=True`) | 消除 eMMC I/O 阻塞主线程导致的 GIL 争用，避免周期性点云覆盖率骤降 |
