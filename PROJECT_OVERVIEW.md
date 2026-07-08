# Cooper_drone 项目功能与技术特点综述

## 项目概述

**Cooper_drone** 是一套基于 MYiR MYD-LD25X（STM32MP257, Cortex-A35 双核 2GB RAM）开发板的无人机机载/伴随计算机系统，运行 **OpenSTLinux v6.0**（Yocto Scarthgap, Linux 6.6.48），负责上层感知、规划与避障逻辑，通过二进制串口协议（非 MAVLink）与底层匿名凌霄飞控通信。

## 核心功能

项目实现**两类自主飞行任务**：

1. **道路视觉循线**（`road_follow_main.py`）：通过 USB 摄像头采集道路图像，利用 YOLO11-seg ONNX 模型进行语义分割，提取道路中线几何信息（像素偏差、中线角度），经 `RoadFollower` P 控制律转换为飞控速度指令（vx + yaw_rate），支持直行/左转/右转分叉决策。

2. **雷达相对目标导航**（`goal_nav_main.py`）：基于双 D500 单线激光雷达融合点云，采用**候选方向管状碰撞检测算法**——在前向 150° 范围内以 2° 步长生成 75 个候选方向，对每个方向构建管状区域进行障碍物距离评估，综合方向偏离代价、安全间隙代价和切换代价选择最优航向，输出避障速度指令。

## 硬件拓扑

- **双雷达**：上雷达（`/dev/ttySTM4`, 水平安装 360° 扫描）与下雷达（`/dev/ttySTM9`, 下倾覆盖前下方盲区），均为 LDROBOT D500，转速 ~600 RPM，230400 baud
- **双摄像头**：道路识别摄像头（`/dev/video9`）与障碍物识别摄像头（`/dev/video7`），均 640×480，OpenCV V4L2 读取；已实现软件白平衡修正
- **飞控通信**：凌霄飞控 UART 500000 baud，二进制协议栈（Base → Protocal → Application）
- **存储**：SD 卡 30G 系统盘 + eMMC 7.3G 备用，日志写入 `/media/sdcard` 严禁占用 RAM

## 设计流程

项目采用**渐进式迭代开发**，从底层硬件探活逐步向高层自主飞行推进：

1. **硬件驱动层**：首先打通 D500 雷达串口数据泵，实现 `Map_Circle` 内存池高频刷新与 CRC 校验，确保物理链路稳定。随后接入双雷达、双摄像头、飞控串口，完成全硬件拓扑验证。
2. **离线算法验证**：在 Headless 环境下通过离线渲染图像，利用霍夫直线检测在纸盒封闭场景中验证 SLAM 位姿解算的可行性，证明点云到坐标系的降维通路畅通。
3. **在线联调**：M4 阶段完成单雷达避障闭环，逐步修复 ARM 内核 sleep 精度、GIL 争用等嵌入式平台特有问题。
4. **NPU 加速攻关**：因 YOLO11-seg CPU 推理仅 0.6 FPS，启动 Debian → OpenSTLinux 的 OS 迁移，使能 VIP9000 NPU。当前 NPU 驱动就绪，模型转换中。
5. **安全加固与数据闭环**：建立 SafetyArbiter 多重防线，实现异步数据记录器支撑离线回放分析。

## 技术特点

### 1. 分层架构与安全仲裁

系统分为四层：**感知层**（道路感知 `road_perception.py` / 雷达世界模型）、**规划层**（`RelativeGoalNavigator` 候选方向搜索 / `RoadFollower` P 控制）、**安全仲裁层**（`SafetyArbiter`，硬故障检查包括雷达超时 > 0.5s、非定点模式、电压 < 10V、倾角 > 25°、前方障碍 < 80cm 紧急停车）、**任务层**（步进状态机 PLAN → EXECUTE → HOLD）。所有飞控指令默认 dry-run，须显式 `--enable-flight` 才实际发送，且 `finally` 块保证异常退出时发送零速指令。

### 2. NPU 加速与 OS 迁移

项目经历了从 Debian 12 到 OpenSTLinux v6.0 的完整 OS 迁移，根本原因是 STM32MP257 内置的 VeriSilicon VIP9000 NPU（800MHz）软件栈深度绑定 OpenSTLinux BSP（glibc 2.38+、gcnano 驱动、libArchModelSw 等）。YOLO11-seg 在 Cortex-A35 CPU 上推理仅 ~0.6 FPS，NPU 预期加速至 60-200 FPS。迁移后 NPU 驱动栈就绪（galcore 6.4.19、onnxruntime 1.19.2 含 VSINPU EP），但原模型含 ConvTranspose/dilated NonMaxPool 等不兼容算子，需经 ST Edge AI Cloud 转换后方可在 NPU 上运行。

### 3. ARM 嵌入式性能优化

解决了一系列嵌入式 ARM 平台特有的性能陷阱：
- **100Hz 内核 sleep 精度**：`time.sleep(0.001)` 实际睡眠 ~10ms，导致雷达数据积压，改为 `time.sleep(0)` busy-wait 方案
- **GIL 争用**：loguru 同步文件写入阻塞主线程 80-90ms，改为异步队列写入（`enqueue=True`），配合串口批量读取策略
- **混合架构 Python 隔离舱**：采用 `virtualenv --system-site-packages` 半透膜沙盒，APT 底层预编译（numpy/opencv）+ PIP 上层轻量包，规避 ARM64 现场编译 OOM

### 4. 步进式避障规划

目标导航采用**步进状态机**（PLAN → EXECUTE → HOLD 循环）：每次前进最多 2 秒（钳位值），到期重新规划；执行阶段每循环用最新雷达数据重新评估安全性，检测到前方新障碍时立即打断前进。`RelativeGoalNavigator` 核心算法对 75 个候选方向进行管状区域投影碰撞检测，结合阻塞释放迟滞（clearance 恢复至 90cm 才解除阻塞）和方向切换迟滞（成本差 ≤ 5.0 保持原方向），在避障安全性与飞行平稳性之间取得平衡。

### 5. 数据记录系统

`record_data.py` 实现雷达原始数据 + 相机帧异步记录器，后台线程将队列数据压缩写入 SD 卡（`np.savez_compressed` + `cv2.imencode` JPEG），主循环采样不受磁盘 I/O 阻塞影响。配合 `visualize_radar_data.py` 可离线回放和可视化雷达点云与 SLAM 结果。
