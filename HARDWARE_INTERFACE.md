# 硬件接口接线文档

**项目**: Cooper_drone  
**开发板**: MYiR MYD-LD25X (STM32MP257)  
**雷达**: LDROBOT D500 单线激光雷达 ×2（上雷达 + 下雷达）

---

## 1. D500 雷达引脚定义（正面视角）

雷达正面（带 Logo 面），引脚从左到右：

| 序号 | 引脚名称 | 功能 | 连接说明 |
|---|---|---|---|
| 1 | 信号传输 (TX) | 雷达数据输出（UART TX） | 连接到开发板 UART RX 引脚 |
| 2 | 调速引脚 (PWM) | 控制雷达电机转速 | 可空置（默认最高转速），**建议接 3.3V** 避免电磁干扰触发 MCU 安全锁导致停转 |
| 3 | GND | 电源地 | 连接到开发板 GND |
| 4 | VCC | 电源输入 (5V) | 连接到开发板 5V 输出 |

> **注意**: 雷达 TX 端输出数据，需连接开发板 RX 端接收。连接为 **交叉连接**：雷达 TX → 开发板 RX。

---

## 2. 开发板 J13 树莓派接口引脚定义

J13 为 2×20 排针接口（Raspberry Pi 兼容），2.54mm 间距。正对 J13 丝印时，引脚编号自 **1 开始，从下往上、从左往右递增**。

> 表中加粗行表示已连接雷达，`(预留)` 表示可选替代引脚。

| 引脚 | 功能 | 连接 |
|------|------|------|
| 1 | VDD_3V3 (output) | **上雷达 PWM** |
| 2 | VDD_5V (output) | **上雷达 VCC** |
| 3 | PB4_I2C2_SDA | — |
| **4** | **VDD_5V (output)** | **下雷达 VCC** |
| 5 | PB5_I2C2_SCL | — |
| 6 | GND | — (预留 GND) |
| 7 | PD11_UART4_TX (GPIO) | — |
| 8 | PG8_UART9_TX | — |
| 9 | GND | **上雷达 GND** |
| **10** | **PI5_UART9_RX** | **下雷达 TX** |
| 11 | PB6_UART4_RX (GPIO) | **上雷达 TX** |
| 12 | PI9_FDCAN2_TX (GPIO) | — |
| 13 | PI6_USART3_TX (GPIO) | — (预留 USART3) |
| **14** | **GND** | **下雷达 GND** |
| 15 | PI7_USART3_RX (GPIO) | — (预留 USART3) |
| 16 | PB11_FDCAN1_RX | — |
| **17** | **VDD_3V3 (output)** | **下雷达 PWM** |
| 18 | PB9_FDCAN1_TX | — |
| 19 | PG11_SPI7_MOSI | — |
| 20 | GND | — |
| 21 | PG12_SPI7_MISO | — |
| 22 | PI10_FDCAN2_RX (GPIO) | — |
| 23 | PG13_SPI7_SCK | — |
| 24 | PI1_SPI7_NSS | — |
| 25 | GND | — |
| 26 | PZ8 (SPI7_CS1) | — |
| 27 | PG2_I2C3_SDA | — |
| 28 | PG1_I2C3_SCL | — |
| 29 | PZ0_SPI8_MOSI (GPIO) | — |
| 30 | GND | — |
| 31 | PF10_UART8_TX (GPIO) | — (预留 UART8) |
| 32 | PG3_ADC1_INP3 (GPIO) | — |
| 33 | PF11_UART8_RX (GPIO) | — (预留 UART8) |
| 34 | GND | — |
| 35 | PI3_USART1_CTS (GPIO) | — |
| 36 | PG4_ADC1_INP4 (GPIO) | — |
| 37 | PG14_USART1_TX | — (预留 USART1) |
| 38 | PI8 (GPIO) | — |
| 39 | GND | — |
| 40 | PG15_USART1_RX | — (预留 USART1) |

---

## 3. 接线对照表

### 3.1 上雷达（index=0）

```
上雷达引脚          →    J13 引脚
─────────────────────────────────
信号传输 (TX)       →    Pin 11 (UART4_RX)   ← 交叉连接
调速 (PWM)          →    Pin 1  (3.3V)       ← 拉高防电磁干扰
GND                 →    Pin 9  (GND)
VCC (5V)            →    Pin 2  (5V)
```

### 3.2 下雷达（index=1）

```
下雷达引脚          →    J13 引脚
─────────────────────────────────
信号传输 (TX)       →    Pin 10 (UART9_RX)   ← 交叉连接
调速 (PWM)          →    Pin 17 (3.3V)       ← 拉高防电磁干扰
GND                 →    Pin 14 (GND)
VCC (5V)            →    Pin 4  (5V)
```

**选线理由**：
- UART9 (Pin 8 TX, Pin 10 RX) 紧邻上雷达的 UART4 (Pin 11)，布线集中
- 供电就近取 Pin 4 (5V) 和 Pin 14 (GND)，与信号线在同一区域
- PWM 取 Pin 17 (3.3V)，靠近下雷达的其他引脚

### 3.4 雷达安装位姿参数

以上雷达扫描中心为机体坐标系原点 (+X=机头方向, +Y=左侧, 角度沿 +X 顺时针递增)。

| 参数 | 上雷达 (index=0) | 下雷达 (index=1) |
|------|------------------|-------------------|
| mount_xy_cm | (0.0, 0.0) | (0.96, 0.15) |
| mount_yaw_deg | 0.0 | 0.0 |
| mount_mirror_y | False | True |
| 安装高度 H (距地) | 待测 | 待测 |
| 扫描面倾角 | 0° (水平) | <2° (近似水平) |

### 3.3 接线示意图

> **注意**: D500 雷达正面引脚从左到右实际顺序为 **TX → PWM → GND → VCC**（TX 引脚靠外）。

```
           上雷达                          下雷达
   ┌──────────────────┐          ┌──────────────────┐
   │ TX  PWM  GND  VCC│          │ TX  PWM  GND  VCC│
   │  │    │    │    │ │          │  │    │    │    │ │
   └──┼────┼────┼────┼─┘          └──┼────┼────┼────┼─┘
      │    │    │    │               │    │    │    │
      ▼    ▼    ▼    ▼               ▼    ▼    ▼    ▼
     RX  3.3V  GND  5V             RX  3.3V  GND  5V
    Pin11 Pin1 Pin9 Pin2         Pin10 Pin17 Pin14 Pin4
   ┌────────────────────────────────────────────────┐
   │                   J13 排针                       │
   │            (MYD-LD25X 开发板)                    │
   └────────────────────────────────────────────────┘

---

## 4. 系统设备映射

| 项目 | 上雷达 (index=0) | 下雷达 (index=1) |
|------|------------------|-------------------|
| 设备路径 | `/dev/ttySTM4` | `/dev/ttySTM9` |
| 波特率 | 230400 | 230400 |
| 底层 UART | UART4 (PB6 RX) | UART9 (PI5 RX) |
| 信道配置 | `stty raw` 原始透传模式 | `stty raw` 原始透传模式 |
| 环境变量 | `RADAR0_PORT` | `RADAR1_PORT` |
| VID/PID | `10C4:EA60` | `10C4:EA60` |

> **注意**：两颗 D500 雷达 VID/PID 相同，`DeviceResolver.resolve_radar_port(index)` 按 index 顺序分配：index=0 → 第一颗，index=1 → 第二颗。若需指定特定雷达，可通过环境变量 `RADAR0_PORT` / `RADAR1_PORT` 直接指定设备路径。
>
> 若 `/dev/ttySTM9` 不存在，检查内核设备树是否使能了 UART9。备选方案：USART3 (Pin 13/15 → `/dev/ttySTM3`)、UART8 (Pin 31/33 → `/dev/ttySTM8`)。

---

## 5. USB 摄像头

### 5.1 物理连接

两个 USB 摄像头通过 USB 下行接口连接至开发板，OpenCV V4L2 读取。

```
开发板 USB Host
  ├── 1-1.1    → /dev/video7, /dev/video8    (障碍物识别摄像头)
  └── 1-1.2.3  → /dev/video9, /dev/video10   (路径识别摄像头)
```

各摄像头对应两个 `/dev/videoN` 节点：主节点为图像流，次节点通常为元数据（忽略）。

> ⚠️ **摄像头 index 在 OpenSTLinux 上与 Debian 12 互换了**。USB 枚举顺序因内核版本差异导致。`road_follow_main.py` 默认值已改为 `--camera-index 9`。

### 5.2 设备映射

| 项目 | 路径识别摄像头 (上/前视) | 障碍物识别摄像头 (下/前下视) |
|------|-----------|-----------|
| 设备路径 | `/dev/video9` | `/dev/video7` |
| USB 物理位置 | `usb-1.1` | `usb-1.2.3` |
| 型号标识 | `USB 2.0 Camera: USB Camera` | `USB Camera: USB Camera` |
| 分辨率 | 640×480 | 640×480 |
| 驱动 | V4L2 (cv2.CAP_V4L2) | V4L2 (cv2.CAP_V4L2) |
| cv2 index (OpenSTLinux) | **9** | **7** |
| cv2 index (Debian 旧) | 7 | 9 |
| **功能用途** | **道路路径识别** | **障碍物类型识别** |
| **色彩问题** | 偏青 (R/G=0.36, B/G=0.79) | 正常 |
| **白平衡** | 软件修正: `--wb-enable --wb-r 2.78 --wb-g 1.0 --wb-b 1.26` | 不需要 |
| **色彩诊断工具** | `FlightController/tools/diagnose_camera_color.py --index 9` | — |

### 5.3 验证命令

```bash
# 基本连通性
python3 -c "
import cv2
for idx, name in [(7, 'cam_front'), (9, 'cam_down')]:
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    ok, frame = cap.read()
    cap.release()
    print(f'{name} /dev/video{idx}: {\"OK\" if ok else \"FAILED\"} size={frame.shape[1]}x{frame.shape[0]}' if ok else f'{name}: FAILED')
"

# 拍摄测试照片
python3 -c "
import cv2
for idx, name in [(7, 'cam_front'), (9, 'cam_down')]:
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    ok, frame = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(f'/media/sdcard/{name}.jpg', frame)
        print(f'{name} -> /media/sdcard/{name}.jpg ({frame.shape[1]}x{frame.shape[0]})')
"
```

---

### 5.4 路径识别摄像头几何标定

实测标定数据（标尺法，飞行高度 17cm，倾角 25° 安装）：

| 参数 | 符号 | 值 | 说明 |
|------|------|-----|------|
| 安装高度 | H | 17 cm | 摄像头距地面 |
| 前向偏移 | cam_forward_offset_m | 10 cm | 摄像头在旋转中心前方距离 |
| 光轴倾角 | α | **30.27°** | 反推值（名义 25°，实际约 30°） |
| 垂直视场角 | VFOV | **55.08°** | 反推值 |
| 水平视场角 | HFOV | **~68°** | 由 VFOV + 宽高比 (640:480) 反推 |
| 半视场角 | β = VFOV/2 | 27.54° | |
| 近端水平距 | d_near | 10.7 cm | 画面最下行对应地面点 |
| 远端水平距 | d_far | 113.7 cm | 画面目标行对应地面点 |
| 目标行占比 | k | 89.52% | 103cm 标尺占画面垂直比例 |
| 图像尺寸 | — | 640×480 | |
| 目标行 m/px | — | ~0.0007 m/px | 道路中线出现区域（中下部） |

**反推公式**：

```
θ_near = arctan(H / d_near) = arctan(17 / 10.7) = 57.81°
θ_far  = arctan(H / d_far)  = arctan(17 / 113.7) = 8.50°
β = (θ_near - θ_far) / (2 × k) = 49.31° / 1.7904 = 27.54°
α = θ_near - β = 57.81° - 27.54° = 30.27°
```

**逐行 m/px 计算**（`road_perception.compute_meters_per_pixel`）：

```
θ(row) = α + β × (1 − 2 × row_from_bottom / 479)
D_ground = H / tan(θ)
meters_per_pixel_x = 2 × D_ground × tan(HFOV/2) / 640
```

| 行位置 (row from bottom) | 地面距离 (m) | meters_per_pixel_x |
|---|---|---|
| 0 (最下行) | 0.107 | 0.00033 |
| 120 (~1/4 处, 道路中线区域) | ~0.23 | ~0.00069 |
| 240 (中间) | ~0.53 | ~0.00114 |
| 479 (最上行) | — | ~0.00244 |

偏移补偿取道路中线所在行（约 row 120）的 `meters_per_pixel_x ≈ 0.0007 m/px`。

---
## 6. 外设诊断命令

### 6.1 快速诊断

```bash
# 串口设备一览
ls -la /dev/tty* 2>/dev/null | grep -E 'tty(USB|ACM|STM|AMA)'

# USB 设备树 + 雷达/飞控 VID:PID 识别
lsusb

# 摄像头设备
ls -la /dev/video*

# /dev/serial/by-id 符号链接
ls -la /dev/serial/by-id/ 2>/dev/null
```

### 6.2 设备 VID:PID 速查

| VID:PID | 设备 | 芯片 | 预期数量 |
|---------|------|------|:--:|
| `10C4:EA60` | D500 激光雷达 | CP210x USB-UART | 2 (仅 USB 连接时) |
| `66CC:2233` | 凌霄飞控 | — | 1 |
| `0BDA:3035` | USB 摄像头 | Realtek | 2 |

> **注意**: D500 雷达直接接在 J13 UART 引脚 (UART4/UART9) 时，不走 USB，`lsusb` 看不到。此时端口为 `/dev/ttySTM4` 和 `/dev/ttySTM9`。

### 6.3 内核日志排查

```bash
# 查看最近插入的 USB/串口设备
dmesg | grep -iE 'tty|cp210|usb.*hub|radar|uart|video' | tail -30

# 持续监控 (插拔设备时观察)
dmesg -w
```

### 6.4 虚拟环境激活

```bash
# 手动激活 (每次登录后)
source /usr/local/UFC_venv/bin/activate

# 开机自动激活 (开发板 shell 为 /bin/sh, 需写入 .profile)
echo 'source /usr/local/UFC_venv/bin/activate' >> ~/.profile
```

### 6.5 雷达数据录制与 PC 传输

```bash
# 开发板: 录制 (仅雷达)
cd ~/Desktop/ObstacleAvoidanceDrone
PYTHONPATH=. python record_data.py --no-camera --output-dir /media/sdcard/recordings

# PC: 下载录制的数据
scp -r root@192.168.31.199:/media/sdcard/recordings/<session_dir> .

# PC: 渲染可视化
python visualize_radar_data.py <session_dir> --video --video-fps 10
```

---

## 7. 参考文档

- 米尔开发板硬件用户手册: `MYD-LD25X-硬件用户手册-V1.3.pdf`
- 雷达驱动实现: `FlightController/Components/LDRadar_Driver.py`
- 雷达数据解析: `FlightController/Components/LDRadar_Resolver.py`
