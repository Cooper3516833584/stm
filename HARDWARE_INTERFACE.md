# 硬件接口接线文档

**项目**: Cooper_drone  
**开发板**: MYiR MYD-LD25X (STM32MP257)  
**雷达**: LDROBOT D500 单线激光雷达 ×2（上雷达 + 下雷达）

---

## 1. D500 雷达引脚定义（正面视角）

雷达正面（带 Logo 面），引脚从左到右：

| 序号 | 引脚名称 | 功能 | 连接说明 |
|---|---|---|---|
| 1 | 调速引脚 (PWM) | 控制雷达电机转速 | 可空置（默认最高转速），**建议接 3.3V** 避免电磁干扰触发 MCU 安全锁导致停转 |
| 2 | 信号传输 (TX) | 雷达数据输出（UART TX） | 连接到开发板 UART RX 引脚 |
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
调速 (PWM)          →    Pin 1  (3.3V)       ← 拉高防电磁干扰
信号传输 (TX)       →    Pin 11 (UART4_RX)   ← 交叉连接
GND                 →    Pin 9  (GND)
VCC (5V)            →    Pin 2  (5V)
```

### 3.2 下雷达（index=1）

```
下雷达引脚          →    J13 引脚
─────────────────────────────────
调速 (PWM)          →    Pin 17 (3.3V)       ← 拉高防电磁干扰
信号传输 (TX)       →    Pin 10 (UART9_RX)   ← 交叉连接
GND                 →    Pin 14 (GND)
VCC (5V)            →    Pin 4  (5V)
```

**选线理由**：
- UART9 (Pin 8 TX, Pin 10 RX) 紧邻上雷达的 UART4 (Pin 11)，布线集中
- 供电就近取 Pin 4 (5V) 和 Pin 14 (GND)，与信号线在同一区域
- PWM 取 Pin 17 (3.3V)，靠近下雷达的其他引脚

### 3.3 接线示意图

```
           上雷达                          下雷达
   ┌──────────────────┐          ┌──────────────────┐
   │ PWM  TX  GND  VCC│          │ PWM  TX  GND  VCC│
   │  │    │    │    │ │          │  │    │    │    │ │
   └──┼────┼────┼────┼─┘          └──┼────┼────┼────┼─┘
      │    │    │    │               │    │    │    │
      ▼    ▼    ▼    ▼               ▼    ▼    ▼    ▼
    3.3V  RX   GND  5V            3.3V  RX   GND  5V
    Pin1 Pin11 Pin9 Pin2         Pin17 Pin10 Pin14 Pin4
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

## 5. 参考文档

- 米尔开发板硬件用户手册: `MYD-LD25X-硬件用户手册-V1.3.pdf`
- 雷达驱动实现: `FlightController/Components/LDRadar_Driver.py`
- 雷达数据解析: `FlightController/Components/LDRadar_Resolver.py`
