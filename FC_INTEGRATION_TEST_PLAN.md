# 飞控联调测试方案

**目标**: 验证开发板与匿名飞控（灵霄）的 USB 串口通信链路，逐步集成单雷达在线避障。

**物理连接**: 开发板 USB host → 飞控 MCU USB 接口

**飞控状态**: 无电机连接（安全），仅 MCU 上电

**协议**: 匿名飞控二进制协议，500000 baud，VID/PID `66CC:2233`

---

## 协议栈速览

```
FC_Controller (__init__.py)
  └─ FC_Application    — safe_takeoff(), 实时控制线程, wait 原语
       └─ FC_Protocol   — unlock/lock/takeoff/land/send_realtime_control_data()
            └─ FC_Base   — AA 22/AA 55 帧封装, 校验和, 心跳/ACK/重连, 状态解析
                 └─ SerialReaderBuffered — 缓冲批量读取, 起始位检测
                      └─ pyserial — 500000 baud, 0.5s 读超时
                           └─ DeviceResolver — VID/PID 66CC:2233 自动发现
```

**关键帧格式**:

- 发送帧: `AA 22 | option(1B) | len(1B) | payload(N) | checksum(1B)`
- 接收帧: `AA 55 | len(1B) | payload(N) | checksum(1B)`（起始位和校验和被 SerialReaderBuffered 剥离）
- 实时控制 (option=0x04): `struct.pack("<hhhh", vel_x, vel_y, vel_z, -yaw)`，无 ACK，高频发送
- 心跳: 每 250ms 发送 `option=0x00, payload=0x01`
- 断线: 500ms 无数据 → `connected=False`
- 状态回传 (cmd=0x01): 16 字段 struct unpack，含姿态/位置/速度/电池/模式/解锁状态

---

## Phase 0: 环境探活

**目的**: 在进入联调前确认开发环境正确。

```bash
# 1. 确认虚拟环境激活
which python && python --version
# 预期: ~/UFC_venv/bin/python, Python 3.11.x

# 2. 确认 PYTHONPATH
python -c "import FlightController; print('FlightController OK')"

# 3. 确认 FC 设备存在
ls /dev/serial/by-id/*66CC*
# 预期: /dev/serial/by-id/usb-*-66CC_2233-* → ../../ttyACM0 (或 ttyUSB0)

# 4. 确认雷达设备存在
ls /dev/ttySTM4
# 预期: /dev/ttySTM4

# 5. 确认日志输出路径（闪存卡）
ls /media/sdcard/
```

所有检查通过后再进入 Phase A。

---

## Phase A: 基础连通性

**目的**: 验证设备发现、串口打开、协议握手、状态回传。

**测试脚本**: [`debug/test_fc_connect.py`](debug/test_fc_connect.py)

```bash
# 自动探测端口
PYTHONPATH=. python debug/test_fc_connect.py

# 指定端口
PYTHONPATH=. python debug/test_fc_connect.py --port /dev/ttyACM0
```

**预期输出**:
```
=== Phase A: 飞控基础连通性测试 ===
正在连接飞控 (自动探测端口)...
飞控已连接，等待状态数据稳定...
mode  = 1 (定高)
unlock = False
bat   = 11.5V
alt   = 0cm (add) / 0cm (fused)
姿态  = roll=0.5 pit=1.2 yaw=45.0
速度  = vx=0 vy=0 vz=0
位置  = x=0 y=0
指令  = cid=0 cmd_0=0 cmd_1=0
[PASS] mode 非默认(>0)
[PASS] 电池电压合理 (>5V)
[PASS] connected=True
Phase A 全部通过！
飞控已断开。
```

**验证项**:
- 端口自动发现（不应报 `RuntimeError`）
- `wait_for_connection()` 在 5 秒内返回
- mode / unlock / bat 回传非默认值
- 3 项检查全部 PASS
- 无异常或超时

**失败排查**:
- 端口找不到 → `ls /dev/serial/by-id/*66CC*` 检查 USB 枚举
- 连接超时 → 确认波特率 500000，检查 USB 数据线
- 状态全零 → 飞控固件版本兼容性，确认二进制协议匹配

---

## Phase B: 指令下发

**目的**: 验证飞控接收并响应指令（ACK 机制、状态变化）。

```bash
PYTHONPATH=. python -c "
from FlightController import FC_Controller
from loguru import logger
import time

fc = FC_Controller()
fc.start_listen_serial(block_until_connected=True)
fc.wait_for_connection()

logger.info(f'当前 mode={fc.state.mode.value} unlock={fc.state.unlock.value}')

fc.set_flight_mode(2)  # HOLD_POS_MODE
fc.wait_for_last_command_done()
logger.info(f'切换后 mode={fc.state.mode.value} (预期=2)')

s = fc.state
logger.info(f'IMU: roll={s.rol.value:.1f} pit={s.pit.value:.1f} yaw={s.yaw.value:.1f}')
logger.info(f'位置: x={s.pos_x.value} y={s.pos_y.value} alt={s.alt_add.value}cm')
logger.info(f'速度: vx={s.vel_x.value} vy={s.vel_y.value} vz={s.vel_z.value}')
logger.info(f'电池: {s.bat.value:.1f}V')

fc.close()
"
```

**预期输出**:
```
当前 mode=1 unlock=0
切换后 mode=2 (预期=2)
IMU: roll=...
```

**验证项**:
- `set_flight_mode(2)` 不掉异常
- `wait_for_last_command_done()` 不超时
- mode 从旧值变为 2
- 所有 16 个状态字段可读无异常

**模式说明**:
| mode 值 | 含义 |
|---------|------|
| 1 | ALT_HOLD (定高) |
| 2 | HOLD_POS (定点，实时控制需此模式) |
| 3 | PROGRAM (程控) |

---

## Phase C: 实时控制

**目的**: 验证 `send_realtime_control_data()` 协议层（电机未接，只测通信不测响应）。

```bash
PYTHONPATH=. python -c "
from FlightController import FC_Controller
from loguru import logger
import time

fc = FC_Controller()
fc.start_listen_serial(block_until_connected=True)
fc.wait_for_connection()
fc.set_flight_mode(2)
fc.wait_for_last_command_done()

for i in range(10):
    fc.send_realtime_control_data(
        vel_x=10 if i < 5 else 0,
        vel_y=0, vel_z=0, yaw=0
    )
    logger.info(f'发送 #{i}: vx={10 if i<5 else 0}, mode={fc.state.mode.value}')
    time.sleep(0.2)

fc.send_realtime_control_data(0, 0, 0, 0)
time.sleep(0.1)
fc.close()
logger.info('实时控制测试完成')
"
```

**验证项**:
- 所有 `send_realtime_control_data()` 不掉异常
- `connected` 全程保持 True
- 无 SerialException / OSError

**说明**: 电机未接，`vel_x` 指令下发后飞控不会实际产生速度变化，`state.vel_x` 回传可能不反映指令值。此阶段仅验证协议层通信正常。

---

## Phase D: 避障链路集成 (--dry-run)

**目的**: 将已验证的 FC 通信与单雷达避障链路合并运行，FC 状态回传和雷达点云同时工作。

```bash
PYTHONPATH=. python -u FlightController/tools/test_radar_avoidance.py \
    --dry-run --profile \
    --raw-latency --raw-latency-stdout \
    --log-file /media/sdcard/fc_test.log
```

**预期输出**:
```
[#0010] FC[mode=2 unlock=0 bat=11.5V alt=0cm] | 点云=810点 | 前方=无 | 指令=(vx=30, vy=0, vz=0, yaw=0) | 原因=free_flight
[#0020] FC[mode=2 unlock=0 bat=11.5V alt=0cm] | 点云=809点 | 前方=无 | 指令=(vx=30, vy=0, vz=0, yaw=0) | 原因=free_flight
[RAW_LATENCY] 雷达帧年龄: 当前=1ms 区间峰值=16ms | 设备钟速=100.0x% | 串口buf峰值=3xxB 解析buf=0B
```

**验证项**:
- 日志中 `FC[mode=...]` 出现且值正常
- 点云数量稳定（800+）
- `设备钟速` 保持 100%（FC 通信未干扰雷达线程）
- `--dry-run` 确保指令不会实际下发到飞控

**手持障碍物测试**:
手持纸板靠近雷达前方，观察日志变化：
```
[#0050] FC[mode=2 ...] | 点云=815点 | 前方=75cm | 指令=(vx=0, vy=0, vz=0, yaw=0) | 原因=free_flight+obstacle_stop
[#0060] FC[mode=2 ...] | 点云=812点 | 前方=120cm | 指令=(vx=15, vy=0, vz=0, yaw=0) | 原因=free_flight+obstacle_slow
[#0070] FC[mode=2 ...] | 点云=808点 | 前方=无 | 指令=(vx=30, vy=0, vz=0, yaw=0) | 原因=free_flight
```

---

## Phase E: 完整链路

**目的**: 去除 `--dry-run`，实际下发飞控指令。

```bash
PYTHONPATH=. python -u FlightController/tools/test_radar_avoidance.py \
    --profile --raw-latency --raw-latency-stdout \
    --log-file /media/sdcard/fc_full.log
```

**验证项**:
- 飞控实际接收实时控制数据（可通过飞控调试口确认）
- 无障碍时 vx=30cm/s 正常下发
- 手持障碍物靠近 → 急停 vx=0 立即下发
- `raw-latency` 指标持续正常

---

## 安全说明

| 阶段 | 雷达 | FC 连接 | 指令下发 | 电机 | 脚本 |
|------|------|---------|---------|------|------|
| 0 | 否 | 否 | 否 | 无 | 手动命令 |
| A | 否 | 是 | 否 | 无 | `debug/test_fc_connect.py` |
| B | 否 | 是 | 模式切换 | 无 | (见下方) |
| C | 否 | 是 | 实时控制 | 无 | (见下方) |
| D | 是 | 是 | `--dry-run` 阻断 | 无 | `FlightController/tools/test_radar_avoidance.py` |
| E | 是 | 是 | **是** | **无** | `FlightController/tools/test_radar_avoidance.py` |

所有阶段均可 Ctrl+C 安全退出，`finally` 块自动发送零速度指令。

---

## 预期问题速查

| 症状 | 可能原因 | 排查命令 |
|------|---------|---------|
| `RuntimeError: No FC port found` | USB 未枚举 | `ls /dev/serial/by-id/*66CC*` |
| `wait_for_connection` 超时 | 波特率不匹配 | 确认 `FC_Base_Uart_Comunication._serial_baudrate = 500000` |
| 状态全零 | 飞控固件协议版本不匹配 | 用逻辑分析仪抓 AA 55 帧 |
| `send_realtime_control_data` 掉 OSError | USB 断连 | `dmesg \| grep -i usb` 检查内核日志 |
| `设备钟速` 掉到 < 95% | FC 线程干扰雷达线程 | 检查 FC 心跳频率 (250ms)，理论上无干扰 |
