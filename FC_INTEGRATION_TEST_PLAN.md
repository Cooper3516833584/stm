# 分阶段联调验收计划

## 总原则

任何阶段失败，都不要进入下一阶段。真实发送速度指令前必须满足：

- 已完成无硬件静态检查。
- 飞控基础通信、模式切换、实时控制帧通信全部通过。
- 雷达 freshness watchdog 通过。
- 道路循迹 dry-run 通过。
- 飞机已拆桨或固定，遥控器可随时接管。

所有自主任务入口默认 dry-run，不自动解锁、不自动起飞、不自动降落。

---

## Phase 0：静态检查

```bash
cd /path/to/stm
export PYTHONPATH=.

python -m py_compile road_perception.py
python -m py_compile FlightController/Components/LDRadar_Driver.py
python -m py_compile FlightController/Components/MultiRadar.py
python -m py_compile FlightController/Solutions/LocalPlanner.py
python -m py_compile FlightController/Solutions/Safety.py
python -m py_compile FlightController/Solutions/RoadFollower.py
python -m py_compile FlightController/Solutions/RelativeGoalNavigator.py
python -m py_compile FlightController/tools/test_radar_avoidance.py
python -m py_compile FlightController/tools/test_dual_radar.py
python -m py_compile road_follow_main.py goal_nav_main.py

PYTHONPATH=. python FlightController/tools/validate_imports.py
grep -R "explicit""_port" -n .
```

通过标准：

- 无 `SyntaxError`。
- 无 `ModuleNotFoundError`。
- `grep -R "explicit""_port" -n .` 无结果。

---

## Phase A：飞控基础连通性

前提：

- 飞控只接 USB 或串口。
- 电机可不接。
- 不解锁。

```bash
PYTHONPATH=. python debug/test_fc_connect.py

# 如果自动识别失败
PYTHONPATH=. python debug/test_fc_connect.py --port /dev/ttyACM0
```

通过标准：

- 能看到 `connected=True`。
- 能读到 `mode`、`unlock`、`roll/pit/yaw`、`alt` 等状态。
- USB 供电时 `bat=0.0V` 只作为 warning，不作为失败。

---

## Phase B：飞控模式切换

```bash
PYTHONPATH=. python debug/test_fc_command.py --target-mode 2
```

通过标准：

- `wait_for_last_command_done()` 成功。
- 最终 `mode=2`。
- 无串口异常。

---

## Phase C：飞控实时控制帧通信

```bash
PYTHONPATH=. python debug/test_fc_realtime.py --count 10 --speed 10
```

通过标准：

- 10/10 指令发送成功。
- 0 异常。
- 0 断连。
- 最终 `connected=True`。

注意：此阶段只验证协议通信，不验证实际飞行响应。

---

## Phase D：单雷达地面 dry-run

```bash
PYTHONPATH=. python -u FlightController/tools/test_radar_avoidance.py \
  --no-fc --dry-run \
  --profile --raw-latency --raw-latency-stdout \
  --log-file /media/sdcard/single_radar_verify.log
```

通过标准：

- 雷达 `connected=True`。
- CRC 错误持续为 0 或极低。
- `last_frame_age` 稳定小于 `0.5s`。
- 前方放纸板时，最近障碍物距离能变化。
- 小于 stop 阈值时，规划输出 `vx=0`。

---

## Phase E：双雷达地面 dry-run

```bash
PYTHONPATH=. python -u FlightController/tools/test_dual_radar.py \
  --no-fc --dry-run --profile \
  --log-file /media/sdcard/dual_radar_verify.log
```

通过标准：

- 上下雷达都有点数。
- 上下雷达 `last_frame_age` 都小于 `0.5s`。
- 机身屏蔽点数合理。
- 前方纸板能触发 slow/stop。
- 拔掉任一雷达 TX 后，约 `0.5s` 进入 radar stale / safety stop。

---

## Phase F：双雷达 + 飞控只读

```bash
PYTHONPATH=. python -u FlightController/tools/test_dual_radar.py \
  --dry-run --fc-port /dev/ttyACM0 \
  --log-file /media/sdcard/dual_radar_fc_dry.log
```

通过标准：

- 飞控 `connected=True`。
- 飞控 `mode=2`。
- 日志里显示 safety 决策，但 `sent=False` 或 dry-run。
- 没有真实发送非零速度。

---

## Phase G：视觉道路感知离线验证

前提：准备道路图片，包括：

- 单路。
- 左岔。
- 右岔。
- 十字。
- 阴影、反光、树荫等困难样本。

```bash
PYTHONPATH=. python road_perception.py \
  --image samples/road_single.jpg \
  --model FlightController/Solutions/model/road_yolo11n_seg.onnx \
  --debug-out /media/sdcard/road_debug_single.jpg
```

可用不同分支偏好重复测试：

```bash
PYTHONPATH=. python road_perception.py \
  --image samples/road_cross.jpg \
  --model FlightController/Solutions/model/road_yolo11n_seg.onnx \
  --branch-preference left \
  --debug-out /media/sdcard/road_debug_cross_left.jpg
```

通过标准：

- 单路中心线稳定。
- 单主路结果满足 `branches=[]`、`selected_branch=None`，且 `centerline_points` 数量合理。
- `branch_preference=left/right/straight` 能改变 `selected_branch`。
- 无模型时返回 `lost`，不导致主程序崩溃。

---

## Phase H：道路循迹主程序 dry-run

无飞控、无雷达，仅摄像头/模型：

```bash
PYTHONPATH=. python road_follow_main.py \
  --no-fc --no-radar --dry-run \
  --model FlightController/Solutions/model/road_yolo11n_seg.onnx \
  --loop-hz 5
```

有雷达、无飞控：

```bash
PYTHONPATH=. python road_follow_main.py \
  --no-fc --dry-run \
  --model FlightController/Solutions/model/road_yolo11n_seg.onnx \
  --loop-hz 5
```

有飞控只读：

```bash
PYTHONPATH=. python road_follow_main.py \
  --dry-run --fc-port /dev/ttyACM0 \
  --model FlightController/Solutions/model/road_yolo11n_seg.onnx
```

通过标准：

- 日志持续输出 `road_state`、`pixel_error`、`angle`、desired command、safety decision。
- 丢失道路时进入 lost/search/zero，不崩溃。
- 雷达超时时 safety stop。
- 默认不真实发送非零速度。

---

## Phase I：拆桨/固定机体真实发送测试

前提必须全部满足：

- 飞控三件套通过。
- 单雷达/双雷达 dry-run 通过。
- `road_follow_main.py` dry-run 通过。
- 飞机拆桨或固定。
- 人手可随时切手动/锁定。

```bash
PYTHONPATH=. python road_follow_main.py \
  --enable-flight \
  --fc-port /dev/ttyACM0 \
  --model FlightController/Solutions/model/road_yolo11n_seg.onnx \
  --max-vx-cm-s 10 \
  --max-yaw-rate-deg-s 10
```

通过标准：

- 指令速度被限幅。
- Ctrl+C 后发送零速度。
- 断雷达、遮挡道路、切飞控模式时立即 safety stop。
- 无异常堆栈导致指令线程失控。

---

## Phase J：低风险实飞前检查

在任何实飞前，必须人工确认：

- `--enable-flight` 是人为显式传入，不是默认值。
- `max_vx_cm_s` 初始很小，例如 `10 cm/s`。
- `max_yaw_rate_deg_s` 初始很小，例如 `10 deg/s`。
- 遥控器可以立刻接管。
- 飞控定点模式稳定。
- 光流/高度保持本身稳定。
- 电池电压检查参数设置合理。
- 场地空旷，无人靠近。

禁止直接做的事：

- 禁止首次运行就带桨开启 `--enable-flight`。
- 禁止自动解锁起飞。
- 禁止雷达未通过 freshness watchdog 时飞行。
- 禁止摄像头偏移补偿未标定时把它作为强依赖。
- 禁止把 relative goal demo 当成全局自主导航。
