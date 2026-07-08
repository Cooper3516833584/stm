# .nb 模型诊断报告 — NPU 未生效

**日期**: 2026-07-08
**开发板**: MYD-LD25X (STM32MP257), OpenSTLinux v6.0, Kernel 6.6.48
**.nb 文件**: `FlightController/Solutions/model/road_yolo11n_seg_1.nb` (6.27 MiB)
**Python API**: `stai_mpu.stai_mpu_network` (libstai-mpu 6.0.1)

---

## 1. 最终结论（TL;DR）

**.nb 模型加载成功但 NPU 硬件未被使用**。推理 669ms 走的是纯 CPU 路径，且在第二次推理时 segfault。ST Cloud 产出的这个 `.nb` 文件不能用于生产。

**两道致命问题**:
1. 模型走 CPU 回退 — 没有 `/dev/galcore` ioctl → NPU 未工作
2. 第二次推理 segfault — 与之前 VSINPU EP 的空指针 bug 表现一致

---

## 2. 测试过程与关键发现

### 2.1 软件栈就绪 ✅

| 组件 | 状态 | 证据 |
|---|---|---|
| `python3-libstai-mpu` (6.0.1) | ✅ | `from stai_mpu import stai_mpu_network` OK |
| `stai-mpu-ovx` | ✅ | `/usr/lib/libstai_mpu_ovx.so.6` 动态加载 |
| `/dev/galcore` | ✅ | `crw-rw-rw-` VIP9000 model=0x8000, rev=6205 |
| 模型加载 | ✅ | 683ms 首次加载, 无报错 |

### 2.2 模型元数据

| 属性 | 实际值 | 期望值 | 差距 |
|---|---|---|---|
| 输入名称 | `input_0` | `images` | ❌ 不匹配管线 |
| 输入 shape | `[1, 3, 416, 416]` | `[1, 3, 320, 320]` | ⚠️ 代码可动态适配 |
| 输入 dtype | float16 | int8/uint8 | ❌ KPU 只加速 INT8 |
| 输出 0 name | `output_0` | `output0` | ❌ 不匹配 `_decode_yolo_segmentation` |
| 输出 0 shape | `[1, 37, 3549]` | `[1, C, N]` | ✅ |
| 输出 1 name | `output_1` | `output1` | ❌ |
| 输出 1 shape | `[1, 32, 104, 104]` | `[1, M, H, W]` | ✅ |
| 输出 dtype | float16 | int8/uint8 | ❌ 浮点路径 |

### 2.3 推理性能

| 指标 | 实测值 | 预期 NPU | 结论 |
|---|---|---|---|
| 首次推理 | 638.8 ms | 5-15 ms | **CPU 水平** |
| 后续推理 (5 runs) | mean=669 ms | ≤ 15 ms | CPU 水平 |
| 第二次 run() | **Segfault** | — | ❌ NPU 路径有 bug |
| 并发 FPS | ~1.5 | 60-200 | 不可用 |

### 2.4 strace 关键证据 — NPU 从未被调用

```bash
strace -f -e ioctl python3 -c "... m.run() ..." | grep ioctl
```

输出 **只有 TCGETS**（终端查询 ioctl），**没有任何对 `/dev/galcore` 的 ioctl 调用**。

```
ioctl(3, TCGETS, ...) = -1 ENOTTY   ← 终端查询
ioctl(0, TCGETS, ...) = 0           ← stdin
ioctl(1, TCGETS, ...) = -1 ENOTTY   ← stdout
ioctl(2, TCGETS, ...) = -1 ENOTTY   ← stderr
                                    ← 没有任何 galcore ioctl！
```

**结论**: `libstai_mpu_ovx.so` 虽然被加载了（"Loading dynamically"）并显示了 "[OVX]: Loading nbg model"，但实际的 `run()` 调用没有触发 NPU 硬件。整个推理走了 CPU fallback。

### 2.5 Segfault 复现

运行两次空输入推理（随机 float16 blob）后崩溃：

```
Loading dynamically: /usr/lib/libstai_mpu_ovx.so.6
[OVX]: Loading nbg model
run 0: 638.8 ms         ← 第一次跑完
Segmentation fault      ← 第二次 run() 崩溃
```

这与此前 [NPU_MODEL_STAGE0_FAILURE_REPORT.md](NPU_MODEL_STAGE0_FAILURE_REPORT.md) 记录的 VSINPU EP 空指针 bug 表现一致——即使走 CPU 路径，`libstai_mpu_ovx` 内部的内存管理仍存在问题。

---

## 3. 根因分析

### 3.1 问题链

```
ST Cloud (STM32MP2 target)
  │
  ├── 生成 .nb (VPMN 格式)
  │      └── 这应该是一个 INT8 量化网络图
  │
  └── 实际产物: float16 输入/输出模型
         │
         ├── NPU (VIP9000) 不支持 float16 算子加速
         │      └── 触发 CPU fallback → 669ms
         │
         └── CPU 路径内存管理有 bug → 第二次推理 segfault
```

### 3.2 为什么 NPU 没有被调用

VIP9000 (VeriSilicon VIP 0x8000) 只对 **INT8/INT16 量化**的算子有硬件加速。float16 的网络层需要回退到 CPU 运行，因为：

1. VIP9000 的 MAC 阵列只加速定点 (integer) 运算
2. float16 在图内部被 `libstai_mpu_ovx` 识别为需要 CPU fallback
3. 因为所有 162 个 epoch 都回退 CPU，等同于 `onnxruntime` 纯 CPU 路径

### 3.3 ST Cloud 配置回顾

根据 [NPU_MODEL_CONVERSION_PLAN.md](NPU_MODEL_CONVERSION_PLAN.md) §4.1 的操作指引，**正确的 STM32MP2 转换流程**应该是：

```
Quantize (INT8, per-tensor, 道路校准)
  → Optimize (算子替换)
    → Benchmark (验证 NPU 延迟)
      → Generate (.nb + .onnx)
```

但实际 ST Cloud 的 "Optimize & Generate" 是合并步骤（文档注明了），实际生成过程中可能发生了一个或多个问题：

| 可能原因 | 证据 |
|---|---|
| **量化未真正生效** | 输入输出全是 float16，没有 QDQ 节点等价物 |
| **校准数据缺失** | PerChannel 和 PerTensor 两个量化模型都是用 `random` 数据 |
| **Target 选错** | 之前 ST Cloud 日志显示选了 `stm32n6`，不是 `stm32mp25` |
| **Generate 阶段参数不对** | `.nb` 文件确实生成了（6.27 MiB, VPMN 头）但内容不含 NPU 加速路径 |

---

## 4. 代码侧已做好的适配

以下改动已提交，**不依赖 .nb 正常**（`_AUTO_USE_NPU = False` 时完全走回原 ONNX 路径）：

### 4.1 新增文件

| 文件 | 用途 |
|---|---|
| [nb_graph.py](nb_graph.py) | `NBGraphSession` — 封装 `stai_mpu.stai_mpu_network`，接口兼容 `onnxruntime.InferenceSession` |
| [FlightController/tools/test_nb_model.py](FlightController/tools/test_nb_model.py) | .nb 模型烟雾测试脚本 |

### 4.2 修改文件

| 文件 | 改动摘要 |
|---|---|
| [road_perception.py](road_perception.py) | `MODEL_PATH_NPU` 常量、`_AUTO_USE_NPU` 开关、`_resolve_model_path()` 返回 `(path, is_nb)`、`_make_session()` 分流 .nb 路径 |
| [road_follow_main.py](road_follow_main.py) | `--model-npu` CLI 参数 |
| [bench_vision_fps.py](FlightController/tools/bench_vision_fps.py) | `--model-npu` CLI 参数 |

### 4.3 .nb 模型还需要的代码修改

当前 .nb 的输入/输出名称（`input_0`, `output_0`, `output_1`）与 ONNX 原始名称（`images`, `output0`, `output1`）不同。`nb_graph.py` 的 `run()` 已按模型实际名称存取，但 `road_perception.py` 的 `_decode_yolo_segmentation()` 会直接对 outputs 数组按索引操作，不依赖名称——**这一项无需额外修改**。

唯一需要关注的是 `_preprocess()` 中 BGR→RGB 转换：ONNX 量化管线中的校准数据也做了 BGR→RGB（见 `calibration_manifest.json`），`.nb` 如使用相同校准数据，此步骤一致。

---

## 5. 解决路径（按优先级排序）

### 路径 A：回 ST Cloud 重新生成 .nb（推荐 ⭐）

确保以下配置全部正确：

| 配置项 | 必须值 | 当前状态 |
|---|---|---|
| **Target 平台** | `STM32MP2` (STM32MP257F-EV1) | ❌ 之前选过 stm32n6 |
| **Runtime** | `ONNX Runtime (X-LINUX-AI)` | 需确认 |
| **量化精度** | INT8 per-tensor | ⚠️ 模型显示 float16 |
| **校准数据** | 道路场景 `.npz`（`road_yolo11n_seg_calibration.npz`） | ⚠️ 之前用 random |
| **输入尺寸** | 320×320（与代码一致，或 416×416 代码可适配） | 当前是 416 |
| **优化目标** | VIP9000 NPU 加速，非 CPU fallback | ❌ 当前是 CPU |

**操作要点**:
1. 进入 ST Cloud → 选择 STM32 MPUs 平台（不是 MCUs）
2. 上传 `road_yolo11n_seg.onnx`（11.5 MB, FP32）
3. Quantize: 勾选 "Disable per channel quantization", 上传 `road_yolo11n_seg_calibration.npz`
4. Optimize: 确保算子替换开启（ConvTranspose → Resize+Conv）
5. Generate: 下载 `.nb` + `.onnx` 两种格式
6. 验证元数据：输入名为 `images`（不是 `input_0`），dtype 为 int8/uint8

### 路径 B：在开发板上直接用 VSINPU EP + ONNX

如果 `.nb` 始终无法工作，回退到 ONNX 路径。已确认 VSINPU EP 的 ONNX Runtime 1.19.2 注册成功（348/353 nodes），问题出在复杂图的多分区空指针 bug。如果 St Cloud 能给出一个经过 Optimize 的 ONNX（没有 ConvTranspose、节点数少），可能不触发多分区路径：

```bash
# 用 VSINPU EP 加载 Optimize 后的 ONNX
PYTHONPATH=. python3 -c "
import road_perception
road_perception._AUTO_USE_NPU = False
road_perception.MODEL_PATH = 'FlightController/Solutions/model/road_yolo11n_seg_npu.onnx'
result = road_perception.get_road_perception(frame)
"
```

### 路径 C：升级 onnxruntime

联系 ST 获取 onnxruntime ≥ 1.20 的 VSINPU EP 补丁（见 [NPU_MODEL_STAGE0_FAILURE_REPORT.md](NPU_MODEL_STAGE0_FAILURE_REPORT.md) §7.2），修复跨分区空指针 bug。

---

## 6. 当前开发板上的快速回退

如果现在需要在开发板上跑视觉管线，有一条临时可用路径：

```bash
# 使用本地量化 ONNX + XNNPACK CPU 推理
PYTHONPATH=. python3 -c "
import road_perception
road_perception._AUTO_USE_NPU = False
road_perception.MODEL_PATH = 'FlightController/Solutions/model/npu_quantization/road_yolo11n_seg_vsinpu_int8_qdq.onnx'
# XNNPACK 自动选择 (VSINPU 会 crash, _select_providers fallback 到 XNNPACK)
"
```

预期推理 400-600ms（XNNPACK ARM SIMD），比纯 CPU ~1800ms 快 3-4 倍，不会 segfault。

---

## 7. 诊断命令速查

```bash
# NPU 硬件确认
cat /sys/kernel/debug/gc/info
ls -la /dev/galcore

# stai_mpu 可用性
python3 -c "from stai_mpu import stai_mpu_network; print('OK')"

# .nb 模型烟雾测试
PYTHONPATH=. python3 FlightController/tools/test_nb_model.py

# .nb 模型 + 真实图片测试
PYTHONPATH=. python3 FlightController/tools/test_nb_model.py \
  --image adjustment/roads/IPC_2026-06-14.10.32.58.1790.jpg

# 抓取 NPU 是否被调用的证据
strace -f -e ioctl python3 -c "
from stai_mpu import stai_mpu_network
import numpy as np
m = stai_mpu_network('FlightController/Solutions/model/road_yolo11n_seg_1.nb', use_hw_acceleration=True)
m.set_input(0, np.random.rand(1,3,416,416).astype(np.float16))
m.run()
" 2>&1 | grep -c 'galcore'
# 期望: 大量 galcore ioctl 行（每个 epoch 至少一次）
# 实际: 0 → NPU 未使用

# ONNX 路径回退
PYTHONPATH=. python3 road_follow_main.py --camera-index 9 --no-fc --no-radar
```
