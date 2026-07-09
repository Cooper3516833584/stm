# ST Cloud 与 STM32MP257 NPU 模型转换实测记录

**日期**: 2026-07-09  
**平台**: MYD-LD25X / STM32MP257 / OpenSTLinux v6.0  
**目标模型**: `road_yolo11n_seg.onnx` / YOLO11n-seg road segmentation  
**核心结论**: 当前已验证 ST Cloud 可以为 FP32 模型生成 `.nb`，但生成物为 float16/CPU fallback，尚未得到可证明由 VIP9000 NPU 执行的 INT8 模型。

---

## 1. 最终结论

### 1.1 已确认事实

1. `road_yolo11n_seg.onnx` 原始 FP32 模型可以在 ST Cloud 中 Optimize。
2. 全改写版 `road_yolo11n_seg_vsinpu_fp32_for_stcloud.onnx` 也可以 Optimize。
3. 单点改写变体均可以 Optimize：
   - 仅删除 `MaxPool.dilations`
   - 仅 `Split -> Slice`
   - 仅 `ConvTranspose -> SubpixelConv`
4. ST Cloud Quantize 后生成的 QDQ ONNX 确实包含量化节点，但 graph 输入/输出仍是 float32。
5. ST Cloud Quantize 生成的 QDQ ONNX 进入 STM32AI MPU Optimize 时会失败，错误为：
   - `Error while generating optimized file. Generation does not contain any output`
6. 本地生成的 QOperator 量化 ONNX 也不能 Optimize。
7. FP32 Optimize 生成的 `.nb` 可加载、可推理，但输入/输出为 float16，推理约 600ms，且 `strace` 未看到 `/dev/galcore` ioctl，说明 NPU 未实际执行。

### 1.2 当前状态判断

```text
FP32 ONNX -> ST Cloud Optimize -> .nb
  结果: 可运行，但 float16 / CPU fallback，不能算 NPU 加速

FP32 ONNX -> ST Cloud Quantize(QDQ) -> Optimize
  结果: Optimize 报 no output

本地 QOperator INT8 ONNX -> ST Cloud Optimize
  结果: Optimize 报 no output
```

因此，当前瓶颈不是某个单独算子改写，而是 **YOLO11-seg 量化图到 STM32MP2 NBG/.nb 的编译链路不通**。

### 1.3 2026-07-09 补充：road_fastseg_256 已可生成 `.nb`

新增测试模型：

```text
FlightController/Solutions/model/road_fastseg_256_fp32.onnx
FlightController/Solutions/model/road_fastseg_256_calibration_0_1_rgb.npz
FlightController/Solutions/model/road_fastseg_256_fp32_PerTensor_quant_road_fastseg_256_calibration_0_1_rgb_npz_1.onnx
```

该模型不是 YOLO11-seg，而是更简单的道路/背景语义分割模型：

```text
input:  3x256x256 float32
output: 2x256x256 float32
class 0: background
class 1: road
```

ST Cloud Quantize 后的 ONNX 仍显示 32-bit graph input/output，这是 QDQ
模型的正常表现。静态检查已确认内部存在大量量化节点：

```text
QuantizeLinear:   146
DequantizeLinear: 309
```

后续补充实验确认：`road_fastseg_256_fp32.onnx` 可以在 ST 云平台
Optimize / Generate 并生成 `.nb`：

```text
road_fastseg_256_fp32.onnx -> ST Cloud Optimize / Generate -> .nb
```

因此，旧结论中“YOLO11-seg 量化图到 STM32MP2 NBG/.nb 的编译链路不通”
不应扩展到 `road_fastseg_256`。当前应改为：

```text
YOLO11-seg 量化/改写路径仍然不适合作为当前主线。
road_fastseg_256 已进入 .nb 板端验收阶段。
```

这与两件已经验证的事实一致：

1. ST Cloud Quantize 确实可以生成内部 QDQ 量化 ONNX。
2. 官方 ST DeepLab 256x256 INT8 `.nb` 已在板端触发 `/dev/galcore`
   并达到约 51-52 ms raw run。

真正需要继续验证的是：

```text
road_fastseg_256 生成的 .nb 是否为 int8/uint8 或 static-affine 路径
road_fastseg_256 生成的 .nb 是否触发 /dev/galcore ioctl
road_fastseg_256 生成的 .nb 是否达到 <80 ms wrapped / <60 ms raw_run
road_fastseg_256 的 road mask 是否在真实道路图片上可用
```

当前不建议继续投入 YOLO11-seg 量化转换。下一步应把
`road_fastseg_256` 生成的 `.nb` 拷贝到板端，运行
`validate_nb_npu_contract.py`、`strace -yy` 和 overlay 可视化验收。

---

## 2. 模型与文件清单

### 2.1 可被 ST Cloud 识别和 Optimize 的 FP32 模型

| 文件 | 说明 | ST Cloud 结果 |
|---|---|---|
| `FlightController/Solutions/model/road_yolo11n_seg.onnx` | 原始 Ultralytics YOLO11n-seg FP32 | Optimize 成功 |
| `FlightController/Solutions/model/stcloud_upload/road_yolo11n_seg_vsinpu_fp32_for_stcloud.onnx` | 全改写版 FP32，去除 `ConvTranspose`/`Split` | Optimize 成功 |
| `FlightController/Solutions/model/stcloud_upload/variants/orig_drop_maxpool_dilations_only.onnx` | 仅删除 `MaxPool.dilations` | Optimize 成功 |
| `FlightController/Solutions/model/stcloud_upload/variants/orig_split_to_slice_only.onnx` | 仅 `Split -> Slice` | Optimize 成功 |
| `FlightController/Solutions/model/stcloud_upload/variants/orig_convtranspose_subpixel_only.onnx` | 仅 `ConvTranspose -> SubpixelConv` | Optimize 成功 |

### 2.2 不能用于 ST Cloud Optimize 的量化模型

| 文件 | 格式 | 结果 |
|---|---|---|
| `road_yolo11n_seg_int8_io.onnx` | graph input/output 为 INT8 | ST Cloud 不能识别输入/输出参数 |
| `road_yolo11n_seg_vsinpu_int8_qdq_upload.onnx` | ORT QDQ，float I/O | Optimize 报 no output |
| `road_yolo11n_seg_vsinpu_int8_qdq_int8out_upload.onnx` | float input + INT8 output | Optimize 报 no output |
| `road_yolo11n_seg_vsinpu_qoperator_s8_selected128.onnx` | QOperator (`QLinearConv` 等) | Optimize 报 no output |
| `road_yolo11n_seg_fp32_for_stcloud_PerTensor_quant_road_calib_selected_128_npz_2.onnx` | ST Cloud Quantize 输出的 QDQ 模型 | Optimize 报 no output |

### 2.3 ST Cloud Quantize 输出模型的实际结构

对 `road_yolo11n_seg_fp32_for_stcloud_PerTensor_quant_road_calib_selected_128_npz_2.onnx` 的静态检查结果：

```text
file size: 3.17 MB
inputs:
  images FLOAT [1, 3, 416, 416]
outputs:
  output0 FLOAT [1, 37, 3549]
  output1 FLOAT [1, 32, 104, 104]
QuantizeLinear: 365
DequantizeLinear: 571
ConvTranspose: 1
Split: 10
```

说明：

- 它确实被量化为 QDQ 图。
- ST 页面显示 `STAI_FORMAT_FLOAT` 是因为 graph 输入/输出仍是 float32，不代表内部没有量化。
- 如果源模型是原始 FP32，量化后仍保留 `ConvTranspose` 和 `Split`。

---

## 3. ONNX 模型要求与经验规则

### 3.1 ST Cloud 前端可识别的模型边界

ST Cloud 更容易识别以下形式：

```text
input:  FLOAT [1, 3, H, W]
output: FLOAT [...]
```

不建议上传 graph 输入/输出本身为 INT8 的 ONNX。实测 `*_int8_io.onnx` 在 ST Cloud UI 中显示：

```text
INPUT: -
OUTPUT: -
MODEL TYPE: -
```

即使该 ONNX 在本地 ONNX Runtime CPU provider 能跑，ST Cloud 前端也不一定接受。

### 3.2 QDQ 与 QOperator

已测试两种量化表达：

| 格式 | 典型节点 | ST Cloud Optimize |
|---|---|---|
| QDQ | `QuantizeLinear` / `DequantizeLinear` | 失败，no output |
| QOperator | `QLinearConv` / `QLinearMul` / `QLinearSigmoid` | 失败，no output |

结论：当前 ST Cloud 的 STM32AI MPU Optimize 对 YOLO11-seg 量化图支持不足，不能仅靠切换 QDQ/QOperator 解决。

### 3.3 FP32 Optimize 成功不等于 NPU 成功

ST Cloud 能生成 `.nb` 只说明生成链路完成，不代表 VIP9000 NPU 被使用。必须上板检查：

```text
输入/输出 dtype 是否为 int8/uint8 或 staticAffine
是否有 /dev/galcore ioctl
推理耗时是否显著低于 CPU
```

当前 FP32 `.nb` 的实测是：

```text
input_0  tensor(float16) [1,3,416,416]
output_0 tensor(float16) [1,37,3549]
output_1 tensor(float16) [1,32,104,104]
mean inference: ~611 ms
grep galcore: no output
```

这属于 float16/CPU fallback，不是可用 NPU 加速。

---

## 4. ST Cloud 使用经验

### 4.1 平台选择

应选择 STM32 MPU / STM32MP2 / X-LINUX-AI 相关平台，不要选 MCU/N6 流程。

目标应对应：

```text
STM32MP257 / STM32MP2
Runtime: ONNX Runtime / X-LINUX-AI
Accelerator: NPU / VIP9000 / GCNano
```

### 4.2 Quantize 页面

页面中的 `Apply post-training quantization` 就是 INT8 后训练量化入口。ST UI 不一定显式写 “INT8”。

关键选项：

```text
Disable per channel quantization: 勾选
```

含义：禁用 per-channel，使用 per-tensor 量化。当前 STM32MPU 页面提示：

```text
For better performance on STM32 MPU, we recommend a per-tensor quantization.
```

### 4.3 校准文件 `.npz`

原始 200 张校准包：

```text
key: images
shape: (200, 3, 416, 416)
dtype: float32
range: 0.0 ~ 1.0
compressed: ~76 MB
uncompressed: ~396 MB
```

ST Cloud 量化失败，但 terminal 无输出。原因推测为后端资源限制或一次性加载过大数组。

实测结果：

| 校准包 | 张数 | 结果 |
|---|---:|---|
| `road_calib_001_images.npz` | 1 | 可用于格式验证 |
| `road_calib_032_images.npz` | 32 | Quantize 成功 |
| `road_calib_selected_096.npz` | 96 | 推荐优先尝试 |
| `road_calib_selected_128.npz` | 128 | Quantize 成功 |
| `road_yolo11n_seg_calibration_images.npz` | 200 | Quantize failed |

建议不要直接上传 200 张大包；优先使用精选 96/128 张。

### 4.4 精选校准集

新增脚本：

```text
FlightController/tools/select_calibration_subset.py
```

用途：

- 从原始道路图片目录中选择代表性校准图。
- 支持图像统计特征：亮度、对比度、颜色偏移、边缘复杂度、时间分布、缩略图布局。
- 支持 `--use-model-features`，在有 ONNXRuntime 环境时追加 FP32 模型输出统计特征。

命令：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .\.venv_inspect\Scripts\Activate.ps1

python FlightController\tools\select_calibration_subset.py `
  --image-dir D:\drone2\adjustment\roads `
  --model FlightController\Solutions\model\stcloud_upload\road_yolo11n_seg_fp32_for_stcloud.onnx `
  --output-dir FlightController\Solutions\model\stcloud_upload\selected_model_features `
  --counts 64 96 128 `
  --use-model-features `
  --copy-images
```

精选标准：

- 暗光、正常、过曝
- 阴影、低对比、高对比
- 偏蓝、偏绿、偏红
- 边缘清晰/模糊
- 直路、转弯、分叉、路缘不清晰
- 避免连续帧高度重复

---

## 5. PC 侧工具与脚本变更

### 5.1 `quantize_yolo_gpu.py`

新增 `--force-int8-io` 实验选项，可把 QDQ 模型边界改为 INT8。

结论：该输出不适合 ST Cloud，前端不能识别参数，保留为实验记录，不推荐继续使用。

### 5.2 `select_calibration_subset.py`

新增校准集精选脚本。最初依赖 PIL，后改为 OpenCV，适配 `.venv_inspect` 环境。

`.venv_inspect` 依赖状态：

```text
onnx: 1.21.0
onnxruntime: 1.24.4
providers: DmlExecutionProvider, CPUExecutionProvider
cv2: 4.13.0
```

注意：该 venv 基于 Windows Store Python 3.13，在 Codex 受限沙箱中可能无法直接启动；使用普通终端或提升权限后可运行。

---

## 6. .nb 上板验证方法

### 6.1 烟雾测试

```bash
PYTHONPATH=. python3 FlightController/tools/test_nb_model.py \
  --model FlightController/Solutions/model/road_yolo11n_seg_vsinpu_fp32_opt.nb \
  --runs 10
```

当前实测：

```text
INPUTS:
  input_0 [1 x 3 x 416 x 416] tensor(float16)
OUTPUTS:
  output_0 [1 x 37 x 3549] tensor(float16)
  output_1 [1 x 32 x 104 x 104] tensor(float16)
mean: 611.78 ms
FPS: 1.6
```

### 6.2 `strace` 验证 NPU 是否实际执行

不需要另开终端，直接在同一个终端用 `strace` 包住测试命令。

推荐命令：

```bash
PYTHONPATH=. strace -f -e openat,ioctl python3 FlightController/tools/test_nb_model.py \
  --model FlightController/Solutions/model/road_yolo11n_seg_vsinpu_fp32_opt.nb \
  --runs 1 2>&1 | tee /tmp/nb_strace.log

grep -E 'galcore|/dev' /tmp/nb_strace.log
```

只看 galcore：

```bash
PYTHONPATH=. strace -f -e ioctl python3 FlightController/tools/test_nb_model.py \
  --model FlightController/Solutions/model/road_yolo11n_seg_vsinpu_fp32_opt.nb \
  --runs 2 2>&1 | grep galcore
```

注意：如果命令前没有 `PYTHONPATH=.`，可能报：

```text
ModuleNotFoundError: No module named 'nb_graph'
```

这是 Python 搜索路径问题，不是 NPU 问题。

### 6.3 判定标准

| 现象 | 结论 |
|---|---|
| 有大量 `/dev/galcore` open/ioctl | NPU 可能被调用 |
| `grep galcore` 无输出 | NPU 未被调用 |
| 输入/输出 float16 | 很可能是 CPU/fallback |
| 输入/输出 int8/uint8/staticAffine | 才可能是 NPU INT8 路径 |
| 推理仍约 600ms | CPU/fallback |
| 推理降到几十 ms 以内 | 才值得继续做精度验证 |

---

## 7. 当前建议路线

### 7.1 短期可运行方案：CPU/FP32 `.nb` 作为功能 fallback

使用 FP32 Optimize 生成的 `.nb` 只能作为功能烟雾测试或临时 fallback，不应作为 NPU 加速方案。

如果必须跑视觉链路，当前实际可用路线仍是：

```text
ONNX Runtime CPU/XNNPACK
```

或保持原 FP32/量化 ONNX 的 CPU 路径，接受 400-1800ms 级推理。

验收标准：

- 能稳定完成多次推理，不 segfault。
- 输出 shape 与 `road_perception.py` 解码逻辑一致。
- 明确标注为 CPU/fallback，不将其计入 NPU 性能目标。

### 7.2 方案 A：继续走 ST Cloud 官方 INT8 `.nb` 链路

目标：得到 ST Cloud / STM32AI MPU Tool 能成功 Optimize 的 INT8 `.nb`，并在板端触发 `/dev/galcore`。

建议动作：

1. 用最小复现包向 ST 提交问题：
   - FP32 原模型可 Optimize。
   - FP32 全改写模型可 Optimize。
   - ST Cloud 自己 Quantize 输出的 QDQ 模型再 Optimize 报 `Generation does not contain any output`。
   - 本地 QOperator 模型也报同类错误。
2. 复现包保留 3 个文件即可：
   - `road_yolo11n_seg_vsinpu_fp32_for_stcloud.onnx`
   - `road_calib_selected_032.npz` 或 `road_calib_selected_128.npz`
   - ST Cloud Quantize 生成的 `*_PerTensor_quant_*.onnx`
3. 询问 ST 明确支持矩阵：
   - STM32MP257 / VIP9000 是否支持 YOLO-seg 类双输出模型的 INT8 `.nb`。
   - STM32AI MPU Tool 对 QDQ 与 QOperator 哪种格式是推荐入口。
   - 是否要求固定 opset、固定 output name、固定 static shape 或特定 ONNX Runtime quantization 参数。

验收标准：

```text
ST Cloud Optimize 成功
.nb 输入/输出或内部元信息显示 int8/uint8/staticAffine 路径
板端 strace 可见 /dev/galcore open/ioctl
推理耗时明显低于 600ms，理想目标为几十 ms 以内
```

风险：

- 当前所有 YOLO11-seg 量化图均卡在 Optimize no output，可能是 ST 后端工具链 bug 或未覆盖该图结构。
- 即使 ST 修复 Optimize，仍需重新做量化精度验证。

### 7.3 方案 B：换成 ST Model Zoo 或官方已验证架构

目标：先跑通“真实 NPU 加速”的端到端路径，再决定是否回到 YOLO11-seg。

建议动作：

1. 优先选择 ST Model Zoo 中面向 STM32MP2/NPU 已验证的 segmentation 或 detection 模型。
2. 如果只需要道路区域，可考虑轻量语义分割模型替代 YOLO-seg：
   - MobileNet/DeepLab 类轻量分割。
   - ENet/Fast-SCNN 类道路分割。
   - ST 示例中已给出量化和 `.nb` 生成流程的模型优先。
3. 用同一套板端测试脚本验证：
   - `test_nb_model.py`
   - `strace -f -e openat,ioctl`
   - 实际摄像头输入下的端到端 FPS。

验收标准：

- 官方或示例模型 `.nb` 在板端出现 `/dev/galcore` ioctl。
- 推理速度达到可用范围。
- 再评估是否重训该架构做道路分割，而不是继续强行迁移 YOLO11-seg。

风险：

- 需要重新训练或转换数据集标签。
- 模型输出格式不同，`road_perception.py` 后处理需要适配。

### 7.4 方案 C：拆图，NPU 只跑卷积主体，CPU 做分割后处理

目标：绕开 YOLO11-seg 中最容易卡住编译链路的后处理/分割头复杂结构，让 NPU 只承担大部分卷积计算。

思路：

```text
camera image
  -> NPU 子图: backbone + neck + 部分 head 卷积
  -> CPU: decode boxes / mask coefficients / prototype combine / resize
```

建议动作：

1. 用 ONNX 图切分工具导出中间特征输出，优先切在纯 Conv/BN/SiLU/Concat 区域之后。
2. 对 NPU 子图单独做 INT8 量化和 ST Cloud Optimize。
3. 在 Python/C++ 中保留 YOLO-seg 的解码、NMS、mask decode。
4. 对比三种耗时：
   - 原始 CPU 全图。
   - FP32 `.nb` fallback。
   - INT8 NPU 子图 + CPU 后处理。

验收标准：

- 子图 `.nb` 能触发 `/dev/galcore`。
- 总耗时相比 CPU 全图有明显下降。
- 输出精度可通过同一张图片的 mask/box 对齐验证。

风险：

- 图切分点选择会影响后处理复杂度。
- 中间特征张量可能很大，NPU/CPU 间搬运成本可能吃掉加速收益。

### 7.5 方案 D：重新训练一个 NPU 友好的道路模型

目标：从模型结构上规避 `ConvTranspose`、复杂 `Split`、动态 shape、复杂双输出 decode。

建议结构要求：

| 项目 | 建议 |
|---|---|
| 输入 | 固定 NCHW，例如 `1x3x416x416` 或更小 |
| 输出 | 尽量单输出、固定 shape |
| 算子 | Conv / DepthwiseConv / Add / Mul / Relu / Sigmoid / Resize nearest 等基础算子 |
| 上采样 | 优先 `Resize nearest + Conv`，避免 `ConvTranspose` |
| 分割形式 | 优先语义分割 mask，少用 YOLO prototype + coefficients |
| 量化 | 训练后量化可接受时用 PTQ；精度不足再做 QAT |

候选路线：

1. 语义分割：直接输出道路/非道路 mask，后处理最简单。
2. 检测 + 简单几何：只检测可行驶区域关键点或边界线。
3. 小尺寸输入：先评估 `320x320` 或 `256x256`，降低 NPU 编译和运行压力。

验收标准：

- FP32 ONNX 可在 ST Cloud 识别。
- Quantize 后可 Optimize。
- `.nb` 板端有 `/dev/galcore` 调用。
- 真实道路图片上的 mask 可用，端到端延迟满足控制周期。

风险：

- 需要准备训练流程和精度评估脚本。
- 初期精度可能低于当前 YOLO11n-seg。

### 7.6 方案 E：板端 VSINPU Execution Provider 直接跑 INT8 ONNX

目标：不经过 `.nb` 文件，尝试用 OpenSTLinux 上的 ONNX Runtime `VSINPUExecutionProvider` 直接加载 NPU 友好 ONNX。

注意：原始 YOLO11-seg 已在 VSINPU EP 中暴露不支持算子和 fallback/segfault 问题，因此这个方案只适合更简单或已改写的子图/新模型。

建议动作：

1. 用 `onnxruntime.get_available_providers()` 确认存在 `VSINPUExecutionProvider`。
2. 对小模型或拆分子图测试：

```python
import onnxruntime as ort
sess = ort.InferenceSession(
    "candidate_int8.onnx",
    providers=["VSINPUExecutionProvider", "CPUExecutionProvider"],
)
print(sess.get_providers())
```

3. 同样使用 `strace -f -e openat,ioctl` 验证 `/dev/galcore`。

验收标准：

- 无 unsupported op fallback。
- 无 segfault。
- 有 `/dev/galcore` ioctl。
- 推理速度显著优于 CPU provider。

风险：

- VSINPU EP 对混合 CPU/NPU fallback 的稳定性已经出现过问题。
- 支持的 ONNX/量化格式可能比 ST Cloud `.nb` 路线更窄。

### 7.7 推荐推进顺序

优先级从高到低：

1. **先用官方/示例模型证明板端 NPU 真能跑**：得到一个有 `/dev/galcore` ioctl、速度明显下降的 `.nb`，建立验收基线。
2. **向 ST 提交 YOLO11-seg 量化 Optimize no output 最小复现**：确认是工具链限制还是模型参数问题。
3. **尝试 NPU 友好新模型或 ST Model Zoo 架构**：如果官方链路能跑通，这是最稳的工程路线。
4. **拆分当前 YOLO11-seg**：在保留现有精度资产的同时，把纯卷积主体交给 NPU。
5. **最后再尝试继续微调 QDQ/QOperator 细节**：当前证据显示收益概率较低。

### 7.8 不建议继续投入的路线

| 路线 | 原因 |
|---|---|
| graph 输入/输出改成 INT8 的 ONNX | ST Cloud 前端无法识别参数 |
| 继续调 QDQ 边界 | QDQ 和 QOperator 均已失败 |
| 只依赖 FP32 `.nb` | 上板验证显示未使用 NPU |
| 继续扩大校准集 | 校准集已经不是当前主阻塞点 |

---

## 8. 需要保留的关键证据

### 8.1 QDQ 模型静态检查

```text
QuantizeLinear / DequantizeLinear 存在
graph I/O 仍是 FLOAT
ST UI 显示 STAI_FORMAT_FLOAT 不等于没有量化
```

### 8.2 FP32 `.nb` 上板检查

```text
Loading dynamically: /usr/lib/libstai_mpu_ovx.so.6
[OVX]: Loading nbg model
input/output dtype: float16
mean inference: ~611 ms
grep galcore: no output
```

### 8.3 核心阻塞表述

```text
ST Cloud 可以处理 FP32 YOLO11-seg 并生成 .nb，
但该 .nb 未实际使用 VIP9000 NPU。

ST Cloud 可以量化模型生成 QDQ ONNX，
但该 QDQ ONNX 无法进入 STM32AI MPU Optimize，报 no output。

因此当前缺口是:
YOLO11-seg INT8 quantized graph -> STM32MP2 NPU executable NBG/.nb
```
