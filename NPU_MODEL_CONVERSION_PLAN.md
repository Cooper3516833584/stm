# NPU 模型转换方案：YOLO11-seg → VIP9000 兼容 ONNX

**日期**: 2026-06-11
**状态**: NPU 驱动就绪，但 YOLO11n-seg 含不兼容算子（`ConvTranspose`、dilated `NonMaxPool`），需转换后才能在 VSINPU EP 上运行
**目标**: 将 `road_yolo11n_seg.onnx` 转换为 VIP9000 NPU 可执行的 ONNX，推理从 ~2600ms(CPU) 降至 5-15ms(NPU)

---

## 0. 2026-07-09 最新实测结论

> 详细记录见 [NPU_ST_CLOUD_20260709_FINDINGS.md](NPU_ST_CLOUD_20260709_FINDINGS.md)。本节用于修正本文早期的乐观假设。

1. `road_yolo11n_seg.onnx` 原始 FP32 模型可以在 ST Cloud 中 Optimize；全改写版 `road_yolo11n_seg_vsinpu_fp32_for_stcloud.onnx` 也可以 Optimize。
2. `ConvTranspose`、`Split`、`MaxPool.dilations` 的单点改写变体均已验证可 Optimize，说明当前 ST Cloud Optimize 阶段并不是被某一个单独 FP32 算子阻塞。
3. ST Cloud Quantize 生成的是 QDQ 量化图，graph 输入/输出仍显示为 float32，这是正常现象，不代表没有量化；静态检查可见大量 `QuantizeLinear`/`DequantizeLinear` 节点。
4. 但 ST Cloud QDQ、本地 QDQ、强制 int8 I/O、本地 QOperator 等量化模型进入 Optimize 时均失败，典型报错为 `Generation does not contain any output`。
5. FP32 Optimize 生成的 `.nb` 可加载和推理，但输入/输出为 float16，板端实测约 600ms，`strace` 未看到 `/dev/galcore` ioctl，因此不能视为 VIP9000 NPU 加速。
6. 当前真正未解决的问题是：**如何得到 ST Cloud/STM32AI MPU 能编译为实际 NPU 执行路径的 INT8 `.nb`**。FP32 `.nb` 只能作为功能链路探活产物，不能作为性能目标。
7. 后续可行方案已记录在 [NPU_ST_CLOUD_20260709_FINDINGS.md](NPU_ST_CLOUD_20260709_FINDINGS.md) 第 7 节，包括 ST 官方链路、官方模型/换架构、拆图、重训 NPU 友好模型和板端 VSINPU EP 直跑等分支。

---

## 1. 问题根因

### 1.1 NPU 硬件栈已就绪

| 组件 | 版本 | 状态 |
|------|------|:--:|
| galcore 内核驱动 | 6.4.19.4 | ✅ `/dev/galcore` |
| VIP9000 NPU | model=0x8000 rev=6205, 800MHz | ✅ |
| gcnano-userland | 6.4.19 | ✅ |
| libopenvx-gcnano | 6.4.19 | ✅ |
| onnxruntime (VSINPU) | 1.19.2 | ✅ |
| X-LINUX-AI benchmark | 6.0.1 | ✅ |

### 1.2 但模型不被 NPU 支持

官方 benchmark 输出：

```
road_yolo11n_seg.onnx failed model not supported with VSINPU execution provider, retrying on CPU...
```

VSINPU EP 日志：

```
NonMaxPool with Dilation parameter is not supported
Uneven splits are not currently supported
Fallback unsupported op ConvTranspose to cpu   ← 回退 CPU 时数据传输 segfault
VSINPU supported: 348/353 nodes
```

**根本原因**：YOLO11-seg 使用 `ConvTranspose` + dilated `NonMaxPool` 做分割头的 mask prototype 上采样，这两个算子 VIP9000 NPU 不加速且混合执行会崩溃。

### 1.3 NPU 算子限制

VIP9000 不支持（来自 ST 文档 + 实测）：

| 算子 | YOLO11-seg 用途 | NPU 行为 |
|------|------|------|
| `ConvTranspose` | 分割头 decoder 上采样 | 回退 CPU → 混合执行 segfault |
| `NonMaxPool` (dilation>1) | 边界框 NMS-like pooling | 回退 CPU → 错误 |
| `Split` (uneven) | 通道分割 | 不支持 |
| `Resize` (某些模式) | 上采样 | 部分支持，取决于 mode |
| `Slice` (动态) | 动态切分 | 不支持 |

---

## 2. 转换后模型硬性要求（验收 Checklist）

以下所有条目转换后**必须满足**，否则 `road_perception.py` 无法正常工作。

### 2.1 算子兼容性

**转换后模型不能含有**以下 VIP9000 不支持的算子（含这些算子会导致 segfault）：

| 禁止的算子 | YOLO11-seg 用途 | 等效替代方案 |
|------|------|------|
| `ConvTranspose` | 分割头 decoder 上采样 | **`Resize` + 标准 `Conv`**：先用 `Resize` 上采样，再标准卷积 |
| `NonMaxPool` (dilation ≠ 1) | 边界框 NMS-like pooling | **`MaxPool` (dilation=1)** + 等效 padding |
| `Split` (uneven) | 非均匀通道切分 | 多个等大小的 **`Slice`** 或等效 `Split` (even) |

**验收命令**（在开发板上执行）：
```bash
# 模型加载时不应出现 "Fallback unsupported op" 警告或 segfault
python3 -c "
import onnxruntime as ort
m = ort.InferenceSession('road_yolo11n_seg_npu.onnx', providers=['VSINPUExecutionProvider'])
print('Provider:', m.get_providers()[0])
print('OK — no crash, no fallback')
"
```

### 2.2 输入格式不变

转换后的模型必须接受与原模型完全相同的输入：

| 属性 | 值 | 不可变 |
|------|-----|:--:|
| 名称 | `images` | ✅ |
| shape | `[1, 3, 320, 320]` | ✅ |
| dtype | `float32` | ✅ |
| 布局 | NCHW | ✅ |
| 值域 | `0.0 ~ 1.0`（`_preprocess()` 做 `/255.0` 归一化） | ✅ |
| 预处理 | `_letterbox()` 等比缩放 + 居中 pad → `_preprocess()` 归一化 + transpose | ✅ |

**验收命令**：
```bash
python3 -c "
import onnxruntime as ort
m = ort.InferenceSession('road_yolo11n_seg_npu.onnx', providers=['VSINPUExecutionProvider'])
inp = m.get_inputs()[0]
assert inp.name == 'images', f'input name mismatch: {inp.name}'
assert inp.shape == [1, 3, 320, 320], f'input shape mismatch: {inp.shape}'
print(f'Input OK: {inp.name} {inp.shape} {inp.type}')
"
```

### 2.3 输出格式不变

转换后的模型必须保持两份输出，格式与原模型一致。`_decode_yolo_segmentation()` 依赖此结构，格式变了会直接炸：

```
output[0]: [1, C, N]  检测头 — 每列含义: [cx, cy, w, h, cls_score(1), mask_coeffs(M)]
output[1]: [1, M, H, W]  分割头 — mask prototypes
```

其中：
- `C = 4 + 1 + M`（bbox 位置 + 置信度 + mask 系数）
- `N` = 检测候选数（动态，取决于输入）
- `M` = mask prototype 通道数（YOLO11n-seg 为 32）
- `H × W` = mask prototype 空间尺寸（通常 160×160）

**操作指引**：在 ST Cloud 中必须选择 **"ONNX Runtime (X-LINUX-AI)"** 运行时。不要选 "ST AI Runtime"——后者可能重构输出格式导致解码代码失败。

**验收命令**：
```bash
python3 -c "
import onnxruntime as ort
m = ort.InferenceSession('road_yolo11n_seg_npu.onnx', providers=['VSINPUExecutionProvider'])
outs = m.get_outputs()
assert len(outs) == 2, f'output count mismatch: {len(outs)}'
o0, o1 = outs[0], outs[1]
assert len(o0.shape) == 3 and o0.shape[0] == 1, f'output0 shape mismatch: {o0.shape}'
assert len(o1.shape) == 3 and o1.shape[0] == 1, f'output1 shape mismatch: {o1.shape}'
print(f'Output0 OK: {o0.name} {o0.shape}')
print(f'Output1 OK: {o1.name} {o1.shape}')
"
```

### 2.4 预处理管线不变

`_preprocess()` 的调用链固定为：

```
原始 BGR frame (uint8, H×W×3)
  → _normalize_frame()          确认 uint8 contiguous
  → _preprocess(frame, 320)
       → _letterbox()           等比缩放 + pad 到 320×320, 保留 scale/pad 参数
       → img / 255.0            float32 归一化到 0-1
       → transpose(2,0,1)       HWC → CHW
       → expand_dims(0)         → NCHW [1,3,320,320]
  → session.run(None, {input_name: blob})
  → _decode_yolo_segmentation()  解码 bbox + mask → 道路中线
```

不需要预归一化（mean/std）、不需要 BGR→RGB 转换。**如果转换工具提示需要归一化参数，全部设为单位值**（mean=[0,0,0], std=[1,1,1]）。

### 2.5 量化建议

| 参数 | 推荐值 | 说明 |
|------|------|------|
| 精度 | 先 **FP32** 验证推理正确，再试 INT8 | FP32 可用于全链路 dry-run |
| INT8 量化方式 | **per-tensor**（非 per-channel） | 勾选 "Disable per channel quantization" |
| 校准数据 | 道路场景图片 ≥50 张 | 不提供则用随机数据，量化后实测精度会大幅下降 |
| 校准图片是否需要标签 | **存疑，待验证** | 待确认 ST Cloud 是否需要标注后的图片，或仅需原始帧 |

---

## 3. 当前模型规格（转换前）

| 属性 | 值 |
|------|-----|
| 文件 | `FlightController/Solutions/model/road_yolo11n_seg.onnx` |
| 大小 | 11.5 MB |
| 架构 | YOLO11n-seg (Ultralytics 导出) |
| 输入 | `images`: [1, 3, 320, 320] float32, NCHW, 归一化 0-1 |
| 输出 0 | [1, C, N] — 检测头: bbox(4) + cls(1) + mask_coeffs(M) |
| 输出 1 | [1, M, 160, 160] — mask prototypes |
| op set | YOLO11 默认 (≥11) |
| 代码管线 | `_preprocess()` → `session.run()` → `_decode_yolo_segmentation()` |
| 不兼容算子 | `ConvTranspose`, `NonMaxPool(dilation)`, `Split(uneven)` |

---

## 4. 转换方案

### 4.1 ST Edge AI Developer Cloud（推荐 ⭐）

ST 官方云端工具，自动分析和修复 NPU 兼容性问题并生成硬件可执行文件。

**URL**: https://stedgeai-dc.st.com

#### Step 1: 首页选择硬件平台
1. 浏览器打开并登录 https://stedgeai-dc.st.com。
2. 在平台选择首页中，找到 **"STM32 MPUs"** 卡片（描述为 "Start with STM32 Microprocessors embedding Cortex-A loaded with X-LINUX-AI"）。
3. 点击卡片下方的黄色按钮 **"Select with another version (ST Edge AI Core 2.2.0)"**，系统将自动切换至兼容 MPU 工具链的核心版本工作区。

#### Step 2: 上传模型与配置 Target
1. 进入工作区后，点击 **"Create Project"** 或 **"Upload Model"**。
2. 上传原始模型文件 `road_yolo11n_seg.onnx`。
3. 在项目配置或 Target 设置中，选择以下目标参数：
   - **Board (开发板)**: `STM32MP257F-EV1`（或 `STM32MP2` 系列）
   - **Runtime (运行时)**: `ONNX Runtime (X-LINUX-AI)`
   - **Hardware accelerator (硬件加速器)**: `NPU (VIP9000 / GCNano)`

#### Step 3: 模型量化 (Quantize) 与校准数据上传
1. 页面将自动引导至 **Model quantization** 面板。
2. **配置量化参数**：勾选 **"Disable per channel quantization"** 下方的复选框（禁用逐通道量化，采用 per-tensor 量化以获得更适合 VIP9000 NPU 的加速性能）。
3. **准备校准数据集**：
   - 必须使用包含了代表性道路场景图片的 `.npz` (NumPy 压缩包) 文件。若不提供，系统将使用随机数据校准，这会导致量化后的模型在实际运行时精度大幅下降。
   - **校准图片是否需要标签暂存疑**——待实际使用 ST Cloud 后确认。
   - 在 **"Load file (.npz)"** 处点击回形针图标，上传按照说明打包好的 `calibration_data.npz` 文件。
4. 点击右下角的 **"Launch quantization"** 启动量化。

#### Step 4: 分析兼容性与优化替换 (Optimize)
1. 量化完成后进入 **Optimize** 阶段，工具链会执行专门针对 STM32 MPUs 的优化器，将不支持的算子（如 `ConvTranspose`、带 Dilation 的 `NonMaxPool`）自动尝试替换为等效的兼容算子组（例如将 `ConvTranspose` 转换为 `Resize + Conv`），并生成专用于 STM32MP2x 硬件加速的网络二进制图（.nb）。
2. 对照 **[§2 验收 Checklist](#2-转换后模型硬性要求验收-checklist)** 逐项检查分析报告，确认：
   - 不兼容算子已被成功替换
   - 输入 shape/dtype 不变
   - 输出数量、shape、顺序不变

#### Step 5: 性能评估与导出 (Benchmark & Generate)
1. 在 **Benchmark** 步骤中，选择量化优化后的网络模型，在 ST 托管的云端开发板集群上远程运行基准测试，获取准确的推理时间和内存占用。
2. 确认性能达标后，进入 **Generate** 步骤。
3. 选择 **"Download Optimized Network Binary"** 下载生成的网络二进制图（.nb），或在模型下拉菜单中选择并下载量化转换后的标准 ONNX 模型（`road_yolo11n_seg_npu.onnx`）。

---

### 4.2 Ultralytics 导出兼容 ONNX（备选）

如果 ST Cloud 无法自动修复，在 PC 上用 Ultralytics 重新导出，排除不兼容算子：

```bash
pip install ultralytics onnx

python -c "
from ultralytics import YOLO

model = YOLO('yolo11n-seg.pt')  # 原始 PyTorch 权重

# 导出为 NPU 友好 ONNX:
#   - opset=14  (VIP9000 推荐)
#   - simplify=True  (移除冗余节点)
#   - dynamic=False  (固定 batch=1)
#   - imgsz=320
model.export(
    format='onnx',
    opset=14,
    simplify=True,
    dynamic=False,
    imgsz=320,
    batch=1,
    half=False,           # FP32, 量化在 ST Cloud 做
    nms=False,            # YOLO11 默认输出格式
)
"
```

导出后再回到 ST Cloud 做分析和 INT8 量化。转换后仍需对照 **[§2 验收 Checklist](#2-转换后模型硬性要求验收-checklist)** 逐项验证。

---

## 5. 代码侧修改

### 4.1 模型路径配置

转换完成后，在 `road_perception.py` 中新增 NPU 模型路径：

```python
# road_perception.py L144 附近
MODEL_PATH = "FlightController/Solutions/model/road_yolo11n_seg.onnx"
MODEL_PATH_NPU = "FlightController/Solutions/model/road_yolo11n_seg_npu.onnx"  # 新增
```

### 4.2 `_select_providers()` 已修复 ✅

当前 `road_perception.py:305` 已修正为 `VSINPUExecutionProvider`（全大写），无需再改。

### 4.3 `_resolve_model_path()` 适配

如果 NPU 模型文件名不同于原模型，需要新增 CLI 参数：

```python
# road_follow_main.py 新增
parser.add_argument("--model", default=MODEL_PATH,
                    help="ONNX model path (default: CPU model)")
parser.add_argument("--model-npu", default=MODEL_PATH_NPU,
                    help="NPU-optimized ONNX model path")
```

但最简单的是：**直接用 NPU 模型覆盖原路径**，代码零改动。

---

## 6. 校准数据集准备（INT8 量化用）

INT8 量化需要 50-200 张代表性输入图片。从道路场景采集：

### 6.1 采集方式

在开发板上用摄像头拍一批道路图片：

```bash
source /usr/local/UFC_venv/bin/activate

python3 -c "
import cv2, os, time
os.makedirs('/usr/local/calibration_images', exist_ok=True)
cap = cv2.VideoCapture(7, cv2.CAP_V4L2)  # 道路摄像头
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

for i in range(100):
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f'/usr/local/calibration_images/road_{i:04d}.jpg', frame)
        print(f'Captured {i+1}/100')
    time.sleep(0.5)
cap.release()
"
```

然后把 `/usr/local/calibration_images/` 下载到 PC，上传到 ST Cloud 作为校准数据集。

### 6.2 备选：用 benchmark 脚本的 camera 帧

如果摄像头已就绪，`bench_vision_fps.py` 已经能读取摄像头帧，稍加修改即可保存。

---

## 7. 验证序列

转换完成后，在开发板上逐级验证：

### 阶段 1：模型加载

```bash
source /usr/local/UFC_venv/bin/activate
python3 -c "
import onnxruntime as ort
m = ort.InferenceSession('FlightController/Solutions/model/road_yolo11n_seg_npu.onnx',
                          providers=['VSINPUExecutionProvider'])
print('Provider:', m.get_providers()[0])
print('Input:', m.get_inputs()[0].name, m.get_inputs()[0].shape)
print('Outputs:', [(o.name, o.shape) for o in m.get_outputs()])
"
```

**预期**: `Provider: VSINPUExecutionProvider`，无 segfault，无 fallback 警告。

### 阶段 2：单帧推理正确性

```bash
PYTHONPATH=. python3 -c "
import cv2, numpy as np, time
from road_perception import get_road_perception

# 用真实道路图或 calibration image 测试
frame = cv2.imread('/usr/local/calibration_images/road_0000.jpg')
if frame is None:
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

t0 = time.perf_counter()
result = get_road_perception(frame)
elapsed = (time.perf_counter() - t0) * 1000
print(f'Inference: {elapsed:.1f}ms  road_found={result.is_road_found}  error={result.pixel_error:.1f}px')
"
```

**预期**: 推理 < 50ms（首次加载可能 ~200ms 由于 NPU 初始化），road_found 取决于输入图片。

### 阶段 3：FPS 基准

```bash
PYTHONPATH=. python FlightController/tools/bench_vision_fps.py --frames 30
```

**预期**: p50 < 15ms, FPS > 60。

### 阶段 4：全链路 dry-run

```bash
PYTHONPATH=. python road_follow_main.py --no-fc --dry-run --loop-hz 10 --camera-index 7
```

---

## 8. 回退方案

如果 ST Cloud 转换仍无法让 NPU 支持 YOLO11-seg：

| 方案 | 操作 | 预期推理 |
|------|------|:--:|
| **A: XNNPACK** | 注释 `_select_providers()` 中的 VSINPU 行 → fallback XNNPACK | ~600-900ms |
| **B: YOLOv8n-seg** | 换 YOLOv8n-seg（算子更简单，ST 可能有已验证模型） | 5-15ms (NPU) |
| **C: ST 官方模型** | 用 X-LINUX-AI 模型 zoo 中的语义分割模型替换 | 5-15ms (NPU) |

---

## 9. 时间线预估

| 阶段 | 耗时 | 说明 |
|------|:--:|------|
| 采集校准图片 | 15 min | 100 张道路场景 |
| 上传 ST Cloud + 分析 | 10 min | 含等待 |
| 模型转换 + 下载 | 15 min | 含 INT8 量化 |
| 部署 + 验证 | 30 min | 加载 → 单帧 → FPS → 全链路 |
| **总计** | **~1h** | 含调试预留 |
