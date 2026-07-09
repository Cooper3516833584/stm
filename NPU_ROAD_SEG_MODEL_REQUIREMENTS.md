# STM32MP257 NPU Road Segmentation Model Requirements

**Target project**: ObstacleAvoidanceDrone / road-following perception  
**Target board**: MYD-LD25X / STM32MP257 / OpenSTLinux v6.0  
**Target accelerator**: VIP9000 / GCNano NPU through STAI MPU + OpenVX + `.nb`  
**Dataset source**: CVAT annotated road images  
**Goal**: train a road/background semantic segmentation model that can replace YOLO11-seg for road-following on the STM32MP257 NPU.

---

## 1. Current Status

The NPU software path has been proven usable:

- `stai_mpu` imports successfully.
- `/usr/lib/libstai_mpu_ovx.so.6` exists.
- `.nb` model loading prints `[OVX]: Loading nbg model`.
- `strace -yy` confirms `/dev/galcore` open/ioctl calls.
- ST DeepLab v3 256x256 INT8 `.nb` reaches about:

```text
wrapped latency mean: about 66-79 ms
raw_run_ms mean: about 51-52 ms
input:  [1, 3, 256, 256] tensor(int8)
output: [1, 2, 256, 256] tensor(int8)
```

However, the official ST DeepLab model predicts all pixels as class 0 on our road images, so it is only a performance baseline. It is not a usable road-following perception model.

Therefore the next model must be trained or fine-tuned on the CVAT road dataset.

Important clarification, updated after the latest `road_fastseg_256` ST Cloud test:

The earlier ST Cloud failure records were first observed on the YOLO11-seg
conversion path:

```text
YOLO11-seg FP32
-> ST Cloud Quantize / local QDQ / QOperator
-> STM32MP2 Optimize
-> failed with "Generation does not contain any output"
```

At that point it was reasonable to treat `road_fastseg_256_fp32.onnx` as a new
NPU-friendly candidate, because it removed the YOLO segmentation head, mask
prototypes, split/decode patterns, and multi-output postprocessing.

The later `road_fastseg_256` tests clarified two different cloud outcomes:

```text
road_fastseg_256_fp32.onnx
-> ST Cloud Optimize / Generate
-> .nb generated successfully, but board metadata is tensor(float16)

road_fastseg_256_fp32_PerTensor_quant_road_fastseg_256_calibration_0_1_rgb_npz_1.onnx
-> ST Cloud Optimize / Generate
-> no output / no usable .nb
```

Therefore the earlier pessimistic conclusion must be narrowed back to the old
YOLO/QDQ paths. The current status is:

```text
YOLO11-seg quantized paths: still blocked or unsuitable for NPU acceptance
road_fastseg_256 FP32 path: .nb generation is possible but produces float16
road_fastseg_256 QDQ path: internally quantized ONNX exists, but .nb generation has no output
```

The project should stop repeating cloud-side conversion experiments with the
same artifacts and settings. A new attempt should only start after changing the
ST tool flow, deployment recipe, or model export.

Additional board result:

```text
road_fastseg_256_fp32_1.nb
input/output: tensor(float16)
wrapped mean latency: about 167 ms
```

This means the `.nb` generated from the FP32 ONNX is not the desired INT8 NPU
artifact. For the next `.nb` attempt, use the QDQ quantized ONNX as the source
artifact only if the ST flow or target settings have changed; the current QDQ
attempt has already produced no output.

---

## 1.1 Current Candidate: road_fastseg_256

Current candidate files:

```text
ONNX:
FlightController/Solutions/model/road_fastseg_256_fp32.onnx

Calibration:
FlightController/Solutions/model/road_fastseg_256_calibration_0_1_rgb.npz

Metadata:
FlightController/Solutions/model/road_fastseg_256_metadata.json

Metrics:
FlightController/Solutions/model/road_fastseg_256_test_metrics.json
```

Model metadata:

```text
size: 256
num_classes: 2
class_order: ["background", "road"]
input_layout: NCHW
onnx_input_range: RGB float32 0.0-1.0
mean/std normalization: embedded in ONNX
canonical_onnx: road_fastseg_256_fp32.onnx
```

Test metrics:

```text
road_iou:        0.5841
mean_iou:        0.6924
bottom_road_iou: 0.6344
pixel_accuracy:  0.8442
```

Calibration file inspection:

```text
file size: about 100,663,562 bytes / 96 MiB
internal entry: images.npy
shape: (128, 3, 256, 256)
dtype: float32
range: 0.0 to 1.0
layout: NCHW
image count: 128
```

Initial comparison against this requirements document:

| Requirement | road_fastseg_256 status |
|---|---|
| Road/background semantic segmentation | Meets |
| Class order background=0, road=1 | Meets |
| Input `[1,3,256,256]` | Meets |
| Output `[1,2,256,256]` | Expected from training/export script |
| ONNX input float32 | Meets |
| RGB 0.0-1.0 input | Meets |
| Mean/std embedded in ONNX | Meets |
| Static batch/shape | Expected from export script |
| Calibration count 64-128 | Meets, 128 |
| Calibration dtype/range/layout | Meets |
| INT8 `.nb` | Not available |
| FP32/f16 `.nb` | Available as `road_fastseg_256_fp32_1.nb`, but rejected |
| `/dev/galcore` board proof | Not applicable until an INT8/static-affine `.nb` exists |

The ST Cloud quantization page showing:

```text
MODEL TYPE: STAI_FORMAT_FLOAT
INPUT:  3x256x256 (32 bits)
OUTPUT: 2x256x256 (32 bits)
```

is normal before post-training quantization. It describes the uploaded FP32
ONNX. Even after QDQ quantization, graph input/output may still display as
float because QDQ keeps float graph boundaries. The real NPU acceptance artifact
is the generated `.nb`, which must expose int8/uint8 metadata on the board.

Concrete ST Cloud quantized output observed:

```text
FlightController/Solutions/model/road_fastseg_256_fp32_PerTensor_quant_road_fastseg_256_calibration_0_1_rgb_npz_1.onnx
```

ST Cloud still reports:

```text
INPUT:  3x256x256 (32 bits)
OUTPUT: 2x256x256 (32 bits)
MODEL TYPE: STAI_FORMAT_FLOAT
```

This is expected for an ONNX Runtime QDQ model and does **not** mean the graph
was not quantized. Static inspection shows:

```text
QuantizeLinear:   146
DequantizeLinear: 309
FP32 source model QuantizeLinear/DequantizeLinear: 0 / 0
```

Therefore this downloaded ONNX is an internally quantized QDQ model with
float32 graph boundaries. It should not be rewritten to force graph I/O to
INT8; that has already been shown to make ST Cloud model recognition worse.

Latest result:

```text
road_fastseg_256_fp32.onnx
-> ST Cloud Optimize / Generate
-> .nb generated successfully
```

Status:

```text
FP32 ONNX: valid candidate for training/accuracy and CPU fallback tests
QDQ ONNX: proof that ST Cloud quantization can run internally
.nb: FP32 path generated float16 fallback-style artifact; QDQ path generated no usable .nb
```

The next step is not another cloud conversion attempt. The next useful actions
are:

1. Copy the generated `.nb` into `FlightController/Solutions/model/`.
2. Do not run overlay/integration on `road_fastseg_256_fp32_1.nb`; it already
   failed the tensor contract.
3. Do not repeat QDQ -> Generate with the same cloud settings; it already
   produced no output.
4. Preserve the FP32 `.nb`, QDQ ONNX, validation log, and cloud failure as
   evidence for the next toolchain investigation.

Fast preflight before generating `.nb`:

```bash
PYTHONPATH=. python3 FlightController/tools/inspect_onnx_npu_candidate.py \
  FlightController/Solutions/model/road_fastseg_256_fp32.onnx

PYTHONPATH=. python3 FlightController/tools/inspect_onnx_npu_candidate.py \
  FlightController/Solutions/model/road_fastseg_256_fp32_PerTensor_quant_road_fastseg_256_calibration_0_1_rgb_npz_1.onnx
```

Expected interpretation:

```text
FP32 ONNX: no QuantizeLinear/DequantizeLinear -> likely f16 .nb
QDQ ONNX: QuantizeLinear/DequantizeLinear present -> structurally the right
          input for INT8, but current ST Cloud Generate still produced no output
```

---

## 2. Task Definition

Use semantic segmentation, not YOLO instance segmentation.

Required task:

```text
class 0: background / non-road
class 1: road / drivable path
```

Required output:

```text
semantic mask -> clean mask -> centerline extraction -> pixel error / angle -> RoadFollower
```

The model does not need to output boxes, object confidence, mask coefficients, prototypes, or NMS results.

---

## 3. Dataset Requirements

### 3.1 CVAT Export

Recommended CVAT export formats:

1. **Segmentation mask format** if available.
2. **COCO instance/segmentation** if masks need to be rasterized.
3. **CVAT XML/JSON polygons** if conversion script will rasterize polygons.

Required dataset contents:

```text
images/
  *.jpg or *.png
masks/
  *.png
```

Each mask should be single-channel:

```text
0 = background
1 = road
```

If CVAT exports 255 for road, convert it before training or inside the loader:

```text
mask = (mask > 0).astype(uint8)
```

### 3.2 Dataset Split

Recommended split:

```text
train: 70-80%
val:   10-20%
test:  10%
```

Do not split highly similar consecutive video frames across train and val/test. Keep adjacent frames in the same split to avoid optimistic validation results.

### 3.3 Required Scene Coverage

The dataset should cover:

- Straight road.
- Left and right turns.
- Forks or intersections if the mission needs branch decisions.
- Bright sunlight, shadow, low contrast, and overexposure.
- Cyan/green color cast from the current road camera.
- Road edges partially missing.
- Grass, soil, gravel, concrete, and background texture similar to road.
- Motion blur.
- Camera pitch/height variation if expected in flight.

### 3.4 Labeling Rules

Label only the drivable road/path area as road.

Recommended rules:

- Include the full visible drivable surface.
- Exclude grass, walls, trees, obstacles, and sky.
- Exclude road-like background if it is not part of the drivable path.
- At forks, label all visible drivable branches if they are valid paths.
- If only the currently intended branch should be followed, label only that branch consistently.

The decision above affects `road_perception.py` behavior:

- Label all branches -> branch detection remains possible.
- Label only selected path -> simpler centerline but weaker fork awareness.

---

## 4. Input/Output Contract

### 4.1 Preferred Input

Primary target:

```text
input name: any stable name is acceptable, but `input_0` or `images` is preferred
shape: [1, 3, 256, 256]
dtype in ONNX: float32
dtype in .nb: int8 or uint8
layout: NCHW
color: RGB preferred, but must be documented
range before quantization: 0.0-1.0 preferred
```

Alternative if 256x256 accuracy is too weak:

```text
shape: [1, 3, 320, 320]
```

Avoid 416x416 unless absolutely necessary. The official 416 DeepLab baseline is too slow for the current road-following budget.

### 4.2 Preferred Output

Primary target:

```text
shape: [1, 2, 256, 256]
dtype in ONNX: float32
dtype in .nb: int8 or uint8
layout: NCHW
channel 0: background
channel 1: road
```

Runtime decoding:

```python
class_map = argmax(output[0], axis=0)
road_mask = (class_map == 1)
```

If the model outputs `[1, 1, H, W]`, use sigmoid thresholding instead:

```python
road_mask = sigmoid(output[0, 0]) > threshold
```

But `[1, 2, H, W]` is preferred because it matches the proven ST DeepLab baseline.

---

## 5. Architecture Requirements

Prefer a lightweight semantic segmentation model.

Recommended architectures:

- ST DeepLab v3 / MobileNetV2 backbone, if retraining is convenient.
- Fast-SCNN style lightweight segmentation.
- MobileNetV2/3 encoder + simple decoder.
- Small U-Net only if exported ops stay NPU-compatible.

Avoid models with:

- YOLO detection/segmentation heads.
- Mask prototypes + mask coefficients.
- NMS or NonMaxSuppression inside the graph.
- Dynamic shape.
- Dynamic Slice/Gather/Shape-heavy postprocessing.
- `ConvTranspose`.
- Large multi-output decode graphs.

Upsampling requirements:

```text
preferred: Resize nearest or bilinear + Conv
avoid: ConvTranspose
```

Activation requirements:

```text
preferred: ReLU / ReLU6 / simple Add / Mul / Sigmoid
use cautiously: SiLU / Swish if ST compiler handles it
```

Model should remain fully static:

```text
batch = 1
height = fixed
width = fixed
no dynamic axes in ONNX export
```

---

## 6. Training Requirements

### 6.1 Training Resolution

Train at the same resolution intended for deployment:

```text
primary: 256x256
fallback: 320x320
```

Use the same resize policy during training and deployment:

- If deployment uses direct stretch, train with direct stretch.
- If deployment uses letterbox, train with letterbox.

For road-following, direct stretch may be acceptable and simpler for semantic segmentation, but it must be validated visually.

### 6.2 Recommended Augmentation

Use moderate augmentations:

- Brightness and contrast.
- Hue/saturation jitter.
- Motion blur.
- Gaussian noise.
- Slight rotation and perspective.
- Random shadow.

Do not overuse crop augmentation that removes the lower road region; the control logic relies heavily on the bottom half of the image.

### 6.3 Loss and Metrics

Recommended loss:

```text
CrossEntropyLoss + DiceLoss
```

Useful metrics:

```text
road IoU
mean IoU
pixel accuracy
bottom-region road IoU
centerline error on validation images
```

The most important project metric is not only IoU. It is whether the extracted centerline gives a stable `pixel_error` and `centerline_angle`.

---

## 7. ONNX Export Requirements

Export ONNX with:

```text
opset: 13 or 14 preferred
batch: fixed 1
dynamic axes: disabled
input dtype: float32
output dtype: float32
input layout: NCHW
```

Required ONNX validation:

```bash
python -m onnx.checker model.onnx
```

Recommended local inspection:

```text
No ConvTranspose
No NonMaxSuppression
No dynamic Resize scales derived from runtime Shape
No dynamic output shape
Input:  [1, 3, 256, 256]
Output: [1, 2, 256, 256]
```

File naming:

```text
road_deeplabv3_mnv2_256_fp32.onnx
road_deeplabv3_mnv2_256_qdq_int8.onnx
road_deeplabv3_mnv2_256_qdq_int8_1.nb
```

---

## 8. Quantization Requirements

Target quantization:

```text
INT8
per-tensor
asymmetric
static calibration
```

Do not use per-channel quantization for the STM32MP2 NPU path unless ST explicitly confirms it is supported for the selected flow.

Calibration data:

```text
representative road images
same preprocessing as deployment
recommended count: 64-128 images
format: .npz accepted by ST tooling/cloud
```

Calibration image coverage should include:

- Bright/dark scenes.
- Normal and cyan-cast camera frames.
- Straight and curved roads.
- Background-heavy frames.
- Low-confidence hard examples.

Avoid random calibration data for production models.

---

## 9. ST Cloud / ST Edge AI Core Requirements

Target configuration:

```text
Board: STM32MP257F-EV1 or STM32MP2 family
Runtime: STM32MPU / STAI MPU / ONNX Runtime X-LINUX-AI flow
Accelerator: NPU / VIP9000 / GCNano
Quantization: INT8 per-tensor
Output: NBG / .nb
```

The ONNX model alone is not the final runtime artifact. The board-side NPU path should use `.nb`.

Expected `.nb` metadata:

```text
input:  tensor(int8) or tensor(uint8)
output: tensor(int8) or tensor(uint8)
```

Reject or treat as fallback:

```text
input/output tensor(float16)
raw_run_ms hundreds of ms
no /dev/galcore ioctl
```

---

## 10. Board-Side Acceptance Tests

### 10.1 Contract and Latency

Run:

```bash
PYTHONPATH=. python3 FlightController/tools/validate_nb_npu_contract.py \
  --model FlightController/Solutions/model/road_deeplabv3_mnv2_256_qdq_int8_1.nb \
  --runs 20 \
  --max-mean-ms 80 \
  --profile-raw-stai
```

Initial pass gate:

```text
wrapped mean latency < 80 ms
raw_run_ms < 60 ms preferred
finite outputs = True
input/output int8 or uint8
```

### 10.2 Hardware Call Proof

Run:

```bash
PYTHONPATH=. strace -f -yy -e openat,ioctl \
  python3 FlightController/tools/validate_nb_npu_contract.py \
    --model FlightController/Solutions/model/road_deeplabv3_mnv2_256_qdq_int8_1.nb \
    --runs 3 \
    --max-mean-ms 80 \
    --profile-raw-stai \
  2>&1 | tee /media/sdcard/road_seg_nb_strace_yy.log

grep -E 'galcore|ioctl\([^)]*</dev/galcore' /media/sdcard/road_seg_nb_strace_yy.log
```

Required:

```text
/dev/galcore open
/dev/galcore ioctl
```

### 10.3 Mask Overlay

Run:

```bash
PYTHONPATH=. python3 FlightController/tools/render_deeplab_nb_overlay.py \
  --model FlightController/Solutions/model/road_deeplabv3_mnv2_256_qdq_int8_1.nb \
  --output-dir /media/sdcard/npu_debug/road_seg_overlay \
  tests/roads/IPC_2026-06-14.10.32.58.1790.jpg
```

Required:

```text
road class is not all black
road class is not all white
road mask aligns with visible road
class histogram has plausible road/background ratio
```

If the road class is channel 0 instead of channel 1, either retrain/export with the expected class order or configure the runtime decoder explicitly.

---

## 11. Integration Requirements

Do not remove the existing YOLO path immediately.

Recommended integration:

```text
road_perception.py
  --model-type yolo-seg   existing path
  --model-type deeplab    new semantic path
```

DeepLab decode path:

```text
NBGraphSession.run()
-> output [1,2,H,W]
-> argmax channel
-> road mask
-> crop/resize from model input to original frame
-> _clean_mask()
-> _extract_centerline_and_intervals()
-> _compute_pixel_error()
-> _compute_centerline_angle()
-> RoadPerceptionResult
```

Fallback behavior:

```text
If NPU model load fails, do not silently fly.
For dry-run tests, report lost perception.
For flight mode, require explicit operator choice before fallback.
```

---

## 12. Final Go/No-Go Criteria

A model is ready for road-following dry-run only when:

- `.nb` loads reliably.
- `/dev/galcore` ioctl is observed.
- Mean wrapped inference is below 80 ms.
- Road mask is visually plausible on multiple saved road images.
- Centerline extraction is stable.
- `RoadFollower` command output is smooth in dry-run.

It is ready for flight tests only after:

- Dry-run camera test passes.
- Radar safety remains active.
- Flight controller output remains disabled unless `--enable-flight` is explicit.
- Mask failures produce safe lost-road behavior.
