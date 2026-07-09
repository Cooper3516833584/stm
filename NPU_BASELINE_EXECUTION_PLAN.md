# STM32MP257 NPU Baseline Execution Plan

**Date**: 2026-07-09
**Board**: MYD-LD25X / STM32MP257 / OpenSTLinux v6.0
**Goal**: first prove a real VIP9000 NPU execution path with an official or simple INT8 `.nb`, then return to the current YOLO11-seg road model.

---

## 1. Current Decision

Do not treat the existing FP32 `.nb` files as successful NPU acceleration:

- `FlightController/Solutions/model/road_yolo11n_seg_1.nb`
- `FlightController/Solutions/model/road_yolo11n_seg_vsinpu_fp32_opt.nb`

Current evidence shows float16 I/O, about 600 ms inference, and no observed `/dev/galcore` ioctl. These are useful only as STAI smoke-test artifacts.

The next milestone is a **known-good INT8 `.nb` baseline** that satisfies all gates below.

---

## 2. True NPU Acceptance Gates

A model is accepted as "real NPU" only if all conditions pass:

1. `.nb` loads through `stai_mpu` / `NBGraphSession`.
2. Tensor metadata is quantized or static-affine, preferably `int8` or `uint8`.
3. Mean inference latency is far below the current fallback path. Initial gate: `< 80 ms`; target gate: `< 30 ms`.
4. `strace` shows `/dev/galcore` open/ioctl activity during inference.
5. Output arrays are finite and stable over repeated runs.

Run the direct contract check:

```bash
PYTHONPATH=. python3 FlightController/tools/validate_nb_npu_contract.py \
  --model FlightController/Solutions/model/<candidate>.nb \
  --runs 20 \
  --max-mean-ms 80
```

Then run the hardware-call proof:

```bash
PYTHONPATH=. strace -f -e openat,ioctl \
  python3 FlightController/tools/validate_nb_npu_contract.py \
    --model FlightController/Solutions/model/<candidate>.nb \
    --runs 5 \
    --max-mean-ms 80 \
  2>&1 | tee /media/sdcard/nb_npu_strace.log

grep -E 'galcore|/dev/galcore|ioctl' /media/sdcard/nb_npu_strace.log
```

Expected result:

```text
[PASS] tensor metadata and latency look NPU-compatible
... openat(... "/dev/galcore" ...)
... ioctl(... galcore fd ...)
```

---

## 3. Official Baseline Candidate

Use ST's official model-zoo flow before spending more effort on YOLO11-seg.

Recommended candidate:

```text
Use case: Semantic Segmentation
Model: DeepLab v3
Target: STM32MP257F-EV1
Input options: 256x256x3, 320x320x3, 416x416x3, 512x512x3
Services: training, evaluation, quantization, benchmarking, prediction, deployment
```

Rationale:

- It is directly listed by ST as deployable to `STM32MP257F-EV1`.
- Semantic segmentation is closer to road-mask output than YOLO instance segmentation.
- It avoids YOLO prototype/mask-coefficient postprocessing and should be a better first NPU proof.

Reference:

- `https://github.com/STMicroelectronics/stm32ai-modelzoo-services`
- `https://github.com/STMicroelectronics/stm32ai-modelzoo`

---

## 4. ST Cloud / Model Zoo Procedure

1. Start with the ST semantic-segmentation DeepLab v3 example or model-zoo configuration.
2. Select target:

```text
Board: STM32MP257F-EV1 or STM32MP2 family
Runtime: ONNX Runtime / X-LINUX-AI or STM32MPU deployment flow
Accelerator: NPU / VIP9000 / GCNano
Quantization: INT8, per-tensor
```

3. Generate/download the optimized `.nb`.
4. Copy it to:

```text
FlightController/Solutions/model/baseline_deeplabv3_mp257_int8.nb
```

5. Run the two validation commands in section 2.
6. Save evidence:

```text
/media/sdcard/nb_npu_strace.log
/media/sdcard/baseline_deeplabv3_validation.txt
```

---

## 5. ST Support Reproduction Package

If the official baseline passes but YOLO11-seg still fails, send ST a minimal package for the YOLO issue.

Use these files:

```text
FlightController/Solutions/model/stcloud_upload/road_yolo11n_seg_vsinpu_fp32_for_stcloud.onnx
FlightController/Solutions/model/stcloud_upload/selected/road_calib_selected_032.npz
FlightController/Solutions/model/road_yolo11n_seg_fp32_for_stcloud_PerTensor_quant_road_calib_selected_128_npz_2.onnx
NPU_ST_CLOUD_20260709_FINDINGS.md
NPU_MODEL_NB_DIAGNOSIS.md
```

Ask ST these exact questions:

1. Does STM32MP257 / VIP9000 support YOLO-seg style dual-output INT8 NBG (`boxes+mask coeffs`, `mask prototypes`)?
2. For STM32MP2, should the Optimize input be QDQ or QOperator ONNX?
3. Are there required opset, IR version, static-shape, output-name, or graph I/O dtype constraints?
4. Is `Generation does not contain any output` a known STM32AI MPU Optimize failure for quantized YOLO-seg graphs?
5. Can ST provide a known-good INT8 `.nb` segmentation sample for STM32MP257F-EV1?

---

## 6. After Baseline Passes

Choose the next branch based on results:

| Result | Next action |
|---|---|
| Official DeepLab v3 `.nb` passes all gates | Adapt road pipeline to a semantic mask model and evaluate retraining/fine-tuning. |
| Official `.nb` loads but no galcore ioctl | Board software stack or STAI deployment flow is still wrong; fix platform before model work. |
| Official `.nb` cannot be generated | Use ST support with the official model-zoo config first. |
| Official baseline passes, YOLO fails | Treat YOLO11-seg as toolchain/model-structure incompatibility; pursue ST ticket, simpler model, or graph split. |

---

## 7. Local Tool Added

New board-side validator:

```text
FlightController/tools/validate_nb_npu_contract.py
```

It intentionally fails current float16 fallback `.nb` models unless `--allow-float-io` is passed. This keeps the acceptance criteria strict.
