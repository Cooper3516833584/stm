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

If tensor metadata is `int8`/`uint8` but latency is still high, rerun with raw
STAI profiling:

```bash
PYTHONPATH=. python3 FlightController/tools/validate_nb_npu_contract.py \
  --model FlightController/Solutions/model/<candidate>.nb \
  --runs 20 \
  --max-mean-ms 80 \
  --profile-raw-stai
```

Interpretation:

| Observation | Meaning |
|---|---|
| `raw_run_ms` is also hundreds of ms | The compiled network/runtime path is slow; check galcore ioctl and cloud benchmark. |
| `raw_run_ms` is fast but total latency is slow | Python input conversion or output dequantization is the bottleneck; adapt the runtime pipeline to feed quantized tensors directly. |
| No `/dev/galcore` in strace | The model is not using the NPU even if I/O is int8. |
| `/dev/galcore` exists but `raw_run_ms` is slow | NPU may be called, but model placement, driver/runtime, or generated NBG is not performant enough. |

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

Model Zoo normally gives an ONNX model first. That is expected. For STM32MP2x
high-performance NPU/GPU execution, ONNX is an input artifact, not the final
runtime artifact. The ONNX must be quantized and converted to NBG (`.nb`) with
ST Edge AI Core or ST Edge AI Developer Cloud.

### 4.1 Get the ONNX baseline

1. Start with the ST semantic-segmentation DeepLab v3 example or model-zoo configuration.
2. Export or collect the generated ONNX model.
3. Use a clear file name when copying it into this project:

```text
FlightController/Solutions/model/baseline_deeplabv3_mp257_fp32.onnx
```

This ONNX can be used to verify model shape and output contract, but it is not
the `.nb` model consumed by `validate_nb_npu_contract.py`.

### 4.2 Convert ONNX to NBG / `.nb`

Use one of these two official routes:

| Route | Use when | Output |
|---|---|---|
| ST Edge AI Developer Cloud | You want the fastest manual path and cloud benchmark/download | optimized NBG / `.nb` |
| ST Edge AI Core offline compiler | You have the host-side ST tool installed and want repeatable local conversion | optimized NBG / `.nb` |

Developer Cloud target settings:

```text
Board: STM32MP257F-EV1 or STM32MP2 family
Runtime: ONNX Runtime / X-LINUX-AI or STM32MPU deployment flow
Accelerator: NPU / VIP9000 / GCNano
Quantization: INT8, per-tensor
```

Important: if the cloud only returns another quantized `.onnx`, continue to the
Optimize / Benchmark / Generate stage for STM32MP2. ST documentation states that
when benchmark is run on an STM32MP2x board with AI hardware accelerator, the
NBG model is generated and can be downloaded.

4. Copy the downloaded `.nb` to:

```text
FlightController/Solutions/model/baseline_deeplabv3_mp257_int8.nb
```

5. Run the two validation commands in section 2.
6. Save evidence:

```text
/media/sdcard/nb_npu_strace.log
/media/sdcard/baseline_deeplabv3_validation.txt
```

### 4.3 Optional ONNX smoke test before `.nb`

If you only have the ONNX right now, use it only as a pre-check:

```bash
PYTHONPATH=. python3 FlightController/tools/validate_vsinpu_model.py \
  --model FlightController/Solutions/model/baseline_deeplabv3_mp257_fp32.onnx \
  --runs 3 \
  --allow-cpu-provider
```

This does not prove the NBG path. It only checks whether the board-side ONNX
Runtime can load/run the model without crashing.

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
