# STM32MP25 NBG toolchain repair and model compatibility record

Date: 2026-07-11

## Result

The ST Edge AI Core official `stm32mp25` path has been repaired and verified to
compile an NBG with the actual STM32MP257 target profile:

```text
VSIMULATOR_CONFIG=GCNANOULTRA31_VIP2_PID0X15
NBG header: 56 50 4d 4e 20 00 01 00 15 00 00 00 ...
                                      ^ pid 0x15
```

This was verified from the existing official control model:

```text
/home/developer/stm32mp257_project/stedgeai/output/control_add_uint8/model.nb
56504d4e20000100150000005f4e4348
```

The original Candidate_A-FloatIO NBG must not be used. Its header was
`...20000010...` (target `0x10000020`), which the board rejected because the
actual device target is `0x15`. Changing only that header gets past the first
check but then fails SRAM/context allocation; it is not a valid workaround.

## Official target selection

The device profile is selected inside the ST Edge AI Core compiler backend,
not by `gen_nbg`'s ordinary environment knobs. The relevant configuration seen
during `stedgeai generate --target stm32mp25` is:

```text
VSIMULATOR_CONFIG=GCNANOULTRA31_VIP2_PID0X15
VIV_VX_ENABLE_GRAPH_TRANSFORM=-pcq:1-fc2conv:2
VIV_VX_ENABLE_SAVE_NETWORK_BINARY=1
VIV_VX_NBG_INPUT_RANK=NCHW
VIV_VX_SAVE_NETWORK_BINARY_PATH=network_binary.nb
```

The profile string is present in ST's
`passes/backend/acuity_code_compiler*.so`. Running a manually exported
`gen_nbg` program with `VIV_VX_GRAPH_DEVICE_ID`, `VIV_GRAPHICS_CARD_ID`,
`VIV_VX_PROFILE`, or SRAM variables never changed the target header from
`0x10000020`. Those variables cannot replace the official compiler path.

## Toolchain repair on the VM

The following wrapper is installed at:

```text
/home/developer/stm32mp257_project/stedgeai/tools/gcc
```

It performs two narrow compatibility fixes for generated `export_ovxlib`
code:

1. Repairs the missing closing brace in generated `vnn_pre_process.c`.
2. Adds `-Wl,-rpath-link,${VIVANTE_SDK_DIR}/lib` for the `gen_nbg` link.

To retain the official ST compiler process while using the wrapper, run:

```bash
export CROSS_COMPILE=/home/developer/stm32mp257_project/stedgeai/tools/
export VIVANTE_SDK_DIR=/home/developer/STMicroelectronics/STEdgeAI/2.2/Utilities/linux/lib/python3.9/site-packages/acuitylib/vsi_sdk/prebuilt-sdk/x86_64_linux

/home/developer/STMicroelectronics/STEdgeAI/2.2/Utilities/linux/stedgeai generate \
  --model /path/to/model.onnx \
  --target stm32mp25 \
  --workspace /path/to/workspace \
  --output /path/to/output \
  --verbosity 3
```

The temporary edit to ST's `vxnetgenerator.so` was restored from
`vxnetgenerator.so.bak_codex_20260711`; no ST binary patch remains active.

## Candidate model diagnosis

Both candidate QDQ models have 208 nodes: 36 `Conv`, 119
`DequantizeLinear`, 45 `QuantizeLinear`, 5 `Add`, 1 `Mul`, 1 `Concat`, and 1
`Resize`. The first computation is an input normalization subgraph:

```text
images -> QDQ -> Mul(scale) -> QDQ -> Add(bias) -> QDQ -> first Conv
```

For the signed INT8 candidate, its effective parameters are:

```text
input shape: [1, 3, 256, 256]
normalized = images * [5.020309, 5.391376, 5.565995]
             + [-2.159241, -2.436312, -1.748412]
input quantization: scale=0.003921568859, zero_point=-128, int8
output dequantization: scale=0.130377740, zero_point=6, int8
```

Under the real `pid 0x15` profile, the original model fails with:

```text
MULTIPLY: FLOAT32, ASYM UINT8 not supported
```

Removing that normalization subgraph and moving it to external preprocessing
only moves the failure to the first convolution:

```text
CONV2D: FLOAT32, SYM INT8 not supported
```

A further derived graph with direct INT8 input/output reaches:

```text
vnn_VerifyGraph: -3
The requested set of parameters produce a configuration that cannot be supported.
```

Therefore the current DeepLabV3-MobileNetV2 QDQ graph is not NBG-compilable
for `GCNANOULTRA31_VIP2_PID0X15`. This is a model compatibility/resource issue
after the official device configuration is active, not an installation,
linker, board package, or NBG-header issue.

## Recommended next action

Re-export or redesign the segmentation model for a fully supported INT8 path:

1. Keep input/output quantization boundaries simple and move float
   normalization to application code.
2. Start with a smaller fully INT8 segmentation graph, then add blocks while
   compiling with the official `stm32mp25` command after each change.
3. Submit the minimized derived model and the `vnn_VerifyGraph: -3` log to ST
   support. It is a compact reproducer showing the failure persists after the
   problematic float normalization is removed.
4. Accept an NBG only when its header word at byte offset 8 is `0x00000015`;
   then validate actual board execution with `nbg_benchmark` and galcore ioctl
   activity.

The original candidate files remain unchanged. The helper scripts used during
this work are preserved in `_codex_files`.
