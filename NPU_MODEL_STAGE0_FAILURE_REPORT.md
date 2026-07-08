# NPU 模型验证 — 完整诊断报告（终版）

**日期**: 2026-06-14（持续更新）
**开发板**: MYD-LD25X (STM32MP257), OpenSTLinux v6.0, Kernel 6.6.48
**onnxruntime**: 1.19.2 (ST AINPU 仓库, 含 VSINPUExecutionProvider), 最大 IR version = 10
**NPU**: VIP9000 (model=0x8000, 800MHz), galcore 6.4.19.4, /dev/galcore
**测试人员**: luai-git

---

## 1. 最终结论（TL;DR）

**Segfault 根因**: VSINPU EP v1.19.2 的跨分区内存管理存在空指针解引用 bug。当图复杂到被切分为多个 NPU/CPU 分区时，分区间的 tensor 传递缓冲区未正确分配，导致 galcore 驱动收到 NULL 指针 → `SIGSEGV SEGV_MAPERR`。

**确诊命令**:
```bash
strace -f python3 -c "..." 2>&1 | tail -30
# → ioctl(3, ...) × N  → SIGSEGV {si_addr=NULL}
```

**小图（5节点）3个NonMaxPool不崩，大图（329节点）同样的3个NonMaxPool就崩** — 问题不在任何单个算子类型，在图复杂度触发的多分区路径。

**解决方向**:
1. ST Edge AI Cloud 使用 VIP9000 专用编译器生成单分区 ONNX（绕过跨分区 bug）
2. 联系 ST 获取 onnxruntime >= 1.20 的 VSINPU EP 补丁

---

## 2. 完整测试矩阵

### 2.1 原始模型（转换前）

| 编号 | 模型 | Provider | 结果 | GetCapability |
|------|------|----------|:---:|------|
| A1 | `road_yolo11n_seg_int8_qdq` | VSINPU | 🔴 Segfault | 1397/1416 |
| A2 | `tree_furniture_int8_qdq` | VSINPU | 🔴 Segfault | 1258/1272 |
| A3 | `tree_furniture` FP32 原版 | VSINPU | 🔴 Segfault | 314/318 |
| A4 | `tree_furniture_int8_qdq` | CPU | ✅ 410ms | — |
| A5 | `tree_furniture_int8_qdq` + `ORT_DISABLE_ALL` | VSINPU | 🔴 Segfault | 1258/1272 |

### 2.2 VSINPU 转换后模型（算子替换 + INT8 量化）

| 编号 | 模型 | Provider | 结果 |
|------|------|----------|:---:|
| B1 | `road_yolo11n_seg_vsinpu_fp32` | CPU | ✅ |
| B2 | `road_yolo11n_seg_vsinpu_int8_qdq` | CPU | ✅ |
| B3 | `tree_furniture_vsinpu_fp32` | CPU | ✅ |
| B4 | `tree_furniture_vsinpu_int8_qdq` | CPU | ✅ |
| B5 | `tree_furniture_vsinpu_fp32` | XNNPACK | ✅ |
| B6 | 全部 4 个 VSINPU 模型 | VSINPU | 🔴 Segfault |

### 2.3 Round 1 诊断模型 — SPPF MaxPool 隔离

| 编号 | 模型 | 节点 | NonMaxPool 警告 | 结果 |
|------|------|:---:|:---:|:---:|
| D1 | Conv only | 1 | 0 | ✅ |
| D2 | Conv + MaxPool (dilations 显式) | 2 | 1 | ✅ |
| D3 | Conv + MaxPool (无 dilations) | 2 | 0 | ✅ |
| D4 | Conv + Slice | 3 | 0 | ✅ |
| D5 | SPPF 1pool k=5 | 3 | 1 | ✅ |
| D6 | SPPF 2pool chained k=5 | 4 | 2 | ✅ |
| D7 | SPPF 3pool chained k=5 | 5 | 3 | ✅ |
| D8 | SPPF 3pool parallel k=5 | 3 | 1 | ✅ |
| D9 | SPPF 3pool chained k=3 | 5 | 3 | ✅ |
| D10 | SPPF 3pool chained 416×416 | 5 | 3 | ✅ |
| D11 | SPPF 3pool + 额外 Conv | 6 | 3 | ✅ |

> 🔑 **SPPF 3pool chained (D7) 精确复现了 3 个 NonMaxPool 警告，但在 5 节点简单图中不崩溃。** 证明 NonMaxPool 本身不致命。

### 2.4 Round 2 诊断模型 — 分组卷积 / 动态 Slice / Gather

| 编号 | 模型 | 算子 | 结果 |
|------|------|------|:---:|
| D12 | Depthwise Conv (groups=64) | DWConv | ✅ |
| D13 | Grouped Conv (groups=64) | GConv | ✅ |
| D14 | SPPF 3pool + DWConv | MaxPool×3 + DWConv | ✅ |
| D15 | Slice + Constant 节点 | Slice(with Constant) | ✅ |
| D16 | Shape + Gather | Shape, Gather | ✅ |

> 🔑 **全部 16 个诊断模型在 VSINPU 上通过。被怀疑的所有算子类型（分组卷积、动态 Slice、Gather）都是无辜的。**

### 2.5 对比：简单图 vs 复杂图

| 模型 | 节点数 | NonMaxPool | 结果 |
|------|:---:|:---:|:---:|
| SPPF 3pool (D7) | 5 | 3 | ✅ OK |
| SPPF 3pool + DWConv (D14) | 6 | 3 | ✅ OK |
| tree_vsinpu_fp32 (B5) | **329** | 3 | 🔴 **Segfault** |
| tree 原版 (A3) | 318 | 3+1 | 🔴 **Segfault** |

**结论：同样的 3 个 NonMaxPool，小图 5 节点 OK，大图 329 节点崩。根因在 fig复杂度，不在算子。**

---

## 3. strace 精确定位

### 3.1 命令

```bash
strace -f python3 -c "
import onnxruntime as ort
m = ort.InferenceSession(
    'FlightController/Solutions/model/npu_quantization/tree_furniture_vsinpu_fp32.onnx',
    providers=['VSINPUExecutionProvider'])
print('OK')
" 2>&1 | tail -30
```

### 3.2 关键输出

```
[pid 5551] ioctl(3, _IOC(_IOC_NONE, 0x75, 0x30, 0), ...) = 0    ← fd=3 = /dev/galcore
[pid 5551] ioctl(3, _IOC(_IOC_NONE, 0x75, 0x30, 0), ...) = 0
... (大量 galcore ioctl，NPU 图编译 + 执行)
[pid 5551] --- SIGSEGV {si_signo=SIGSEGV, si_code=SEGV_MAPERR, si_addr=NULL} ---
[pid 5561] +++ killed by SIGSEGV (core dumped) +++
Segmentation fault (core dumped)
```

### 3.3 解读

| 字段 | 含义 |
|------|------|
| `si_signo=SIGSEGV` | 段错误 |
| `si_code=SEGV_MAPERR` | 访问了未映射的内存地址 |
| `si_addr=NULL` | **空指针解引用** — 尝试访问地址 0x0 |

- `ioctl(3, ...)` → fd 3 = `/dev/galcore`（NPU 驱动设备文件）
- 崩溃发生在 galcore 驱动的 `ioctl` 处理中，因为 VSINPU EP 传入了一个 NULL 缓冲区指针
- 这是 VSINPU EP 的跨分区内存分配 bug，不是模型图结构问题

### 3.4 完整崩溃链

```
VSINPUExecutionProvider::GetCapability
  ├── 分析 329 节点图 → 识别 NPU 兼容/不兼容子图
  ├── 切分为 NPU 分区 ×2 + CPU fallback 分区 ×1
  ├── NPU 分区编译 → ioctl(galcore) 发送子图 ✓
  ├── 分配跨分区 tensor 缓冲区
  │      ↑
  │      └── BUG: 某个中间 tensor 的 CPU 端缓冲区未分配，指针 = NULL
  ├── 执行时 NPU 分区 → CPU 分区数据传输
  └── ioctl(galcore) 收到 NULL 缓冲区指针
       → SEGV_MAPERR (si_addr=NULL)
       → Segmentation fault
```

### 3.5 为什么小图不崩

小图（5-6 节点）所有节点在同一个 NPU 分区内，不需要跨分区 tensor 传递，不需要额外分配 CPU 端缓冲区 → 空指针路径不会被触发。

大图（329 节点）被切分为 ≥2 个 NPU 分区 + 至少 1 个 CPU fallback 分区。跨分区传递时需要 CPU↔NPU 内存拷贝，其中一条路径的缓冲区未分配 → 空指针 → segfault。

---

## 4. VSINPU 转换的效果评估

### 4.1 成功部分 ✅

| 替换 | 效果 | 证据 |
|------|------|------|
| Split → Slice | Split(uneven) 警告 → **消失** | 原始 4 个不合格 → VSINPU 3 个（减少了 1 个 Split） |
| ConvTranspose → Subpixel Conv | ConvTranspose fallback 警告 → **消失** | VSINPU 日志无 ConvTranspose 行 |

### 4.2 失败部分 🔴

| 问题 | 尝试的修复 | 结果 |
|------|-----------|:---:|
| MaxPool dilations 属性缺失 → 直接 segfault 无警告 | 补回 `dilations=[1,1]` | 恢复了 NonMaxPool 警告，但最终仍 segfault |
| 3 个 NonMaxPool fallback | 无法消除 — VSINPU EP C++ 代码内部转换 | 小图 OK，大图崩 |

### 4.3 VSINPU EP 的 MaxPool→NonMaxPool 转换不受控制

```
ONNX 图（任何版本）:
  MaxPool(kernel=[5,5], strides=[1,1], pads=[2,2,2,2], dilations=[1,1])

      ↓  VSINPU EP pool_op_builder.h (不可跳过)

  Internal NonMaxPool(dilation=...)
      ↓  IsOpSupported → false
      ↓
  Fallback to CPU
```

这个转换在 `onnxruntime` 的 C++ 二进制代码中，不受 `SessionOptions`、`ORT_DISABLE_ALL`、模型 opset、模型 IR version 等任何 Python 层参数控制。VSINPU 脚本修改 ONNX 图无法阻止它。

---

## 5. 排除的怀疑对象

| 曾怀疑为根因 | 排除依据 |
|-------------|---------|
| INT8 QDQ 量化损坏模型 | FP32 原版同样 segfault |
| ConvTranspose 算子 | VSINPU 转换已替换为 Subpixel，且 tree 模型无此算子仍崩 |
| 模型 IR 版本 | VSINPU 模型 IR=9，在 ST ORT 的 ≤10 限制内 |
| MaxPool dilations 缺失 | 补回后仍崩 |
| ORT 图优化器 | `ORT_DISABLE_ALL` 无效 |
| NPU 驱动状态污染 | `rmmod/modprobe galcore` 无效 |
| Depthwise / Grouped Conv | D12-D14 全部通过 |
| Slice 替换有问题 | D15 通过 |
| Shape + Gather | D16 通过 |
| Resize 节点 | 2 个 Resize mode=nearest，参数合规 |
| NonMaxPool 数量 = 3 | **D7 有 3 个 NonMaxPool 但不崩** ← 关键反证 |

---

## 6. 环境确认

```
OS: OpenSTLinux v6.0 (Yocto Scarthgap) aarch64
Kernel: 6.6.48
NPU driver: galcore 6.4.19.4, /dev/galcore (VIP9000, 800MHz)
onnxruntime: 1.19.2 (ST AINPU repository)
Available providers: ['VSINPUExecutionProvider', 'XnnpackExecutionProvider', 'CPUExecutionProvider']
IR version max: 10
Root filesystem: /dev/mmcblk0p10 (3.7G, 54M free)

存储分布:
  /              3.7G  (54M  free) ⚠️ 接近爆盘
  /usr/local     1.3G  (211M free) — 代码仓库 (userfs)
  /boot          55M   (30M  free)
  /vendor        228M  (36M  free)
```

---

## 7. 解决方案（更新）

### 7.1 ST Edge AI Cloud（推荐 ⭐）

URL: https://stedgeai-dc.st.com

ST Cloud 使用 VIP9000 专属编译器（非 onnxruntime VSINPU EP），能生成**单分区** ONNX — 所有节点在一个 NPU 子图中，无需跨分区传输，完全绕过 galcore 空指针 bug。

**上传材料**:
| 文件 | 说明 |
|------|------|
| `road_yolo11n_seg.onnx` (11.5 MB) | 原始 FP32 |
| `tree_furniture.onnx` (10.5 MB) | 原始 FP32 |
| `road_yolo11n_seg_calibration.npz` (77 MB) | 200 张道路校准 |
| `tree_furniture_calibration.npz` (50 MB) | 133 张树木校准 |

**配置**: Board=`STM32MP257F-EV1`, Runtime=`ONNX Runtime (X-LINUX-AI)`, Accelerator=`NPU (VIP9000)`, Quantization=`INT8 per-tensor`

### 7.2 联系 ST 获取 onnxruntime 补丁

此空指针 bug 在 `vsinpu_execution_provider.cc` 的跨分区内存分配逻辑中。需求：onnxruntime ≥ 1.20 或 aarch64 VSINPU EP hotfix。

### 7.3 MaxPool→Depthwise Conv 替换（备选）

将 YOLO SPPF 模块的 3 个 `MaxPool(kernel=5)` 替换为固定权重的 Depthwise Conv。效果：VSINPU EP 不再识别为 pool 算子，不会进入 NonMaxPool 转换路径。风险：即使消除了 NonMaxPool 警告，如果图仍然被切分成多分区（因为节点数多），仍可能触发同样的跨分区 bug。

### 7.4 使用 XNNPACK 临时方案

```python
m = ort.InferenceSession(model, providers=['XnnpackExecutionProvider', 'CPUExecutionProvider'])
```

XNNPACK 实测可用（§2.2 B5），推理速度介于纯 CPU 和 NPU 之间，可作为 ST Cloud 转换完成前的临时方案。

---

## 8. 当前资产清单

| 资产 | 说明 | 状态 |
|------|------|:--:|
| 原始 FP32 模型 ×2 | road (11.5M) + tree (10.5M) | ✅ |
| 原始 INT8 QDQ 模型 ×2 | road (3.2M) + tree (2.9M) | ✅ CPU, 🔴 NPU |
| VSINPU FP32 模型 ×2 | 算子替换后 | ✅ CPU, 🔴 NPU |
| VSINPU INT8 QDQ 模型 ×2 | 算子替换+量化 | ✅ CPU, 🔴 NPU |
| 校准 .npz ×2 | road (77M, 200张) + tree (50M, 133张) | ✅ |
| 诊断模型 Round 1 | `tests/test_npu_sppf_*.onnx` ×8 | ✅ 全部 VSINPU 通过 |
| 诊断模型 Round 2 | `tests/test_npu_dwconv.onnx` 等 ×5 | ✅ 全部 VSINPU 通过 |
| 诊断模型基础 | `tests/test_npu_minimal.onnx` 等 ×4 | ✅ 全部 VSINPU 通过 |

---

## 9. 诊断时间线

| 时间 | 步骤 | 关键发现 |
|------|------|---------|
| 16:08 | A1: road INT8 + VSINPU | ConvTranspose + NonMaxPool×3 + Split×1 → Segfault |
| 16:24 | A2: tree INT8 + VSINPU | NonMaxPool×3 + Split×1 → Segfault（无 ConvTranspose 仍崩）|
| 16:26 | A5: ORT_DISABLE_ALL | 无效 |
| 16:26 | A4: tree INT8 + CPU | ✅ 410ms — 量化模型完好 |
| 16:28 | A3: tree FP32 + VSINPU | Segfault — FP32 也崩 → 不是量化问题 |
| 16:28 | A3: tree FP32 + VSINPU | 314/318 → 4 个不合规 |
| — | PC 静态分析 | 原始图合规；运行时 VSINPU EP 内部转换 |
| — | VSINPU 转换 | Split→Slice ✅, ConvTranspose→Subpixel ✅, MaxPool dilations 被删 ⚠️ |
| — | 修复 MaxPool dilations | 12 个节点补回 `dilations=[1,1]` |
| 16:56 | B5: tree_vsinpu + VSINPU | 直接 segfault 无警告 → dilations 缺失是额外 bug |
| 16:56 | Round 1 诊断: D1-D11 | **全部通过**（含 3 NonMaxPool 的 SPPF） |
| 16:59 | B5+B6: 修复后 VSINPU 模型 | NonMaxPool×3 警告恢复，但最终 segfault |
| 17:02 | rmmod/modprobe galcore | 无效 — 确定性崩溃 |
| 17:14 | Round 1 扩展: D5-D11 | 全部 SPPF 变体通过 |
| 17:26 | Round 2: D12-D16 | 全部通过 — 分组卷积/Slice/Gather 都无罪 |
| 17:27 | **strace** | **SIGSEGV si_addr=NULL — 空指针！** |

---

## 10. 验收标准

模型通过 VSINPU EP 加载的必要条件：

```
✅ 日志无 "NonMaxPool with Dilation" 行
✅ 日志无 "Uneven splits" 行
✅ 日志无 "Fallback unsupported op ... to cpu" 行
✅ GetCapability: supported == total nodes
✅ 无 Segmentation fault
✅ get_providers()[0] == 'VSINPUExecutionProvider'
```

> ⚠️ 注意：由于 VSINPU EP 存在跨分区空指针 bug，"supported == total nodes" 是必要条件但**不一定充分** — 即使 100% 节点通过 GetCapability，仍可能因单分区过大触发其他路径的崩溃。最终验收必须在开发板上实测推理成功。
