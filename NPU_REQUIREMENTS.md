# NPU 软件包需求分析

**日期**: 2026-06-07  
**平台**: STM32MP257 (MYD-LD25X), Debian 12, Python 3.11, aarch64  
**最终决策**: 切换至 OpenSTLinux v6.2

---

## 1. 硬件与底层库现状

| 层 | 文件 | 状态 | 来源 |
|------|------|:--:|------|
| 内核驱动 | `galcore.ko` (v6.4.15.6) | ✅ 已加载 | BSP 内核 |
| 用户态接口 | `libGAL.so` (1.9MB) | ✅ 已安装 | vendorfs ext4 |
| NPU 编译器运行时 | `libVSC.so` (17MB) | ✅ 已安装 | vendorfs ext4 |
| 着色器编译器 | `libCLC.so`, `libGLSLC.so`, `libSPIRV_viv.so` | ✅ 已安装 | vendorfs ext4 |
| OpenCL NPU 后端 | `libOpenCL_VSI.so.3` (890KB) | ✅ 已安装 | vendorfs ext4 |
| OpenGL ES | `libGLESv2.so`, `libEGL.so` | ✅ 已安装 | vendorfs ext4 |
| 图形缓冲 | `libgbm_viv.so` (67KB) | ✅ 已安装 | vendorfs ext4 |
| Vulkan NPU 后端 | `libvulkan_VSI.so.1` (1.2MB) | ✅ 已安装 | vendorfs ext4 |
| 设备节点 | `/dev/galcore` (crw-rw-rw-) | ✅ 可访问 | — |

---

## 2. NPU 可加速的计算任务

| 优先级 | 子系统 | 算子 | 当前 CPU 耗时 | NPU 预期 |
|:---:|------|------|:--:|:--:|
| 🔴 P0 | 道路视觉感知 | YOLO11-seg ONNX 推理 | **~1800ms/帧 (0.6 FPS)** | **5-15ms (60-200 FPS)** |
| 🟡 P1 | 目标检测 | FastestDet/YOLO/DAMO ONNX 推理 | 未测 | 实时可用 |
| 🟢 P2 | mask 后处理 | 形态学 (dilate/erode/close/open) | <5ms | 不必要 |
| ⬜ 不需要 | 雷达点云 | `np.linalg.norm`, 布尔索引 | 0.3ms | 不必要 |
| ⬜ 不需要 | 雷达 SLAM | `cv2.HoughLinesP` (800×800 稀疏) | <1ms | 不必要 |
| ⬜ 不需要 | 路径规划 | 势场法 3600 格 | 非帧级 | 不必要 |

---

## 3. NPU 适配历程 (2026-06-07)

### 3.1 已下载的 ST 软件包分析

| 包 | 内容 | 有用？ |
|------|------|:--:|
| `meta-st-x-linux-ai-6.2.0.zip` (145MB) | Yocto 编译脚本 + `stai_mpu` (Python 3.12) | ❌ |
| `AISDK-Y-MP2` SDK (247MB) | 交叉编译工具链 + ONNX Runtime CPU 版 | ❌ |

### 3.2 从 ST APT 仓库下载的 .deb 包

成功从 `http://extra.packages.openstlinux.st.com/AINPU/6.2` 下载：

| 包 | 大小 | VSINPU 符号 |
|------|------|:--:|
| `onnxruntime_1.19.2_arm64.deb` | 8.9MB | ✅ `VSINPUExecutionProvider` 已编译在内 |
| `python3-onnxruntime_1.19.2_arm64.deb` | 5.1MB | ⚠️ Python 3.12 绑定, 与 Debian 3.11 不兼容 |
| `tim-vx_1.2.22_arm64.deb` | 216KB | ✅ |
| `tim-vx-tools_1.2.22_arm64.deb` | 643KB | ✅ |
| `x-linux-ai-benchmark_6.2.0_arm64.deb` | 71KB | ✅ |

### 3.3 为什么 Debian 12 上无法运行

```
libonnxruntime.so (含 VSINPU)
  → GLIBC_2.38 (fmod, __isoc23_strtoll)  ← Debian 12 只有 glibc 2.36
  → GLIBCXX_3.4.32                        ← Debian 12 只有 GCC 12.2
  → libtim-vx.so                          → libovxlib.so
    → libArchModelSw.so                   ← Debian 镜像中不存在
      → gcnano-driver-stm32mp              ← 仅 OpenSTLinux 提供
```

ST 的 NPU 软件栈深度绑定 OpenSTLinux BSP。替换个别 .so 无法解决——glibc 版本差异影响整个工具链。

---

## 4. 最终决策

**切换操作系统: Debian 12 → OpenSTLinux v6.2**

- OpenSTLinux v6.2 内置完整的 X-LINUX-AI v6.2、gcnano NPU 驱动、`onnxruntime` NPU 变体
- 迁移方案详见: `OS_MIGRATION_PLAN.md`
- 迁移窗口: ~4 小时（含烧录 + 环境重建 + 全链路验证）
- 回退方案: 使用 SD 卡烧录 OpenSTLinux，保留 eMMC 中的 Debian 系统
