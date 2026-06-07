# 操作系统迁移方案：Debian 12 → OpenSTLinux v6.2

**日期**: 2026-06-07  
**决策**: 将 MYD-LD25X 操作系统从 Debian 12 Bookworm 切换至 OpenSTLinux v6.2 (Yocto Scarthgap)，以获得 NPU 硬件加速支持
**预计耗时**: 4-6 小时（含烧录 + 环境重建 + 全链路验证）

---

## 1. 迁移原因

### 1.1 根本原因

道路视觉感知管线的 ONNX 推理在 Cortex-A35 CPU 上仅 **0.6 FPS**（每帧 ~1800ms），完全不满足飞行实时性要求。STM32MP257 内置 VeriSilicon NPU（VIP9000），预期加速 100-500×，使 YOLO11-seg 推理降至 5-15ms。

### 1.2 NPU 适配为何无法在 Debian 12 上完成

经过 2026-06-07 全天诊断与尝试，结论是**ST 的 NPU 软件栈深度绑定了 OpenSTLinux 的 BSP**：

| 依赖层 | Debian 12 | OpenSTLinux v6.2 | 兼容？ |
|------|------|------|:--:|
| glibc | 2.36 | 2.39 | ❌ |
| libstdc++ | GCC 12.2 (GLIBCXX_3.4.30) | GCC 13.4 (GLIBCXX_3.4.32) | ❌ |
| GPU/NPU 用户态驱动 | 仅有基础 `libGAL.so`, `libVSC.so` | 完整 `gcnano-driver-stm32mp` + `gcnano-userland-multi-binary` | ❌ |
| `libArchModelSw.so` | 不存在 | vendorfs 内置 | ❌ |
| `libovxlib.so` (OpenVX NPU 后端) | 不存在 | vendorfs 内置 | ❌ |
| `libopenvx-gcnano` | 不存在 | vendorfs 内置 | ❌ |
| APT 仓库 | Debian mirrors | `packages.openstlinux.st.com` + `extra.packages.openstlinux.st.com/AINPU` | ❌ |
| Python | 3.11 | 3.12 | ⚠️ |

简单替换 `.so` 文件不可行——`libonnxruntime.so` 需要 `GLIBC_2.38` (`fmod`, `__isoc23_strtoll` 等)，而 Debian 12 的 glibc 2.36 不提供这些符号。底层 glibc 升级会连锁触发整个系统工具链的重建。

### 1.3 已验证但不可行的尝试

- [x] 从 `vendorfs` 提取 NPU 用户态库 (`libGAL.so`, `libVSC.so` 等 12 个) → 已安装到系统
- [x] 从 ST APT 仓库下载 `onnxruntime_1.19.2_arm64.deb` (含 `VsiNpuExecutionProvider`) → 已确认 VSINPU 符号存在
- [x] 从 SDK 和 rootfs 提取 `libovxlib.so`, `libOpenVX.so`, `libtim-vx.so` → 已提取
- [x] 从 rootfs 提取 OpenSTLinux glibc/libstdc++ → `LD_LIBRARY_PATH` 隔离尝试
- [x] `LD_PRELOAD` 加载 `libovxlib.so` → 触发 `libArchModelSw.so` 缺失
- [ ] 用 `patchelf` 降级符号版本 → 未进行（`libArchModelSw.so` 等 BSP 依赖同样需要 gcnano 驱动栈）
- [ ] 联系 MYiR FAE 获取 Debian 兼容 NPU 包 → 待确认，但 Debian 12 的 BSP 可能无 NPU 支持计划

**结论**: 直接切换 OpenSTLinux 是最快且唯一可靠的路径。OpenSTLinux v6.2 内置完整的 X-LINUX-AI v6.2 + gcnano NPU 驱动 + `onnxruntime` NPU 变体。

---

## 2. 可用镜像文件

位于 `E:\files\嵌赛\stm32mp257\02-Images\8E2D\myir-image-full\`：

| 文件 | 大小 | 说明 |
|------|------|------|
| `myir-image-full-openstlinux-weston-myd-ld25x.rootfs.ext4` | 4.0G | 根文件系统 (ext4) |
| `st-image-bootfs-openstlinux-weston-myd-ld25x.bootfs.ext4` | 64M | 启动分区 |
| `st-image-vendorfs-openstlinux-weston-myd-ld25x.vendorfs.ext4` | 48M | Vendor 分区 (含 gcnano NPU 驱动) |
| `FlashLayout_sdcard_myb-ld25x-8e2d-myir-image-full/*.raw` | — | SD 卡烧录镜像 (可直接 `dd` 到 eMMC 或 SD) |

**建议使用 SD 卡烧录方式** — 保留当前 Debian eMMC 作为回退。

---

## 3. 预迁移备份（CRITICAL — 先做这一步）

在烧录前，将 Debian 系统中**不可替代的配置和数据**备份到 SD 卡或 PC：

```bash
# 在开发板上执行
# 1. 备份串口设备绑定（udev 规则）
sudo cp -r /etc/udev/rules.d/ /media/sdcard/backup_udev/

# 2. 备份 fstab（确认 SD 卡挂载配置）
cp /etc/fstab /media/sdcard/backup_fstab

# 3. 备份完整项目代码（已在 SD 卡上，验证即可）
ls -la /media/sdcard/ObstacleAvoidanceDrone/

# 4. 备份 HARDWARE_INTERFACE.md 中记录的接线信息
#    (已在 git 仓库中，pull 到 PC 即可)

# 5. 记录当前设备路径映射
ls -la /dev/serial/by-id/ > /media/sdcard/backup_serial_by_id.txt
ls -la /dev/ttySTM* > /media/sdcard/backup_ttySTM.txt
ls -la /dev/video* > /media/sdcard/backup_video.txt
dmesg | grep -i "tty\|video\|galcore\|usb" > /media/sdcard/backup_dmesg.txt

# 6. 备份 SSH host key（可选，避免 known_hosts 警告）
sudo cp /etc/ssh/ssh_host_* /media/sdcard/backup_ssh/

# 7. 将关键文件拉回 PC
# 在 PC Git Bash 上:
# scp stm@myd-ld25x:/media/sdcard/backup_* /c/Users/DELL/Desktop/debian_backup/
```

---

## 4. 烧录步骤

### 4.1 方案 A：SD 卡烧录（推荐 — 保留 eMMC Debian）

```bash
# 1. 在 Windows 上用 Win32 DiskImager 或 balenaEtcher
#    将 FlashLayout_sdcard_myb-ld25x-8e2d-myir-image-full.raw
#    写入一张 ≥8GB 的 microSD 卡

# 2. 将 SD 卡插入开发板 SD 卡槽

# 3. 设置拨码开关从 SD 卡启动
#    (参考 MYD-LD25X Quick Start Guide)
```

### 4.2 方案 B：直接烧录 eMMC（无回退）

```bash
# 1. 将 raw 镜像复制到开发板 SD 卡
# 在 PC 上解压 FlashLayout zip → 复制 .raw 文件到 SD 卡

# 2. 在开发板上通过 U-Boot 或 Linux 烧录
#    (需参考 MYD-LD25X Linux Software Development Guide)

# 或使用 STM32CubeProgrammer 通过 USB OTG 烧录
```

**推荐方案 A** — 如果 SD 卡启动有问题可以随时拔卡回到 Debian eMMC。

---

## 5. 首次启动与基础配置

### 5.1 网络配置

```bash
# OpenSTLinux 默认用户名通常为 "root"，具体查看 Quick Start Guide
# 连接 WiFi:
connmanctl
> enable wifi
> scan wifi
> services
> agent on
> connect <SSID>
# (输入密码后退出: quit)

# 查看 IP:
ip addr show wlan0

# 如果使用 SSH:
# 从 PC: ssh root@<board_ip>
```

### 5.2 存储配置

```bash
# 1. 确认 SD 卡挂载
lsblk
# 30G 外置卡应为 mmcblk0

# 2. 如果未自动挂载，手动挂载
sudo mkdir -p /media/sdcard
sudo mount /dev/mmcblk0p1 /media/sdcard

# 3. 配置 fstab 自动挂载
echo "/dev/mmcblk0p1 /media/sdcard vfat rw,uid=root,gid=root,noatime 0 0" | sudo tee -a /etc/fstab
```

---

## 6. Python 环境重建

### 6.1 创建虚拟环境

OpenSTLinux v6.2 自带 Python 3.12 + pip + numpy + opencv 预编译包：

```bash
# 1. 确认 Python 版本
python3 --version  # 应为 3.12.x

# 2. 在 SD 卡上创建新的虚拟环境（不占 eMMC）
python3 -m venv --system-site-packages /media/sdcard/venv_npu
source /media/sdcard/venv_npu/bin/activate

# 3. 升级 pip
pip install --upgrade pip
```

### 6.2 安装项目依赖

OpenSTLinux 已通过 APT 提供的包（使用 `--system-site-packages` 继承）：

| 包 | 安装方式 | 说明 |
|------|------|------|
| `numpy` | APT (`opkg` / `apt-get`) | 系统预装 |
| `opencv` | APT | 系统预装 |
| `onnxruntime` (NPU 版) | `apt-get install onnxruntime python3-onnxruntime` | 含 `VsiNpuExecutionProvider` |
| `loguru` | pip | 纯 Python，无 C 扩展 |
| `pyserial` | pip | — |
| `simple-pid` | pip | — |

```bash
source /media/sdcard/venv_npu/bin/activate

# OpenSTLinux 用 `apt-get` 装系统包（非 `apt`）
sudo apt-get update
sudo apt-get install -y onnxruntime python3-onnxruntime tim-vx tim-vx-tools x-linux-ai-benchmark config-npu

# pip 安装纯 Python 依赖
pip install loguru==0.7.3 pyserial==3.5 simple-pid==2.0.1

# ⚠️ numpy 版本约束（防止 NumPy 2.x API 断裂）
# OpenSTLinux 系统级 numpy 可能是 1.x，确认后无需额外安装
pip install "numpy<2.0.0" --only-binary :all: 2>/dev/null || echo "Using system numpy"
```

### 6.3 代码仓库克隆

```bash
# 从 GitHub 重新 clone（或从 SD 卡已有仓库复制）
cd /media/sdcard/
git clone git@github.com:Cooper3516833584/stm.git ObstacleAvoidanceDrone

# 或直接从旧 SD 卡复制（如果未格式化）
# cp -r /media/sdcard/ObstacleAvoidanceDrone /media/sdcard/ObstacleAvoidanceDrone

# 创建软链接保持原路径兼容
mkdir -p ~/Desktop
ln -sf /media/sdcard/ObstacleAvoidanceDrone ~/Desktop/ObstacleAvoidanceDrone
```

### 6.4 串口权限

```bash
sudo usermod -aG dialout $USER
# 重新登录生效
```

---

## 7. 验证序列（逐阶段检查）

### 阶段 1：NPU 可用性 ⭐ 最高优先级

```bash
source /media/sdcard/venv_npu/bin/activate
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

**预期输出**（至少包含一项 NPU）:
```
['VsiNpuExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']
```

**如果未出现 `VsiNpu`**: 
```bash
# 诊断
find /usr/lib -name "libonnxruntime*" -o -name "libtim-vx*" -o -name "libovxlib*" 2>/dev/null
strings /usr/lib/libonnxruntime.so.* 2>/dev/null | grep -i vsinpu | head -5
```

### 阶段 2：视觉推理性能基准

```bash
cd ~/Desktop/ObstacleAvoidanceDrone
source /media/sdcard/venv_npu/bin/activate

PYTHONPATH=. python FlightController/tools/bench_vision_fps.py --frames 30
```

**预期输出**（NPU 加速后）:
```
perception p50:   5-15ms   (vs Debian CPU 1800ms)
FPS:              60-200    (受摄像头 30fps 限制时稳定 30)
```

**如果 < 5 FPS**: 检查 provider 是否为 `VsiNpuExecutionProvider`，而非 `CPUExecutionProvider`。

### 阶段 3：雷达链路

```bash
cd ~/Desktop/ObstacleAvoidanceDrone
source /media/sdcard/venv_npu/bin/activate

# 3a. 查看串口设备是否与 Debian 一致
ls -la /dev/ttySTM* /dev/serial/by-id/

# 3b. 双雷达烟雾测试
PYTHONPATH=. python FlightController/tools/smoke_dual_radar.py \
    --upper-port /dev/ttySTM4 --lower-port /dev/ttySTM9

# 3c. 如果串口设备名变了，更新 HARDWARE_INTERFACE.md 并排查 udev 规则
```

### 阶段 4：飞控通信

```bash
cd ~/Desktop/ObstacleAvoidanceDrone
source /media/sdcard/venv_npu/bin/activate

# 4a. FC 连通性
PYTHONPATH=. python debug/test_fc_connect.py

# 4b. FC 模式切换
PYTHONPATH=. python debug/test_fc_command.py --target-mode 2
```

### 阶段 5：双雷达 + FC 并行运行

```bash
PYTHONPATH=. python -u FlightController/tools/test_dual_radar.py --no-fc --loop-hz 30
```

**预期**: 设备钟速 100%，CRC 错误 0，CPU ~15%。

### 阶段 6：摄像头

```bash
# 6a. 查看摄像头设备
ls -la /dev/video*

# 6b. 如果设备号变了（debian 上 cam#7 / cam#9）
#     用 v4l2-ctl 确认
v4l2-ctl --list-devices 2>/dev/null || ls /dev/v4l/by-id/

# 6c. OpenCV 摄像头探活
python -c "
import cv2
for idx in range(10):
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f'cam index {idx}: {frame.shape} OK')
        cap.release()
"
```

### 阶段 7：NPU 推理 + 双雷达 + FC 全链路

```bash
cd ~/Desktop/ObstacleAvoidanceDrone
source /media/sdcard/venv_npu/bin/activate

# road_follow_main 测试（dry-run 模式，不下发速度指令）
PYTHONPATH=. python road_follow_main.py --dry-run --camera-index <VALUE> --loop-hz 10
```

### 阶段 8：eMMC 空间确认

```bash
df -h /
df -h /media/sdcard/
```

---

## 8. 已知差异与注意事项

### 8.1 Python 版本差异

| | Debian 12 | OpenSTLinux v6.2 |
|------|------|------|
| Python | 3.11 | 3.12 |
| numpy (系统) | 1.24.x (APT) | 1.26.x (Yocto) |
| onnxruntime 安装 | `pip install onnxruntime` (CPU only) | `apt-get install python3-onnxruntime` (NPU) |

⚠️ `pyproject.toml` 中 `requires-python = ">=3.10"` 兼容 3.12，无需修改。
⚠️ `numpy<2.0.0` 约束继续有效——ONNX Runtime 1.19.x 仍依赖 NumPy 1.x ABI。

### 8.2 串口设备名可能改变

OpenSTLinux 的设备树可能给 UART 分配不同的 `ttySTMx` 编号。启动后用以下命令确认：

```bash
ls /dev/ttySTM*
dmesg | grep tty
```

如果设备名变了，更新 `HARDWARE_INTERFACE.md` 第 4 节和 `goal_nav_main.py` / `road_follow_main.py` 的默认值。

### 8.3 图形界面 (Weston)

OpenSTLinux v6.2 默认启动 Weston (Wayland 合成器)，占用 RAM。若需 Headless 模式：

```bash
systemctl stop weston
systemctl disable weston
```

### 8.4 loguru 日志路径

OpenSTLinux 的文件系统布局与 Debian 略有不同。首次运行时确认 `--log-file` 路径可写。

### 8.5 包管理器差异

OpenSTLinux 使用 `apt-get`（基于 Yocto 编译的 `.deb` 包），但仓库是 `packages.openstlinux.st.com`，非标准 Debian。需要用 `opkg` 作为备选。

---

## 9. 回退方案

如迁移后 NPU 仍不可用或系统无法正常工作：

1. 拔出 OpenSTLinux SD 卡
2. 开发板自动从 eMMC Debian 启动
3. Debian 环境和代码仓库完整保留

**如果已烧录 eMMC（方案 B）**，需用 `STM32CubeProgrammer` 或 SD 卡烧录工具重新烧录 Debian 镜像至 eMMC。

---

## 10. 时间线预估

| 阶段 | 耗时 | 说明 |
|------|:--:|------|
| 备份 Debian 配置 | 15 min | 串口设备列表、fstab、SSH key |
| 制作 SD 卡 / 烧录 | 30 min | 取决于镜像大小和写入速度 |
| 首次启动 + 网络配置 | 15 min | WiFi 连接、SSH 确认 |
| Python 环境重建 | 30 min | venv + pip install |
| NPU 验证 | 15 min | `onnxruntime.get_available_providers()` |
| 视觉推理基准 | 15 min | `bench_vision_fps.py` |
| 雷达 + FC 验证 | 30 min | 烟雾测试 + 全链路 |
| 摄像头验证 | 15 min | 设备探活 + 帧捕获 |
| 文档更新 | 30 min | 更新 HARDWARE_INTERFACE.md 设备路径 |
| **总计** | **~3-4h** | 含调试时间 |

---

## 11. 参考

- MYD-LD25X Quick Start Guide (在 SDK `01-Docs/` 中)
- MYD-LD25X Linux Software Development Guide
- X-LINUX-AI Application Note (在 SDK `01-Docs(ZH)/应用笔记/` 中)
- ST Wiki: https://wiki.st.com/stm32mpu/wiki/X-LINUX-AI_OpenSTLinux_Expansion_Package
- 项目 NPU 需求分析: `NPU_REQUIREMENTS.md`
- 硬件接线文档: `HARDWARE_INTERFACE.md`
