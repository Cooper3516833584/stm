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

位于 `E:\files\嵌赛\stm32mp257\02-Images\8E2D\`：

### 2.1 预构建 SD 卡镜像（直接烧录用）

| 文件 | 路径 | 大小 | 说明 |
|------|------|------|------|
| **`FlashLayout_sdcard_myb-ld25x-8e2d-myir-image-burn.raw`** | `FlashLayout_sdcard_myb-ld25x-8e2d-myir-image-burn/` | **5.7 GB** | **SD 卡完整镜像，含所有分区（bootfs + vendorfs + rootfs + userfs）** |

> 这是预构建好的完整 SD 卡镜像，包含 GPT 分区表 + 所有固件，可直接用 balenaEtcher / Win32DiskImager / `dd` 写入 SD 卡。**大多数用户只需这一个文件。**

### 2.2 独立分区镜像（按需烧录/高级用）

位于 `myir-image-full/` 子目录：

| 文件 | 大小 | 说明 |
|------|------|------|
| `myir-image-full-openstlinux-weston-myd-ld25x.rootfs.ext4` | 4.0G | 根文件系统 (ext4) |
| `st-image-bootfs-openstlinux-weston-myd-ld25x.bootfs.ext4` | 64M | 启动分区 (kernel + device tree) |
| `st-image-vendorfs-openstlinux-weston-myd-ld25x.vendorfs.ext4` | 48M | Vendor 分区 (含 gcnano NPU 驱动) |
| `st-image-userfs-openstlinux-weston-myd-ld25x.userfs.ext4` | 128M | 用户数据分区 |
| `flashlayout_myir-image-full/optee/FlashLayout_sdcard_myb-stm32mp257x-2GB-ca35tdcid-ostl-optee.tsv` | — | 分区布局描述文件 (STM32CubeProgrammer 用) |
| `scripts/create_sdcard_from_flashlayout.sh` | — | 从 TSV 生成 .raw 的脚本 (Linux 用) |
| `arm-trusted-firmware/` | — | TF-A 固件 (fsbl) |
| `fip/` | — | FIP 固件 (U-Boot + OP-TEE) |
| `kernel/` | — | Linux kernel |
| `u-boot/` | — | U-Boot bootloader |

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

## 4. SD 卡烧录详细步骤

### 4.1 准备工作

#### 硬件清单

| 物品 | 要求 | 说明 |
|------|------|------|
| **microSD 卡** | **≥ 8 GB**, Class 10 / U1 或更高 | 镜像文件 5.7 GB，建议用 **16GB** 以上留余量 |
| **USB 读卡器** | 支持 microSD | PC 上写卡用 |
| **MYD-LD25X 开发板** | 断电状态 | 烧录时不要插 SD 卡 |
| **5V DC 电源适配器** | 开发板供电 | 烧录完成后上电启动 |

#### 软件准备

选择以下任一工具（Windows）：

| 工具 | 推荐度 | 下载 | 说明 |
|------|:---:|------|------|
| **balenaEtcher** | ⭐⭐⭐ | https://www.balena.io/etcher/ | 开源、UI 简洁、自动校验 |
| **Win32 Disk Imager** | ⭐⭐ | https://sourceforge.net/projects/win32diskimager/ | 经典工具 |
| **Rufus** | ⭐⭐ | https://rufus.ie/ | 更灵活，但需选 "DD 镜像" 模式 |

> Linux 用户可直接用 `dd`，见 [§4.3](#43-linux--wsl-用户-dd-方式)。

#### 镜像文件位置

```
E:\files\嵌赛\stm32mp257\02-Images\8E2D\FlashLayout_sdcard_myb-ld25x-8e2d-myir-image-burn\
  └── FlashLayout_sdcard_myb-ld25x-8e2d-myir-image-burn.raw  (5.7 GB)
```

---

### 4.2 方案 A：SD 卡烧录（推荐 ⭐ — 保留 eMMC Debian 作为回退）

这是最安全的方式。OpenSTLinux 烧在 SD 卡上，**不碰 eMMC 中的 Debian 系统**。
如果 SD 卡启动失败或 NPU 不工作，拔卡即可回到 Debian。

#### Step 1: 写入镜像到 SD 卡

**使用 balenaEtcher（推荐）：**

```
1. 打开 balenaEtcher
2. 点击 "Flash from file" → 选择：
   E:\files\嵌赛\stm32mp257\02-Images\8E2D\FlashLayout_sdcard_myb-ld25x-8e2d-myir-image-burn\
     FlashLayout_sdcard_myb-ld25x-8e2d-myir-image-burn.raw

3. 点击 "Select target" → 选择你的 SD 卡（注意：不要选错盘符！）
4. 点击 "Flash!" → 等待写入完成（约 10-20 分钟，取决于读卡器速度）
5. 等待 "Flash Complete!" 提示，balenaEtcher 会自动校验
```

**使用 Win32 Disk Imager：**

```
1. 打开 Win32DiskImager（以管理员身份运行）
2. Image File: 选择 .raw 文件（文件类型过滤选 *.* 才能看到 .raw）
3. Device: 选择 SD 卡盘符（如 E:\, F:\）
4. 点击 "Write"
5. 等待写入完成，点击 "Exit"
```

> ⚠️ **写入前确认**：选中的设备是 SD 卡，不是你的硬盘/U盘！写错设备会导致数据永久丢失。
>
> 如果不确定盘符，打开 Windows 磁盘管理（`diskmgmt.msc`），根据磁盘大小（~8GB/16GB/32GB）确认哪个是 SD 卡。

#### Step 2: 设置开发板从 SD 卡启动

MYD-LD25X 通过 **BOOT 拨码开关** 选择启动介质。

**MYD-LD25X 拨码开关设置（SD 卡启动）：**

```
           BOOT1    BOOT2
           ┌───┐    ┌───┐
           │ ON│    │OFF│
           │   │    │   │
           └───┘    └───┘
           ┌───┐    ┌───┐
           │OFF│    │OFF│
           └───┘    └───┘
```

> ⚠️ **拨码开关具体位置和编号可能因硬件版本而异，请以 MYD-LD25X Quick Start Guide 或开发板丝印为准。**
>
> 通常规律：
> - **eMMC 启动**（当前 Debian）：BOOT1=OFF, BOOT2=ON
> - **SD 卡启动**（OpenSTLinux）：BOOT1=ON, BOOT2=OFF
> - **USB/UART 烧录模式**：BOOT1=OFF, BOOT2=OFF

**操作步骤：**

```
1. 断开开发板电源（拔掉 DC 电源线）
2. 将写好的 microSD 卡插入开发板 SD 卡槽（J4 或标记为 "SD Card" 的槽位）
3. 调整 BOOT 拨码开关为 SD 卡启动模式
4. 重新接通电源
```

#### Step 3: 首次启动观察

**连接调试串口（可选但强烈推荐）：**

MYD-LD25X 通常有 USB-to-UART 调试接口，通过 microUSB 连接到 PC 可观察启动日志：

```
1. 用 microUSB 线连接开发板 J-Link/调试口 到 PC
2. Windows 设备管理器查看新增的 COM 端口
3. 打开串口终端（PuTTY / MobaXterm / Tera Term）：
   - 波特率: 115200
   - 数据位: 8
   - 停止位: 1
   - 无校验
   - 无流控
4. 给开发板上电，观察启动日志
```

**正常启动过程（约 30-60 秒）：**

```
1. TF-A (FSBL) 启动        ← "NOTICE:  Model: STMicroelectronics STM32MP257F-EV1..."
2. OP-TEE 初始化            ← "I/TC: ..."
3. U-Boot SPL → U-Boot     ← "U-Boot 2024..."
4. Linux kernel 加载        ← "Starting kernel ..."
5. 各分区挂载               ← bootfs, vendorfs, rootfs
6. systemd 服务启动         ← systemd 初始化
7. Weston (图形) 启动       ← 如果连接了 HDMI 显示器会看到桌面
8. 登录提示                ← "myd-ld25x login:"
```

**正常启动的可见信号：**

- 开发板电源 LED 常亮
- 约 10s 后 U-Boot 启动 LED 闪烁
- 约 30-60s 后系统就绪
- HDMI 显示器显示 Weston 桌面（如果接了）
- 以太网口 LED 闪烁（如果接了网线）

#### Step 4: 首次登录

**方式 1：调试串口（无需网络）**

```
串口终端中直接看到登录提示：
myd-ld25x login: root
（默认无密码，或见 Quick Start Guide）
```

**方式 2：SSH（需先配网络，见 §5.1）**

```
ssh root@<board_ip>
```

**方式 3：HDMI + USB 键鼠**

直接操作本地桌面终端。

---

### 4.3 Linux / WSL 用户：`dd` 方式

如果你有 Linux 环境（或 WSL2 挂载了物理磁盘），可以直接用 `dd`：

```bash
# 1. 确认 SD 卡设备名（插入前后对比 lsblk）
lsblk
# 假设 SD 卡是 /dev/sdb（根据大小确认，16GB/32GB 的设备）

# 2. 卸载 SD 卡所有已挂载分区（如有）
sudo umount /dev/sdb* 2>/dev/null

# 3. 写入镜像
sudo dd if="E:/files/嵌赛/stm32mp257/02-Images/8E2D/FlashLayout_sdcard_myb-ld25x-8e2d-myir-image-burn/FlashLayout_sdcard_myb-ld25x-8e2d-myir-image-burn.raw" \
        of=/dev/sdb bs=8M conv=fdatasync status=progress

# 注意：of=/dev/sdb（整个磁盘），不是 /dev/sdb1（分区）！

# 4. 同步并校验
sync
sudo sgdisk /dev/sdb -v   # 验证 GPT 分区表完整性
```

> ⚠️ **`of=` 参数必须写对整个磁盘设备（如 /dev/sdb），写错会导致数据永久丢失！**
>
> 可以用 `lsblk -o NAME,SIZE,TYPE,MOUNTPOINT` 在插卡前后各执行一次来确认新增的设备名。

---

### 4.4 方案 B：直接烧录 eMMC（无回退，仅高级用户）

此方案把 OpenSTLinux 直接写入 eMMC，**覆盖 Debian 系统**。仅在确定不回头时使用。

#### 方式 1：通过 SD 卡中转（推荐）

```
1. 先在 SD 卡上启动 OpenSTLinux（方案 A）
2. 在 OpenSTLinux 系统内，把 .raw 文件复制到 SD 卡或 tmpfs
3. 用 dd 写入 eMMC：
   dd if=FlashLayout_*.raw of=/dev/mmcblk1 bs=8M conv=fdatasync status=progress
   # mmcblk1 是 eMMC, mmcblk0 是 SD 卡
4. 关机，拔出 SD 卡，切换拨码回 eMMC 启动
```

#### 方式 2：STM32CubeProgrammer（USB OTG 烧录）

```
1. 安装 STM32CubeProgrammer (Windows/Linux)
   https://www.st.com/en/development-tools/stm32cubeprog.html

2. 设置拨码开关为 USB/UART 烧录模式 (BOOT1=OFF, BOOT2=OFF)

3. 用 USB Type-C 线连接开发板 USB OTG 口到 PC

4. 打开 STM32CubeProgrammer，选择：
   - Port: USB
   - Flash layout: myir-image-full/flashlayout_myir-image-full/optee/
                    FlashLayout_sdcard_myb-stm32mp257x-2GB-ca35tdcid-ostl-optee.tsv
   - 修改 TSV 中的 IP 列为 mmc1（eMMC）或用 emmc 版本的 TSV

5. 点击 "Download" 开始烧录
```

---

### 4.5 常见问题排查

#### Q1: balenaEtcher 报 "Something went wrong" 写入失败

```
- 换一个 USB 读卡器（部分廉价读卡器对大文件不稳定）
- 换一张 SD 卡（卡可能损坏）
- 尝试 Win32 Disk Imager 或 Rufus
- Windows 上确保以管理员身份运行
```

#### Q2: 写入后 SD 卡在 Windows 上显示 "需要格式化"

```
这是正常现象！SD 卡上写入了 Linux ext4 文件系统，Windows 无法识别。
点击 "取消"，不要把卡格式化。
```

#### Q3: 插入 SD 卡后开发板不启动（无任何输出）

```
- 确认拨码开关设置为 SD 卡启动
- 确认 SD 卡完全插入卡槽（卡入到位，有 "咔嗒" 声）
- 检查电源适配器（5V, ≥2A）
- 连接调试串口观察有无 TF-A/U-Boot 输出
  如果完全没有输出 → 拨码开关或硬件问题
  如果有 TF-A 输出但卡住 → 固件不匹配
```

#### Q4: U-Boot 启动但 Kernel 加载失败

```
调试串口通常会打印错误原因：
- "MMC: no card present" → SD 卡未识别，换卡或清洁触点
- "Bad Linux ARM64 Image magic" → 镜像损坏，重新写入
- "Unable to mount root fs" → rootfs 分区损坏，重新写入
```

#### Q5: 启动成功但根文件系统是只读的

```
# ext4 可能因 unclean unmount 自动进入只读模式
mount -o remount,rw /

# 如果报 I/O error，SD 卡可能正在损坏
dmesg | grep -i "mmc\|error"
```

#### Q6: 烧录后如何回到 Debian？

```
1. 断开电源
2. 拔出 SD 卡
3. 将 BOOT 拨码开关切回 eMMC 启动模式
4. 上电启动 → 回到原来的 Debian 12 系统
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
