# MYD-LD25x 系统重烧录与项目恢复清单

本文记录新 SD 卡重新烧录系统后，已经完成、已经验证和尚待完成的工作，作为后续恢复
`ObstacleAvoidanceDrone` 运行环境的依据。

当前恢复范围已经明确：

- 使用现有蓝牙 NAP/NAT 作为主要联网与 SSH 通道，不再把 WiFi 恢复列为当前任务。
- 后续主要操作新 SD 卡。
- eMMC 暂不启用、不恢复、不自动同步，也不作为当前备用启动目标。
- 分区、安装软件、恢复项目和写入系统配置均需单独批准。

## 1. 背景

- 旧 SD 卡发生物理开裂，随后文件损坏并无法正常启动；除物理损坏外，其他潜在损坏未知。
- 旧卡最后一次向板端 eMMC 烧录系统后，曾再次从旧卡启动并成功连接 WiFi，当时未出现异常。
- 后续从 eMMC 启动时也观察到 WiFi 问题，因此不能仅凭“更换 SD 卡后出现问题”断定新卡或
  新烧录系统是唯一原因。
- 新 SD 卡使用与旧卡相同的 RAW 文件重新烧录：

```text
D:\drone\嵌赛\stm32mp2-01-Docs(ZH)\8E2D\
FlashLayout_sdcard_myb-ld25x-8e2d-myir-image-full\
FlashLayout_sdcard_myb-ld25x-8e2d-myir-image-full.raw
```

RAW 文件信息：

```text
大小：6028263424 bytes（约 5.614 GiB）
```

## 2. 当前系统基线

当前从新 SD 卡启动，实测系统信息：

```text
系统：ST OpenSTLinux - Weston 5.0.3-snapshot-20250320
Yocto 分支：scarthgap
架构：aarch64
内核：6.6.48-gbebcf479fd77
根分区：/dev/mmcblk0p10（SD 卡 rootfs）
/usr/local：/dev/mmcblk0p11（SD 卡 userfs）
```

启动参数中的 root PARTUUID 与 `/dev/mmcblk0p10` 匹配，确认当前不是从 eMMC rootfs 启动。

### 2.1 SD 卡分区与容量

```text
物理设备：/dev/mmcblk0，约 29.1G
bootfs：/dev/mmcblk0p8，64M，挂载到 /boot
vendorfs：/dev/mmcblk0p9，250M，挂载到 /vendor
rootfs：/dev/mmcblk0p10，4G，挂载到 /
userfs：/dev/mmcblk0p11，1.3G，挂载到 /usr/local
```

当前空间：

```text
/:          3.8G 总量，约 977M 已用，约 2.6G 可用
/usr/local: 1.3G 总量，约 17M 已用，约 1.2G 可用
```

29.1G 卡上现有 RAW 分区合计只覆盖约 5.6G，剩余大部分容量尚未纳入当前 rootfs/userfs 或项目
数据路径。恢复仓库、虚拟环境和模型前，应先决定扩展 `userfs` 还是新建独立数据分区，避免再次
出现空间不足或把日志写入 RAM 文件系统。

### 2.2 eMMC 边界

eMMC 为 `/dev/mmcblk1`，约 7.3G。当前系统能发现其原有分区，但检查结果为：

```text
旧项目目录：不存在
旧 UFC_venv：不存在
本次 myir-bt-nap.service：不存在
本次 bnep.ko：不存在
```

当前未发现指向 `mmcblk1` 的自动烧录 systemd/init 服务，本次恢复没有自动把新 SD 卡内容同步
到 eMMC。

当前决策是后续主要使用 SD 卡，eMMC 不列入恢复流程：

- 不向 eMMC 同步 BNEP、蓝牙 NAP、项目仓库或虚拟环境。
- 不验证 eMMC 启动。
- 不把 eMMC 分区作为项目数据目录。
- 除非未来重新明确提出，否则保持 eMMC 现状且不执行写入。

## 3. 已经完成并验证

### 3.1 基础系统

- [x] 将指定 RAW 镜像烧录到新 SD 卡。
- [x] 开发板可以从新 SD 卡正常启动并进入 root 串口终端。
- [x] 确认启动设备为 SD 卡 `/dev/mmcblk0p10`。
- [x] 确认 `/usr/local` 使用 SD 卡 `/dev/mmcblk0p11`。
- [x] 确认 Python 3 基础解释器存在：`/usr/bin/python3`。
- [x] 确认系统提供 `apt`；未发现 `opkg`。

### 3.2 基础硬件和 NPU 内核层

- [x] `/dev/galcore` 存在，权限为 `root:video`。
- [x] `galcore` 内核模块已加载。
- [x] 串口节点 `/dev/ttySTM4`、`/dev/ttySTM9` 存在，权限为 `root:dialout`。
- [ ] 尚未进行重烧录后的 NPU 用户态推理和真实硬件数据链路测试；设备节点存在不等于项目功能
  已恢复。

### 3.3 蓝牙联网

- [x] 确认原内核未启用 `CONFIG_BT_BNEP`，板端没有现成 `bnep.ko`。
- [x] 使用匹配的厂商 SDK、内核配置和 `Module.symvers` 单独构建 BNEP 模块。
- [x] 将 `bnep.ko` 安装到当前 SD 卡 rootfs，执行 `depmod` 并成功加载。
- [x] 创建并启用 `myir-bt-nap.service`，由开发板提供 NAP。
- [x] Windows 作为 PANU，通过“连接方式 → 接入点”建立 BNEP 链路。
- [x] Windows 蓝牙接口配置为 `192.168.137.1/24`。
- [x] Windows 创建 `MYIR-Bluetooth-NAT`，内部网段为 `192.168.137.0/24`。
- [x] 开发板 `pan0` 配置为 `192.168.137.2/24`，默认路由经 PC。
- [x] 验证开发板可访问公网 IP 并解析域名。

完整过程见 [BLUETOOTH_PAN_HOST_NETWORKING.md](BLUETOOTH_PAN_HOST_NETWORKING.md)。

### 3.4 SSH

- [x] Dropbear 已安装。
- [x] `dropbear.socket` 为 `enabled/active`，监听 TCP 22。
- [x] 蓝牙网络内开发板 IP 固定为 `192.168.137.2`。
- [x] SSH 目标命令为 `ssh root@192.168.137.2`。
- [ ] 尚未执行开发板和 PC 的完整重启验证；重启后 Windows 可能需要重新选择一次“连接方式 →
  接入点”。

### 3.5 WiFi 现状与决策

重烧录后的实测状态：

```text
wlan0：DOWN
当前 WiFi：未连接
wpa_supplicant@wlan0.service：disabled/inactive
/etc/wpa_supplicant：没有接口配置文件
历史固定地址 192.168.31.199：当前未使用
```

- [x] 已确认当前无 WiFi 配置和开机自连。
- [x] 当前不再恢复 WiFi，以已经可用的蓝牙 NAP/NAT 作为主要网络通道。
- [ ] WiFi 不属于当前恢复待办；只有未来明确提出时再重新评估和配置。

## 4. 当前缺失的项目环境

当前 SD 卡 `/usr/local` 检查结果：

```text
/usr/local/ObstacleAvoidanceDrone：不存在
/usr/local/UFC_venv：不存在
git：未安装
```

系统 Python 当前未发现以下项目依赖：

```text
numpy
cv2 / OpenCV
scipy
pyserial
loguru
simple_pid
onnxruntime
matplotlib
```

因此当前只达到“基础系统可启动、可通过蓝牙联网和 SSH 管理”，尚不能运行项目程序。

## 5. 待完成事项

### P0：SD 卡空间方案

- [ ] 备份当前 SD 卡分区表和关键系统配置。
- [ ] 确认未分配空间的准确范围。
- [ ] 在扩展 `/usr/local` 所在 `userfs` 和新建独立项目数据分区之间选择。
- [ ] 明确项目仓库、虚拟环境、模型和日志分别放在哪个持久分区。
- [ ] 分区修改完成后检查文件系统和挂载持久化。

任何分区或文件系统修改均需单独批准。该步骤应先于安装依赖和复制模型。

### P1：恢复项目运行环境

- [ ] 经批准后安装 `git`。
- [ ] 从本地/云端 Git 仓库把已批准版本恢复到 `/usr/local/ObstacleAvoidanceDrone` 或最终选定的
  SD 卡持久目录。
- [ ] 不在板端直接编辑项目源码；板端不得向远端 push。
- [ ] 在 `/usr/local/UFC_venv` 或最终选定的 SD 卡路径重建虚拟环境。
- [ ] 恢复 NumPy、OpenCV、SciPy、PySerial、Loguru、simple-pid、Matplotlib 和 ONNX Runtime。
- [ ] 优先使用匹配系统 ABI 的预编译包，避免在 2GB RAM 板端大规模现场编译。
- [ ] 恢复项目所需模型、配置和非 Git 大文件，并逐项核对来源与版本。
- [ ] 确认项目日志和数据目录位于 SD 卡持久存储，禁止把大量数据写入 `/tmp`。

### P2：恢复后验证

- [ ] 运行无硬件导入/环境检查，确认 Python ABI 和原生库加载正常。
- [ ] 验证 ONNX Runtime 可用 provider，并重新执行已知 NPU 基线模型测试；当前只确认 galcore
  内核层存在。
- [ ] 验证 `/dev/ttySTM4`、`/dev/ttySTM9` 的真实雷达数据和飞控串口通信。
- [ ] 验证摄像头稳定设备路径、采集格式和双摄映射。
- [ ] 依次运行项目的 no-hardware、serial、radar、camera/NPU smoke tests。
- [ ] 在不启动实际飞行输出的条件下完成主程序 dry-run。

### P3：蓝牙持久化验证

- [ ] 仅重启开发板，验证 BNEP 模块、NAP 服务、`pan0` 地址、默认路由和 Dropbear。
- [ ] 重启 PC，验证静态蓝牙地址和 `MYIR-Bluetooth-NAT` 是否保留。
- [ ] 记录开发板重启和 PC 重启后是否必须手动重新选择“连接方式 → 接入点”。
- [ ] 验证 PC 更换上游 WiFi、有线网络或手机热点后，开发板仍能通过蓝牙访问网络。
- [ ] 确认 PC 新接入的网络不与蓝牙网段 `192.168.137.0/24` 冲突。

## 6. 当前不做的事项

- [x] 不恢复板端 WiFi，不配置 WiFi 开机自连。
- [x] 不恢复历史固定 WiFi 地址 `192.168.31.199`。
- [x] 不同步或重烧 eMMC。
- [x] 不从 eMMC 启动验证。
- [x] 不把 eMMC 用作项目或日志存储。

这些事项只有在未来需求发生变化并重新获得批准后才重新进入计划。

## 7. 建议执行顺序

```text
1. 备份 SD 卡分区表和关键配置
2. 决定并完成 SD 卡剩余空间布局
3. 安装 Git
4. 恢复项目仓库
5. 重建 UFC_venv 和依赖
6. 恢复模型、配置和数据目录
7. 执行无硬件检查
8. 执行 NPU、串口、雷达和相机硬件检查
9. 完成开发板/PC 蓝牙重启与上游网络切换验证
```

先处理存储布局，再恢复虚拟环境和模型，可以避免在当前 1.3G `userfs` 中途耗尽空间。

## 8. 恢复完成判定

满足以下条件后，才可将本次系统恢复标记为完成：

- [ ] SD 卡启动、根分区和项目数据分区稳定。
- [ ] Git 仓库、虚拟环境、依赖、模型及配置全部恢复并可复现。
- [ ] 无硬件测试和必要硬件 smoke tests 通过。
- [ ] NPU 用户态推理链路通过已知基线模型验证。
- [ ] 蓝牙 NAP/NAT 和 SSH 重启行为已验证。
- [ ] PC 切换上游网络后，板端仍能通过蓝牙联网。
- [ ] 项目日志不会写入 `/tmp` 或填满 rootfs/userfs。
- [ ] 文档明确 WiFi 和 eMMC 当前不在使用范围内。
