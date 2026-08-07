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

## 2. 当前系统与存储基线

```text
系统：ST OpenSTLinux - Weston 5.0.3-snapshot-20250320
Yocto 分支：scarthgap
架构：aarch64
内核：6.6.48-gbebcf479fd77
物理 SD 卡：/dev/mmcblk0，约 29.1G
根分区：/dev/mmcblk0p10，约 4G，挂载到 /
原 userfs：/dev/mmcblk0p11，约 1.3G，挂载到 /usr/local
项目数据分区：/dev/mmcblk0p12，约 23G，挂载到 /data
```

启动参数中的 root PARTUUID 与 `/dev/mmcblk0p10` 匹配，当前不是从 eMMC 启动。最终空间检查：

```text
/:      约 3.8G，总用量约 1.2G，可用约 2.4G（约 35% 已用）
/data:  约 23G，总用量约 306M，可用约 23G（约 2% 已用）
```

代码、虚拟环境、模型、日志和较大数据必须优先写入 `/data`；不要把项目移回较小的 rootfs 或
1.3G userfs，也不要把持续日志、抓包或模型写入 `/tmp`（RAM tmpfs）。恢复过程产生的抓包文件
已经清理。

### 2.1 当前项目路径

```text
/data/ObstacleAvoidanceDrone
/usr/local/ObstacleAvoidanceDrone -> /data/ObstacleAvoidanceDrone
/data/UFC_venv
/usr/local/UFC_venv -> /data/UFC_venv
```

符号链接保留旧脚本对 `/usr/local` 路径的兼容性，实际空间消耗落在 `/data`。`git` 已安装，
远端 `origin` 已通过纯蓝牙网络完成只读验证。

### 2.2 软件源顺序

下载板端软件包时使用以下优先级：

1. 中科大源：`https://mirrors.ustc.edu.cn`；
2. 清华源：`https://mirrors.tuna.tsinghua.edu.cn`；
3. 阿里源：`https://mirrors.aliyun.com`；
4. 发行版原默认源作为最后回退。

不执行全系统 `upgrade`；安装后清理包缓存，并在下载前后检查 `/` 与 `/data` 剩余容量。

### 2.3 eMMC 边界

eMMC 为 `/dev/mmcblk1`，约 7.3G。本次恢复没有把 SD 卡内容同步到 eMMC。继续保持：不写入、
不重烧、不验证 eMMC 启动、不把 eMMC 用作项目或日志目录。除非未来重新明确授权，否则不改变。

## 3. 已完成并验证

### 3.1 基础系统、项目和模型

- [x] 新 SD 卡完成 RAW 镜像烧录，并从 `/dev/mmcblk0p10` 正常启动。
- [x] 新建并持久挂载约 23G 的 `/data`（`/dev/mmcblk0p12`）。
- [x] 项目恢复到 `/data/ObstacleAvoidanceDrone`，兼容链接位于 `/usr/local`。
- [x] `UFC_venv` 及项目 Python 依赖恢复到 `/data/UFC_venv`，兼容链接位于 `/usr/local`。
- [x] `git` 已安装并能读取远端仓库；板端不 push，且未明确要求时不执行 `git pull`。
- [x] 仅保留代码实际使用的
  `FlightController/Solutions/model/new_road_seg_v5_final_fp32.nb` 和 CPU 回退
  `road_yolo11n_seg_128.onnx`；未使用模型和 `.npz` 不下载到板端。
- [x] `/dev/galcore` 存在、`galcore` 模块已加载，NPU V5 与 CPU 回退的板端执行路径已恢复。
- [x] `/dev/ttySTM4`、`/dev/ttySTM9` 设备节点存在；当前没有连接任何外设，真实数据链路仍待测。

### 3.2 蓝牙 NAP、SSH 与公网

- [x] 针对当前内核构建并安装匹配 `Module.symvers` 的 `bnep.ko`。
- [x] `bt.service`、`myir-bt-nap.service`、`myir-bt-pairing-agent.service` 均已持久化启用。
- [x] 标准 `bluetooth.service` 禁用，避免与厂商 bluetoothd 争用 D-Bus 名称。
- [x] 板端正常冷启动地址为 `54:78:C9:E6:FB:6D`；PC 地址为 `E8:C8:29:25:40:E9`。
- [x] Windows PAN 为 `192.168.137.1/24`，`MYIR-Bluetooth-NAT` 为
  `192.168.137.0/24`；板端 `pan0` 为 `192.168.137.2/24`。
- [x] 在 J11 网线物理拔除状态下验证 `bnep0 UP,LOWER_UP`、默认路由经 `pan0`、公网 ping、
  `ssh root@192.168.137.2` 和 Git 远端读取。
- [x] PC 重启后配对、静态 PAN 地址和 NAT 均保留；PAN 会话必要时仍需手动重新选择“接入点”。
- [x] 完整断电冷启动后板端真实蓝牙地址、服务、配对、NAP、公网和 SSH 均已恢复验证。

完整过程和失败证据见 [BLUETOOTH_PAN_HOST_NETWORKING.md](BLUETOOTH_PAN_HOST_NETWORKING.md)。

### 3.3 J11 有线维护通道

- [x] 查阅硬件文档并确认物理可用的中间 RJ45 为 J11；启动项使用
  `myb-stm32mp257x-2GB-ethswitch`。
- [x] 确认 J11 对应 `sw0p2`，CPU 端口为 `sw0ep`。
- [x] 板端 `/etc/systemd/network/12-sw0ep.network` 持久保存到 PC
  `192.168.0.2/32` 的链路路由，板端重启后仍有效。
- [x] PC 为 `192.168.0.2/24`，开发板为 `192.168.0.10`，插线后可直接 SSH。
- [x] 原启动和网络配置备份在 `/data/recovery-backup-20260807-j11/`。
- [x] 桌面 `Enable-J11-Internet.ps1`/`Disable-J11-Internet.ps1` 分别启用和关闭 J11 公网 NAT；
  Disable 不删除直连地址，也不影响蓝牙 NAT。
- [x] 最终无线验收时 J11 已拔除、`MYIR-J11-NAT` 已删除；J11 仅作为维护后备通道。

### 3.4 WiFi 决策

- [x] 板端 WiFi 模块不再继续检查或恢复。
- [x] 不恢复历史固定 WiFi 地址 `192.168.31.199`，不配置 WiFi 开机自连。
- [x] 日常无网线联网和 SSH 使用蓝牙 NAP；J11 只作维护后备。

## 4. 失败经验与不可重复操作

### 4.1 蓝牙软重启不等于硬件复位

板端软重启后曾出现异常地址 `43:45:C5:00:1F:AC`、HCI command tx timeout，以及 Windows
“没有响应”/“输入码无效”。`btmon` 没有连接请求、Agent 没有调用、ACL 计数为零，说明请求没有
到达 BlueZ。重启服务、修改 PIN、重复删除设备、重启 NAP/NAT 均不能修复这一层故障。

厂商脚本还报 `/etc/myir_test/myir_bt: line 14: kill_process_fun: command not found`；
`brcm_patchram_plus` 重发 HCI Reset 也收不到响应。正确处理为：

```bash
sync
systemctl poweroff
```

确认关机后物理断电至少 10 秒，再重新上电。只有控制器恢复正常地址
`54:78:C9:E6:FB:6D` 后才重新配对。

### 4.2 Windows 配对入口不能混用

- 首次/重新配对：使用现代“设置 → 蓝牙和设备 → 添加设备 → 蓝牙”。
- 配对后建立 PAN：使用经典“设备和打印机 → myd-ld25x → 连接方式 → 接入点”。
- 旧 `DevicePairingWizard` 在本机出现过本地“输入码无效”，当时请求没有到达板端 Agent。
- Windows Intel 蓝牙设备若显示“等待系统重新启动”，完整重启 PC；本机
  `pnputil /restart-device` 返回不支持和退出码 50，不能替代重启。

### 4.3 已排除项

- Ethernet Switch DTB 与普通 DTB 的蓝牙 UART/regulator 语义相同；J11 DTB 不是蓝牙故障根因。
- Windows Classic Inquiry 能找到设备，故不是天线完全无信号。
- NAP、BNEP 和 NAT 位于配对之后，不是“输入码无效”的直接原因。
- 旧失败目录无需保存；只保留最终有效配置、必要备份和文档证据。

## 5. 尚待完成

### P2：无外设及真实硬件验证

- [x] 项目、虚拟环境及主要 Python 原生库可加载。
- [x] NPU V5 `.nb` 与 CPU 回退 `.onnx` 执行路径恢复。
- [ ] 连接外设后验证 `/dev/ttySTM4`、`/dev/ttySTM9` 的雷达和飞控真实数据。
- [ ] 验证摄像头稳定设备路径、采集格式和双摄映射。
- [ ] 依次执行 serial、radar、camera/NPU 硬件 smoke tests。
- [ ] 在具备安全条件且不输出真实飞行控制时完成主程序 dry-run。

### P3：网络边界验证

- [x] PC 重启后的 PAN 静态地址、NAT 和配对保留行为已验证。
- [x] 板端软重启失败行为与完整断电冷启动恢复流程已验证并记录。
- [x] 记录 BNEP 会话可能需要在 Windows 手动重连“接入点”。
- [ ] 切换 PC 上游 WiFi、有线网络或手机热点后，重新验证板端公网访问。
- [ ] 每次切换上游网络时确认其不占用 `192.168.137.0/24`。

## 6. 当前不做的事项

- [x] 不恢复或继续检查板端 WiFi。
- [x] 不同步、重烧或验证 eMMC 启动。
- [x] 不把 eMMC 用作项目或日志存储。
- [x] 不把未使用模型、`.npz`、临时抓包或旧失败目录放到板端。
- [x] 不通过 SSH 直接编辑板端项目源码；所有源码变更先在本地/云端 Git 提交。

## 7. 后续建议顺序

```text
1. 正常使用蓝牙 PAN；若控制器地址异常，先完整断电冷启动
2. 需要大量下载时先检查 df -h / /data，并按 USTC → TUNA → Aliyun → 默认源回退
3. 需要有线维护时插入 J11；需要公网再运行桌面 Enable-J11-Internet.ps1
4. 外设接入后依次验证串口、雷达、摄像头和安全 dry-run
5. 更新代码时先在本地/云端提交，仅让板端拉取已批准提交
```

## 8. 恢复完成判定

- [x] SD 卡启动、根分区和 `/data` 项目数据分区稳定。
- [x] Git 仓库、虚拟环境、依赖、当前使用模型及配置恢复。
- [x] NPU V5 和 CPU 回退的用户态执行路径恢复。
- [x] 蓝牙 NAP/NAT、SSH、PC 重启和板端冷启动行为已验证并归档。
- [x] 项目和大文件位于 `/data`，未使用模型及 `.npz` 不下发。
- [ ] 无外设检查之外的串口、雷达和摄像头硬件 smoke tests 尚未执行。
- [ ] PC 切换不同上游网络后的公网行为尚待逐种验证。

因此，**系统、存储、项目环境和无网线蓝牙主链路的恢复已经完成**；整机项目恢复仍需在外设
接入后完成真实硬件 smoke tests，且不能把“设备节点存在”视为外设功能已验证。
