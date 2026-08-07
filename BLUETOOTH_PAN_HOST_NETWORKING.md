# MYD-LD25x 通过 Windows 主机蓝牙 PAN 联网

本文记录 MYD-LD25x 在板端 WiFi 不可用时，通过蓝牙连接 Windows PC，并借助 PC 的上游网络
访问互联网及接受 SSH 连接的实际配置过程。

本文描述的是本机已经验证成功的方案，不代表所有 Windows/BlueZ 组合都具有相同角色能力。
修改板端系统文件、systemd 服务、Windows IP 或 NAT 前仍需获得操作批准。

## 1. 最终拓扑

```text
开发板（BlueZ NAP）
pan0: 192.168.137.2/24
        │
        │ Bluetooth BNEP/PAN
        │
Windows PC（PANU）
蓝牙网络连接: 192.168.137.1/24
        │
        │ Windows NetNat: MYIR-Bluetooth-NAT
        │
PC 的当前上游网络（WiFi、有线网络或手机热点）
```

角色和用途：

- 开发板充当蓝牙 **NAP**（Network Access Point）。
- Windows PC 充当 **PANU** 客户端，并把自己的上游网络通过 NAT 转发给开发板。
- PC 可以通过蓝牙内网地址 `192.168.137.2` SSH 到开发板。
- NAT 转发开发板主动发起的请求及其响应，不会把开发板直接暴露到互联网。
- PC 更换上游 WiFi/有线网络通常不改变蓝牙 SSH 地址。

当前地址和服务：

| 项目 | 值 |
| --- | --- |
| Windows 蓝牙接口 | `蓝牙网络连接`，`192.168.137.1/24` |
| Windows NAT | `MYIR-Bluetooth-NAT`，`192.168.137.0/24` |
| 板端蓝牙网桥 | `pan0`，`192.168.137.2/24` |
| 板端蓝牙数据接口 | `bnep0`，连接后自动加入 `pan0` |
| 板端默认路由 | `default via 192.168.137.1 dev pan0 metric 50` |
| 板端 NAP 服务 | `myir-bt-nap.service` |
| 板端配对 Agent | `myir-bt-pairing-agent.service` |
| 板端蓝牙地址（正常冷启动） | `54:78:C9:E6:FB:6D` |
| Windows 蓝牙地址 | `E8:C8:29:25:40:E9`（设备名 `LUAI`） |
| 板端 SSH 服务 | Dropbear，`dropbear.socket` 监听 TCP 22 |
| SSH 命令 | `ssh root@192.168.137.2` |

## 2. 为什么采用“板端 NAP、Windows PANU”

最初尝试让 Windows PC 提供 NAP、开发板作为客户端连接。实际检查发现，本机 Windows 蓝牙栈
只对外提供 PANU 服务，没有提供 NAP 服务；板端请求 `Network1.Connect("nap")` 返回
`Operation is not supported`。

因此最终反转角色：

- 开发板通过 BlueZ 注册 NAP 服务；
- Windows 连接开发板的“接入点”；
- Windows 只负责 PANU 和 IPv4 NAT。

Windows 新设置页面把开发板识别为音频/未指定设备，并不表示 PAN 失败。PAN 操作入口位于经典
“设备和打印机”界面。

## 3. 板端 BNEP 内核能力补齐

### 3.1 初始问题

板端初始内核状态：

```text
Linux 6.6.48-gbebcf479fd77 aarch64
# CONFIG_BT_BNEP is not set
CONFIG_MODULES=y
CONFIG_MODVERSIONS=y
```

同时满足以下情况：

- `/lib/modules/6.6.48-gbebcf479fd77` 下没有 `bnep.ko`；
- `modinfo bnep` 返回模块不存在；
- 板端没有内核 headers/devsrc，不能直接现场编译。

### 3.2 编译环境与产物

采用“厂商 SDK + 只编译内核 BNEP 模块”的路线：

- WSL 发行版：导入在 `D:\WSL\Ubuntu-20.04`；
- 厂商 SDK 安装包：
  `D:\WSL\SDK\myir-image-full-openstlinux-weston-myd-ld25x.rootfs-x86_64-toolchain-5.0.3-snapshot.sh`；
- PC 上厂商源码包：`D:\drone\嵌赛\stm32mp257\04-Sources`；
- 编译器版本必须匹配板端：`aarch64-ostl-linux-gcc 13.3.0`；
- 使用板端 `kernel.config` 和匹配内核构建的 `Module.symvers`；
- 只构建 `net/bluetooth/bnep`，不重编完整系统镜像。

保留的本地产物：

```text
D:\WSL\Output\BNEP-6.6.48-gbebcf479fd77\
├── bnep.ko
├── kernel.config
├── Module.symvers
├── serial-install.log
├── serial-install-attempt2.log
├── windows-bluetooth-nat.log
└── windows-ics-nap.log
```

当前 `bnep.ko`：

```text
SHA256: d073dc59e5dd1051e8290a1cf71c1932e0891e54be7b42d53fb15e7906c158e3
vermagic: 6.6.48-gbebcf479fd77 SMP preempt mod_unload modversions aarch64
```

> 如果内核版本、Git revision、`CONFIG_MODVERSIONS`、`Module.symvers` 或工具链发生变化，不能
> 继续复用这个 `bnep.ko`，必须针对新内核重新构建。

### 3.3 模块安装过程

本次通过串口把 `bnep.ko` 的 Base64 内容分块传到 `/tmp/bnep.ko.b64`，解码并核对 SHA256 后，
执行以下安装命令：

```bash
install -d /lib/modules/$(uname -r)/kernel/net/bluetooth/bnep
install -m 0644 /tmp/bnep.ko \
  /lib/modules/$(uname -r)/kernel/net/bluetooth/bnep/bnep.ko
depmod -a
modprobe bnep
```

验证：

```bash
modinfo bnep | grep -E '^(filename|description|depends|vermagic):'
test -d /sys/module/bnep && echo BNEP_LOADED=YES
lsmod | grep -E '^(bnep|bluetooth)[[:space:]]'
```

模块最终安装在：

```text
/lib/modules/6.6.48-gbebcf479fd77/kernel/net/bluetooth/bnep/bnep.ko
```

## 4. 板端 NAP 持久化配置

### 4.1 为什么需要常驻注册进程

BlueZ 的 `org.bluez.NetworkServer1.Register("nap", "pan0")` 注册与 D-Bus 调用者生命周期绑定。
如果使用执行后立即退出的 `busctl call`，NAP UUID 会在调用进程退出后被撤销，Windows 重新配对
时也不会发现“Personal Area Network NAP Service”。

因此使用常驻 Python/Gio 进程完成注册，并由 systemd 管理。

### 4.2 NAP 注册脚本

文件：`/usr/local/sbin/myir-bt-nap-register.py`

```python
#!/usr/bin/env python3
import time
from gi.repository import Gio, GLib

while True:
    try:
        bus = Gio.bus_get_sync(Gio.BusType.SYSTEM, None)
        for name, value in (
            ("Powered", True),
            ("Discoverable", False),
            ("Pairable", False),
        ):
            bus.call_sync(
                "org.bluez",
                "/org/bluez/hci0",
                "org.freedesktop.DBus.Properties",
                "Set",
                GLib.Variant(
                    "(ssv)",
                    ("org.bluez.Adapter1", name, GLib.Variant("b", value)),
                ),
                None,
                Gio.DBusCallFlags.NONE,
                -1,
                None,
            )

        bus.call_sync(
            "org.bluez",
            "/org/bluez/hci0",
            "org.bluez.NetworkServer1",
            "Register",
            GLib.Variant("(ss)", ("nap", "pan0")),
            None,
            Gio.DBusCallFlags.NONE,
            -1,
            None,
        )
        break
    except GLib.Error as error:
        print(f"NAP registration retry: {error}", flush=True)
        time.sleep(3)

print("NAP_REGISTERED", flush=True)
GLib.MainLoop().run()
```

该脚本会保持蓝牙电源开启，并在正常运行时关闭可发现和可配对，避免开发板长期暴露在扫描列表中。
已经配对的 PC 仍可连接。需要配对新设备时，应临时打开可发现/可配对，完成后再关闭。

### 4.3 systemd 服务

文件：`/etc/systemd/system/myir-bt-nap.service`

```ini
[Unit]
Description=MYIR Bluetooth PAN/NAP service
Requires=bt.service
After=bt.service
PartOf=bt.service

[Service]
Type=simple
ExecStartPre=/sbin/modprobe bnep
ExecStartPre=/bin/sh -c '/sbin/ip link show pan0 >/dev/null 2>&1 || /sbin/ip link add pan0 type bridge'
ExecStartPre=/sbin/ip link set pan0 up
ExecStartPre=/sbin/ip addr replace 192.168.137.2/24 dev pan0
ExecStartPre=/sbin/ip route replace default via 192.168.137.1 dev pan0 metric 50
ExecStart=/usr/bin/python3 /usr/local/sbin/myir-bt-nap-register.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

安装并启动：

```bash
chmod 0755 /usr/local/sbin/myir-bt-nap-register.py
chmod 0644 /etc/systemd/system/myir-bt-nap.service
systemctl daemon-reload
systemctl enable --now myir-bt-nap.service
```

验证：

```bash
systemctl is-enabled myir-bt-nap.service
systemctl is-active myir-bt-nap.service
systemctl status myir-bt-nap.service --no-pager -l
bluetoothctl show | grep -E 'Powered:|Discoverable:|Pairable:|UUID: NAP'
```

期望看到服务为 `enabled/active`，并出现：

```text
UUID: NAP (00001116-0000-1000-8000-00805f9b34fb)
```

### 4.4 持久化配对 Agent

为避免无人值守板端在重新配对时没有进程响应 BlueZ 的认证请求，已安装：

```text
/usr/local/sbin/myir-bt-pairing-agent.py
/etc/systemd/system/myir-bt-pairing-agent.service
```

服务使用 `NoInputNoOutput` 能力，注册 `org.bluez.Agent1`，自动接受 SSP/Just Works 确认和服务
授权；兼容路径中的 PIN/Passkey 分别返回 `0000` 和 `0`。服务与厂商 `bt.service` 绑定，配置为
`enabled/active`、异常退出后 3 秒重启。成功配对时日志中实际出现了 A2DP/AVRCP UUID 的
`AuthorizeService` 调用，证明 Windows 的请求已经到达板端 Agent。

板端使用厂商 `bt.service` 启动 `/usr/libexec/bluetooth/bluetoothd`。标准
`bluetooth.service` 保持 `disabled/inactive`，否则两个 bluetoothd 会争用 D-Bus 名称并报：

```text
D-Bus setup failed: Name already in use
```

当前应同时保持以下三个服务：

```bash
systemctl is-active bt.service myir-bt-nap.service myir-bt-pairing-agent.service
```

## 5. Windows 配对和连接接入点

配对和建立 PAN 是两个不同步骤，必须使用不同入口：

1. 先确认板端控制器地址是正常值 `54:78:C9:E6:FB:6D`，且三个蓝牙服务均为 active。如果地址
   变成 `43:45:C5:00:1F:AC` 或 HCI 报超时，不要继续删除/添加设备，按第 10 节完整断电恢复。
2. 临时打开板端可发现、可配对：

   ```bash
   bluetoothctl power on
   bluetoothctl pairable on
   bluetoothctl discoverable-timeout 0
   bluetoothctl discoverable on
   ```

3. 在 Windows 11 **现代“设置 → 蓝牙和设备 → 添加设备 → 蓝牙”** 中选择 `myd-ld25x` 完成
   SSP/Just Works 配对。不要使用旧版 `DevicePairingWizard` 完成这一步；本机实测旧向导会在
   请求尚未到达板端时显示“输入码无效”。
4. 在板端信任已经配对的 PC：

   ```bash
   bluetoothctl trust E8:C8:29:25:40:E9
   ```

5. 如果 Windows 是在 NAP 注册前配对的，应删除旧设备记录，再按以上步骤配对，使 Windows
   重新枚举 NAP 服务。
6. **配对完成后**打开经典“设备和打印机”，右键 `myd-ld25x`，选择“连接方式 → 接入点”。
   经典界面用于建立 PAN，而不是用于首次配对。
7. 连接成功后，Windows 中的“蓝牙网络连接”状态变为 `Up`；板端出现 `bnep0`，并自动加入
   `pan0`。
8. 配对完成后关闭暴露状态；已配对、已信任的 PC 仍可连接：

   ```bash
   bluetoothctl discoverable off
   bluetoothctl pairable off
   ```

Windows 设备属性中应能看到：

```text
Personal Area Network NAP Service
UUID: 00001116-0000-1000-8000-00805F9B34FB
```

仅显示“已配对”不等于 PAN 已连接；必须出现 Windows 蓝牙网络接口 `Up` 和板端 `bnep0`。

## 6. Windows IPv4 和 NAT

### 6.1 ICS 尝试及失败

曾尝试通过 Windows Internet Connection Sharing：

- WLAN 作为公共/上游接口；
- “蓝牙网络连接”作为私有接口。

本机在执行蓝牙私有侧 `EnableSharing(1)` 时持续返回：

```text
E_UNEXPECTED (0x8000FFFF)
```

因此没有继续依赖 ICS，改用可明确指定内部网段的 Windows `NetNat`。

### 6.2 最终 NetNat 配置

在管理员 PowerShell 中完成以下结果：

```powershell
# 蓝牙 PAN 接口地址
New-NetIPAddress `
  -InterfaceAlias '蓝牙网络连接' `
  -IPAddress 192.168.137.1 `
  -PrefixLength 24

# 只在该 NAT 尚不存在时创建
New-NetNat `
  -Name 'MYIR-Bluetooth-NAT' `
  -InternalIPInterfaceAddressPrefix '192.168.137.0/24'
```

实际配置前还移除了蓝牙接口自动生成的 APIPA 地址 `169.254.138.229/16`，并关闭了此前部分成功、
但未完整建立的 WLAN ICS 共享。不要批量删除其他接口地址或无关 NAT。

验证：

```powershell
Get-NetAdapter -Name '蓝牙网络连接'
Get-NetIPAddress -InterfaceAlias '蓝牙网络连接' -AddressFamily IPv4
Get-NetNat -Name 'MYIR-Bluetooth-NAT'
```

期望值：

```text
蓝牙网络连接: Up
IPv4: 192.168.137.1/24
NAT: MYIR-Bluetooth-NAT
Internal prefix: 192.168.137.0/24
Active: True/1
```

## 7. 链路和 SSH 验证

### 7.1 板端

```bash
ip -brief address show pan0
ip -brief link show bnep0
ip route
ping -c 3 1.1.1.1
ping -c 2 github.com
systemctl status dropbear.socket --no-pager
```

本次验证结果：

- `pan0` 为 `192.168.137.2/24`；
- `bnep0` 为 `UP,LOWER_UP`；
- 默认路由经 `192.168.137.1 dev pan0 metric 50`；
- 公网 IP 可达；
- DNS 可解析 `github.com`；
- Dropbear 的 `dropbear.socket` 监听 TCP 22。

### 7.2 Windows

```powershell
ping 192.168.137.2
ssh root@192.168.137.2
```

如果系统重烧录导致 Dropbear 主机密钥变化，只移除该地址的旧记录：

```powershell
ssh-keygen -R 192.168.137.2
ssh root@192.168.137.2
```

不要清空整个 `known_hosts`。

`git` 已恢复，已通过纯蓝牙链路读取远端仓库 `origin`，得到远端 HEAD：

```text
7e246e63ae49ba6977ec2cd197c8b3a865717986 HEAD
```

这只证明当前网络和 Git 远端读取正常；仍须遵守“源码先在本地/云端提交，板端只拉取已批准
提交”的部署约束。

## 8. 日常使用和重启

PC 上游网络可以更换为其他 WiFi、有线网络或手机热点，蓝牙内网 SSH 地址仍保持：

```powershell
ssh root@192.168.137.2
```

条件是：

- Windows“蓝牙网络连接”仍处于连接状态；
- `192.168.137.0/24` 未与 PC 新接入的网络冲突；
- Windows 静态地址和 `MYIR-Bluetooth-NAT` 仍存在。

重启行为已经实测：

- PC 完整重启后，配对记录、`192.168.137.1/24` 和 `MYIR-Bluetooth-NAT` 均保留。PAN 断开时
  可能暂时出现 APIPA/Tentative 地址；重新连接“接入点”后只保留静态地址为 Preferred。
- 仅对板端执行软件重启并不一定会复位 Broadcom 蓝牙模块。本次曾在软重启后出现 HCI 命令
  超时和异常地址 `43:45:C5:00:1F:AC`，此时反复配对无效。
- 遇到异常地址或 HCI 超时时，应执行 `sync; systemctl poweroff`，待系统关机后切断开发板电源
  至少 10 秒，再重新上电。完整冷启动已验证能恢复真实地址 `54:78:C9:E6:FB:6D` 和正常配对。
- 无论哪一端重启，原 BNEP 会话都会断开；Windows 不保证自动重新建立 PAN。等待板端三个服务
  启动后，按需在经典“设备和打印机”中重新选择“连接方式 → 接入点”。通常不需重新配对。

## 9. 与板端 WiFi 并存时

当前重烧录后的板端实测状态是：

```text
wlan0: DOWN
WiFi: Not connected
wpa_supplicant@wlan0.service: disabled/inactive
/etc/wpa_supplicant: 没有接口配置文件
```

因此当前唯一默认路由是蓝牙 `pan0`，板端外网请求全部经 PC 转发。

以后为 `wlan0` 配置 WiFi 后，开发板可以连接与 PC 不同的网络，但此时会存在蓝牙和 WiFi 两条
联网路径。板端访问远端网络实际使用哪一条，取决于 `ip route` 中默认路由的 metric；metric
更低者优先。

特别注意：如果板端 WiFi 或 PC 新接入的网络也使用 `192.168.137.0/24`，会与当前蓝牙 PAN
发生路由冲突。应先断开其中一条链路，或经批准后修改一个网段。

## 10. 本次配对失败的根因与判定方法

### 10.1 失败链路

软重启后，板端曾从正常地址 `54:78:C9:E6:FB:6D` 变为异常地址
`43:45:C5:00:1F:AC`。Windows 仍能通过 Classic Inquiry 扫描到该设备，但连接时分别出现：

- 现代设置：“我们没有从设备收到任何响应”；
- 旧版 `DevicePairingWizard`：“输入码无效”；
- Windows API `BluetoothAuthenticateDeviceEx` 返回 `258 (WAIT_TIMEOUT)`。

同时板端证据为：

- `btmon` 没有 HCI Connection Request/Complete；
- 配对 Agent 没有收到任何调用；
- ACL 收发计数保持为零；
- 内核可能出现 `Bluetooth: hci0: command tx timeout` 或读取本地名称超时。

这说明错误发生在 Windows UI 或 PIN 校验之前：低层扫描尚可用，但 ACL 连接没有到达 BlueZ，
因此反复修改 PIN、重启 NAP、切换 NAT 或重新添加 Windows 设备都不能解决。

### 10.2 已排除的原因

- J11 Ethernet Switch DTB 与普通 DTB 的蓝牙 UART、蓝牙电源 regulator 语义相同；冷启动后在
  Ethernet Switch DTB 下也已成功配对，故 J11 DTB 不是根因。
- Windows 能完成低层 Inquiry，故不是“天线完全无信号”。
- NAP/BNEP/NAT 都发生在配对之后，不会导致配对阶段的“输入码无效”。
- 异常地址阶段请求根本没有到达 BlueZ，因此当时即使补充配对 Agent，也不能修复底层 HCI
  失响应；Agent 是为控制器健康时的后续认证提供持久响应。
- Windows 设备元数据中的 `0x80070490` 同时出现在多个无关设备上，不能作为本故障根因。

### 10.3 软件恢复为何失败

尝试停止 NAP/Agent/蓝牙进程并重新启动厂商 `bt.service` 时，`brcm_patchram_plus` 每 4 秒发送
一次 HCI Reset（`01 03 0c 00`），模块均不响应，最终 `/sys/class/bluetooth` 为空且系统没有
默认控制器。厂商脚本还暴露出：

```text
/etc/myir_test/myir_bt: line 14: kill_process_fun: command not found
```

因此该脚本不能保证在同一次上电周期内安全地解绑并重新初始化 UART。正确恢复方式是安全关机后
完整断电冷启动，而不是继续循环执行服务重启。

PC 侧也曾出现 Intel Bluetooth 设备等待重启，`pnputil /restart-device` 返回“不支持”及退出码
50；只有 PC 完整重启后设备才恢复 `Started`、`bthserv` 恢复 `RUNNING`。如果 Windows 明确显示
设备正在等待系统重新启动，应先重启 PC，不要持续删除配对记录。

## 11. 快速排障表

| 现象 | 优先检查 |
| --- | --- |
| 设备地址不是 `54:78:C9:E6:FB:6D` | 停止配对，安全关机并完整断电至少 10 秒 |
| “无响应”或“输入码无效”，板端 Agent 无日志 | 用 `btmon` 判断请求是否到达；若同时有 HCI 超时，执行冷启动 |
| Windows 蓝牙设备“等待系统重新启动” | 完整重启 PC；`pnputil /restart-device` 在本机不能替代重启 |
| Windows 找不到“接入点” | NAP 注册服务是否常驻、Windows 是否在 NAP 注册后重新配对 |
| 首次配对出现“输入码无效” | 是否误用了旧版 `DevicePairingWizard`；改用现代“添加设备 → 蓝牙” |
| 只有“已配对”，没有网络 | 是否在经典“设备和打印机”中执行“连接方式 → 接入点” |
| 板端没有 `bnep0` | Windows PAN 是否已连接、`bnep` 是否已加载 |
| Windows 蓝牙接口出现 `169.254.x.x` | 静态 `192.168.137.1/24` 是否丢失 |
| PC 能访问板端，板端不能访问公网 | `MYIR-Bluetooth-NAT`、PC 上游网络和板端默认路由 |
| 能 ping 公网 IP，域名失败 | 板端 DNS 配置 |
| SSH 提示 host key changed | 确认重烧录后，仅删除该开发板 IP 的旧 known_hosts 条目 |
| 换 PC 网络后异常 | 新网络是否与 `192.168.137.0/24` 冲突，PAN 是否仍连接 |
| WiFi 与蓝牙同时在线但出口不符预期 | `ip route` 中两条默认路由及 metric |

## 12. J11 有线维护通道

最终运行目标仍是无网线的蓝牙 PAN；中间 RJ45（J11）仅作为维护、恢复和蓝牙失效时的后备
通道。硬件限制下采用 Ethernet Switch DTB：

```text
/boot/mmc0_extlinux/myb-stm32mp257x-2GB_extlinux.conf
DEFAULT myb-stm32mp257x-2GB-ethswitch
```

原配置备份在 `/data/recovery-backup-20260807-j11/`。该 DTB 下 J11 对应 `sw0p2`，CPU 端口为
`sw0ep`。为确保到 PC 的直连路由不误走 `end1`，板端持久配置为：

```ini
# /etc/systemd/network/12-sw0ep.network
[Match]
Name=sw0ep

[Network]
KeepConfiguration=yes
LinkLocalAddressing=ipv6

[Route]
Destination=192.168.0.2/32
Scope=link
Metric=10
```

地址及操作：

- PC Realtek 以太网：`192.168.0.2/24`；
- 开发板 J11：`192.168.0.10`；
- 直连 SSH：`ssh root@192.168.0.10`；
- 桌面脚本：`Enable-J11-Internet.ps1` 和 `Disable-J11-Internet.ps1`。

Enable 脚本为 J11 创建 `MYIR-J11-NAT` 并临时让板端默认路由经 `192.168.0.2`；Disable 脚本删除
J11 默认路由和 NAT，但保留 `192.168.0.2/24`，因此以后重新插入 J11 仍可直接 SSH，且不会
破坏 `MYIR-Bluetooth-NAT`。最终无线验证时 J11 网线已拔除、`MYIR-J11-NAT` 已删除，当前唯一
公网 NAT 为 `MYIR-Bluetooth-NAT`。

## 13. 项目部署约束

- 不要通过 SSH 直接修改开发板上的项目源码。
- 源码修改必须先在本地/云端 Git 仓库完成并提交。
- 板端只能拉取已经批准并已提交的版本，不得向远端 push。
- 未明确要求时，不在板端执行 `git pull`。
