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

## 5. Windows 配对和连接接入点

1. 临时让开发板蓝牙处于可发现、可配对状态。
2. 在 Windows 中添加蓝牙设备并完成配对。板端设备名为 `myd-ld25x`。
3. NAP 注册服务运行后，如果 Windows 之前是在 NAP 注册前配对的，应删除设备并重新配对，
   让 Windows 重新枚举 NAP 服务。
4. 打开经典“设备和打印机”界面，而不是只使用 Windows 11 新设置页面。
5. 右键 `myd-ld25x`，选择“连接方式 → 接入点”。
6. 连接成功后，Windows 中的“蓝牙网络连接”状态变为 `Up`；板端出现 `bnep0`，并自动加入
   `pan0`。

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

当前镜像没有安装 `git`。SSH 和公网连通成功并不代表可以立即执行 `git fetch/pull`；安装 `git`
属于板端系统修改，应另行批准。

## 8. 日常使用和重启

PC 上游网络可以更换为其他 WiFi、有线网络或手机热点，蓝牙内网 SSH 地址仍保持：

```powershell
ssh root@192.168.137.2
```

条件是：

- Windows“蓝牙网络连接”仍处于连接状态；
- `192.168.137.0/24` 未与 PC 新接入的网络冲突；
- Windows 静态地址和 `MYIR-Bluetooth-NAT` 仍存在。

重启行为：

- 仅开发板重启：`myir-bt-nap.service` 会自动启动，不需要重新配置或重新配对；原 BNEP 会话
  会断开，Windows 不保证自动恢复。如果蓝牙网络未恢复，重新选择“连接方式 → 接入点”。
- 仅 PC 重启：配对、静态地址和 NAT 应保留，但通常仍需检查并按需重新连接“接入点”。
- 两者同时重启：等待开发板 NAP 服务启动后，再从 PC 连接接入点。
- 当前尚未执行完整重启验证，因此按“可能需要手动重新连接接入点”处理。

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

## 10. 快速排障表

| 现象 | 优先检查 |
| --- | --- |
| Windows 找不到“接入点” | NAP 注册服务是否常驻、Windows 是否在 NAP 注册后重新配对 |
| 只有“已配对”，没有网络 | 是否在经典“设备和打印机”中执行“连接方式 → 接入点” |
| 板端没有 `bnep0` | Windows PAN 是否已连接、`bnep` 是否已加载 |
| Windows 蓝牙接口出现 `169.254.x.x` | 静态 `192.168.137.1/24` 是否丢失 |
| PC 能访问板端，板端不能访问公网 | `MYIR-Bluetooth-NAT`、PC 上游网络和板端默认路由 |
| 能 ping 公网 IP，域名失败 | 板端 DNS 配置 |
| SSH 提示 host key changed | 确认重烧录后，仅删除该开发板 IP 的旧 known_hosts 条目 |
| 换 PC 网络后异常 | 新网络是否与 `192.168.137.0/24` 冲突，PAN 是否仍连接 |
| WiFi 与蓝牙同时在线但出口不符预期 | `ip route` 中两条默认路由及 metric |

## 11. 项目部署约束

- 不要通过 SSH 直接修改开发板上的项目源码。
- 源码修改必须先在本地/云端 Git 仓库完成并提交。
- 板端只能拉取已经批准并已提交的版本，不得向远端 push。
- 未明确要求时，不在板端执行 `git pull`。

