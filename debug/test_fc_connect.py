"""
Phase A: 飞控基础连通性测试。

验证: 设备发现 → 串口打开 → 协议握手 → 状态回传。

用法:
    PYTHONPATH=. python debug/test_fc_connect.py
    PYTHONPATH=. python debug/test_fc_connect.py --port /dev/ttyUSB0
"""

import argparse
import sys
import time
from pathlib import Path


def _setup_path() -> None:
    root = Path(__file__).resolve().parents[1]
    for p in (root,):
        value = str(p)
        if value not in sys.path:
            sys.path.insert(0, value)


def main() -> None:
    _setup_path()

    from FlightController import FC_Controller
    from loguru import logger

    parser = argparse.ArgumentParser(description="飞控基础连通性测试 (Phase A)")
    parser.add_argument("--port", default=None, help="飞控串口路径 (默认: 自动探测)")
    args = parser.parse_args()

    logger.info("=== Phase A: 飞控基础连通性测试 ===")

    fc = FC_Controller()

    try:
        logger.info("正在连接飞控 (自动探测端口)...")
        fc.start_listen_serial(serial_dev=args.port, block_until_connected=True)

        if not fc.wait_for_connection(timeout_s=5):
            logger.error("连接超时！请检查飞控 USB 连接和供电。")
            return

        logger.info("飞控已连接，等待状态数据稳定...")
        time.sleep(1.0)

        s = fc.state
        logger.info(f"mode  = {s.mode.value} ({'定高' if s.mode.value == 1 else '定点' if s.mode.value == 2 else '程控' if s.mode.value == 3 else '未知'})")
        logger.info(f"unlock = {s.unlock.value}")
        logger.info(f"bat   = {s.bat.value:.1f}V")
        logger.info(f"alt   = {s.alt_add.value}cm (add) / {s.alt_fused.value}cm (fused)")
        logger.info(f"姿态  = roll={s.rol.value:.1f} pit={s.pit.value:.1f} yaw={s.yaw.value:.1f}")
        logger.info(f"速度  = vx={s.vel_x.value} vy={s.vel_y.value} vz={s.vel_z.value}")
        logger.info(f"位置  = x={s.pos_x.value} y={s.pos_y.value}")
        logger.info(f"指令  = cid={s.cid.value} cmd_0={s.cmd_0.value} cmd_1={s.cmd_1.value}")

        # 验证项
        all_ok = True

        # 电池电压: USB供电时bat=0是正常的
        if s.bat.value == 0.0:
            logger.info("[WARN] 电池电压=0V — 飞控可能为USB供电，电池未接")
        elif s.bat.value < 5.0:
            logger.warning(f"[FAIL] 电池电压过低 ({s.bat.value:.1f}V < 5V)")
            all_ok = False
        else:
            logger.info(f"[PASS] 电池电压 ({s.bat.value:.1f}V)")

        checks = [
            ("mode 非默认(>0)", s.mode.value > 0),
            ("connected=True", fc.connected),
        ]
        for desc, result in checks:
            status = "PASS" if result else "FAIL"
            if not result:
                all_ok = False
            logger.info(f"[{status}] {desc}")

        if all_ok:
            logger.info("Phase A 全部通过！")
        else:
            logger.warning("Phase A 部分检查未通过，请排查。")

    except RuntimeError as e:
        logger.error(f"飞控设备未找到: {e}")
        logger.error("排查: ls /dev/serial/by-id/*66CC*")
    except Exception as e:
        logger.error(f"未预期的错误: {type(e).__name__}: {e}")
    finally:
        fc.close()
        logger.info("飞控已断开。")


if __name__ == "__main__":
    main()
