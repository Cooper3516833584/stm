"""
Phase B: 飞控指令下发测试。

验证: set_flight_mode() → ACK 机制 → mode 状态变化 → 全状态字段可读。

用法:
    PYTHONPATH=. python debug/test_fc_command.py
    PYTHONPATH=. python debug/test_fc_command.py --port /dev/ttyACM0
"""

import argparse
import sys
import time
from pathlib import Path


def _setup_path() -> None:
    root = Path(__file__).resolve().parents[1]
    value = str(root)
    if value not in sys.path:
        sys.path.insert(0, value)


MODE_NAMES = {1: "定高 (ALT_HOLD)", 2: "定点 (HOLD_POS)", 3: "程控 (PROGRAM)"}


def main() -> None:
    _setup_path()

    from FlightController import FC_Controller
    from loguru import logger

    parser = argparse.ArgumentParser(description="飞控指令下发测试 (Phase B)")
    parser.add_argument("--port", default=None, help="飞控串口路径 (默认: 自动探测)")
    parser.add_argument("--target-mode", type=int, default=2, help="目标飞行模式 (1=定高 2=定点 3=程控, 默认=2)")
    args = parser.parse_args()

    if args.target_mode not in (1, 2, 3):
        logger.error(f"无效模式 {args.target_mode}，必须为 1/2/3")
        return

    logger.info("=== Phase B: 飞控指令下发测试 ===")

    fc = FC_Controller()

    try:
        # 连接
        logger.info("正在连接飞控...")
        fc.start_listen_serial(serial_dev=args.port, block_until_connected=True)
        if not fc.wait_for_connection(timeout_s=5):
            logger.error("连接超时！")
            return
        logger.info("飞控已连接")

        # 记录切换前状态
        old_mode = fc.state.mode.value
        old_mode_name = MODE_NAMES.get(old_mode, f"未知({old_mode})")
        logger.info(f"当前模式: mode={old_mode} ({old_mode_name}), unlock={fc.state.unlock.value}")

        # 执行模式切换
        target = args.target_mode
        target_name = MODE_NAMES[target]
        logger.info(f"发送 set_flight_mode({target}) → {target_name}...")

        try:
            fc.set_flight_mode(target)
        except ValueError as e:
            logger.error(f"set_flight_mode 参数错误: {e}")
            return

        ok = fc.wait_for_last_command_done(timeout_s=10)
        if not ok:
            logger.error("wait_for_last_command_done 超时！飞控未响应 ACK")
            return
        logger.info("飞控 ACK 确认")

        # 等待状态刷新
        time.sleep(0.5)

        # 验证
        new_mode = fc.state.mode.value
        new_mode_name = MODE_NAMES.get(new_mode, f"未知({new_mode})")
        logger.info(f"切换后模式: mode={new_mode} ({new_mode_name})")

        checks = []
        if new_mode == target:
            checks.append((f"模式已切换 (mode={old_mode}→{new_mode})", True))
        else:
            checks.append((f"模式已切换 (预期={target}, 实际={new_mode})", False))

        # 打印全部状态字段
        s = fc.state
        headers = [
            ("IMU 姿态", f"roll={s.rol.value:.1f}°  pit={s.pit.value:.1f}°  yaw={s.yaw.value:.1f}°"),
            ("高度", f"alt_add={s.alt_add.value}cm (光流激光测距)"),
            ("速度", f"vx={s.vel_x.value}  vy={s.vel_y.value}  vz={s.vel_z.value}  cm/s"),
            ("位置", f"pos_x={s.pos_x.value}  pos_y={s.pos_y.value}  cm"),
            ("电池/解锁", f"bat={s.bat.value:.1f}V  unlock={s.unlock.value}"),
            ("当前指令", f"cid={s.cid.value}  cmd_0={s.cmd_0.value}  cmd_1={s.cmd_1.value}"),
        ]
        for label, value in headers:
            logger.info(f"  {label}: {value}")

        # 汇总
        all_ok = True
        for desc, result in checks:
            status = "PASS" if result else "FAIL"
            if not result:
                all_ok = False
            logger.info(f"[{status}] {desc}")

        if all_ok:
            logger.info("Phase B 全部通过！")
        else:
            logger.warning("Phase B 部分检查未通过，请排查。")
            if new_mode != target:
                logger.info("提示: 确认飞控固件支持定点模式(2)，某些固件仅支持定高模式(1)")

    except RuntimeError as e:
        logger.error(f"飞控设备未找到: {e}")
    except Exception as e:
        logger.error(f"未预期的错误: {type(e).__name__}: {e}")
    finally:
        fc.close()
        logger.info("飞控已断开。")


if __name__ == "__main__":
    main()
