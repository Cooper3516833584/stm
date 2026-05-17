"""
Phase C: 飞控实时控制通信测试。

验证 send_realtime_control_data() 协议层 — 电机未接，仅测通信不测物理响应。

用法:
    PYTHONPATH=. python debug/test_fc_realtime.py
    PYTHONPATH=. python debug/test_fc_realtime.py --count 20 --speed 15
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


def main() -> None:
    _setup_path()

    from FlightController import FC_Controller
    from loguru import logger

    parser = argparse.ArgumentParser(description="飞控实时控制通信测试 (Phase C)")
    parser.add_argument("--port", default=None, help="飞控串口路径 (默认: 自动探测)")
    parser.add_argument("--count", type=int, default=10, help="发送指令数量 (默认=10)")
    parser.add_argument("--speed", type=int, default=10, help="测试速度 cm/s (默认=10)")
    parser.add_argument("--interval", type=float, default=0.2, help="发送间隔秒 (默认=0.2)")
    args = parser.parse_args()

    logger.info("=== Phase C: 飞控实时控制通信测试 ===")

    fc = FC_Controller()

    try:
        # 连接
        logger.info("正在连接飞控...")
        fc.start_listen_serial(serial_dev=args.port, block_until_connected=True)
        if not fc.wait_for_connection(timeout_s=5):
            logger.error("连接超时！")
            return
        logger.info("飞控已连接")

        # 确认当前模式
        logger.info(f"当前模式: mode={fc.state.mode.value}, unlock={fc.state.unlock.value}")

        # 切换到定点模式（实时控制需要 HOLD_POS）
        if fc.state.mode.value != 2:
            logger.info("切换到定点模式 (mode=2)...")
            fc.set_flight_mode(2)
            if not fc.wait_for_last_command_done(timeout_s=10):
                logger.error("模式切换 ACK 超时！")
                return
            logger.info(f"模式已切换: mode={fc.state.mode.value}")

        logger.info("开始发送实时控制指令...")

        sent = 0
        errors = 0
        disconnects = 0
        t_start = time.perf_counter()

        for i in range(args.count):
            # 交替发送正/零速度，测试协议帧封装
            vx = args.speed if i < args.count // 2 else 0

            try:
                fc.send_realtime_control_data(vel_x=vx, vel_y=0, vel_z=0, yaw=0)
                sent += 1
            except Exception as e:
                errors += 1
                logger.error(f"send_realtime_control_data #{i} 异常: {type(e).__name__}: {e}")

            if not fc.connected:
                disconnects += 1
                logger.error(f"发送 #{i} 后 connected=False！")

            if (i + 1) % 5 == 0 or i == 0:
                logger.info(f"  发送 #{i}: vx={vx}, connected={fc.connected}, mode={fc.state.mode.value}")

            time.sleep(args.interval)

        elapsed = time.perf_counter() - t_start

        # 零速归位
        fc.send_realtime_control_data(vel_x=0, vel_y=0, vel_z=0, yaw=0)
        time.sleep(0.1)

        # 汇总
        logger.info(f"--- 发送完成 ---")
        logger.info(f"发送: {sent}/{args.count} 成功, {errors} 异常, {disconnects} 断连")
        logger.info(f"耗时: {elapsed:.1f}s, 速率: {sent/elapsed:.1f} 包/秒")
        logger.info(f"最终状态: connected={fc.connected}, mode={fc.state.mode.value}")

        # 验证
        checks = [
            ("全部指令发送成功", sent == args.count),
            ("零异常", errors == 0),
            ("全程无断连", disconnects == 0),
            ("最终 connected=True", fc.connected),
        ]
        all_ok = True
        for desc, result in checks:
            status = "PASS" if result else "FAIL"
            if not result:
                all_ok = False
            logger.info(f"[{status}] {desc}")

        if all_ok:
            logger.info("Phase C 全部通过！")
        else:
            logger.warning("Phase C 部分检查未通过，请排查。")

    except RuntimeError as e:
        logger.error(f"飞控设备未找到: {e}")
    except Exception as e:
        logger.error(f"未预期的错误: {type(e).__name__}: {e}")
    finally:
        fc.close()
        logger.info("飞控已断开。")


if __name__ == "__main__":
    main()
