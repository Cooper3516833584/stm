"""
双雷达统一避障链路测试。

接线: 上雷达 UART4 /dev/ttySTM4 (index=0), 下雷达 UART9 /dev/ttySTM9 (index=1, 倒装 mirror_y)
机身范围 -25cm < x < 25cm, -25cm < y < 25cm 内的点自动屏蔽。

用法:
    # 仅测双雷达连通性和点云统计
    PYTHONPATH=. python -u FlightController/tools/test_dual_radar.py --no-fc

    # 双雷达 + 飞控（不发送指令，只读状态）
    PYTHONPATH=. python -u FlightController/tools/test_dual_radar.py --dry-run

    # 完整链路
    PYTHONPATH=. python -u FlightController/tools/test_dual_radar.py
"""

import argparse
import sys
import time
from pathlib import Path


def _setup_path() -> None:
    root = Path(__file__).resolve().parents[1]
    for p in (root, root.parent):
        value = str(p)
        if value not in sys.path:
            sys.path.insert(0, value)


def _body_mask(points: "np.ndarray", x_half: float = 25.0, y_half: float = 25.0) -> "np.ndarray":
    """滤除机身范围内的自反射点。"""
    return points[~((abs(points[:, 0]) < x_half) & (abs(points[:, 1]) < y_half))]


def main() -> None:
    _setup_path()

    from FlightController import FC_Controller
    from FlightController.Components import MultiRadar, RadarConfig
    from FlightController.Solutions.LocalPlanner import LocalPlanner, PlannerConfig
    import numpy as np
    from loguru import logger

    parser = argparse.ArgumentParser(description="双雷达统一避障链路测试")
    parser.add_argument("--upper-port", default="/dev/ttySTM4", help="上雷达串口 (默认: /dev/ttySTM4)")
    parser.add_argument("--lower-port", default="/dev/ttySTM9", help="下雷达串口 (默认: /dev/ttySTM9)")
    parser.add_argument("--fc-port", default=None, help="飞控串口 (默认: 自动探测)")
    parser.add_argument("--no-fc", action="store_true", help="不连接飞控")
    parser.add_argument("--dry-run", action="store_true", help="连接飞控但不发送指令")
    parser.add_argument("--max-distance-cm", type=float, default=300.0, help="障碍物检测最大距离/cm")
    parser.add_argument("--stop-distance-cm", type=float, default=80.0, help="急停距离/cm")
    parser.add_argument("--slow-distance-cm", type=float, default=150.0, help="减速距离/cm")
    parser.add_argument("--corridor-half-width-cm", type=float, default=50.0, help="前方走廊半宽/cm")
    parser.add_argument("--body-x-half-cm", type=float, default=25.0, help="机身 X 屏蔽半宽/cm")
    parser.add_argument("--body-y-half-cm", type=float, default=25.0, help="机身 Y 屏蔽半宽/cm")
    parser.add_argument("--profile", action="store_true", help="打印每帧耗时")
    parser.add_argument("--debug-dump", action="store_true", help="打印前方走廊原始点云数据")
    parser.add_argument("--log-file", default=None, help="日志文件路径")
    args = parser.parse_args()

    if args.log_file:
        logger.add(args.log_file, enqueue=True, level="DEBUG")

    # ── 飞控 ──
    fc = None
    if not args.no_fc:
        fc = FC_Controller()
        fc.start_listen_serial(block_until_connected=True, explicit_port=args.fc_port)
        fc.wait_for_connection(timeout_s=10)
        fc.set_flight_mode(2)
        fc.wait_for_last_command_done()
        logger.info("[FC] 连接成功，已切定点模式")

    # ── 双雷达 ──
    configs = [
        RadarConfig(
            name="upper",
            index=0,
            mount_xy_cm=(0.0, 0.0),
            mount_yaw_deg=0.0,
            port=args.upper_port,
        ),
        RadarConfig(
            name="lower",
            index=1,
            mount_xy_cm=(0.96, 0.15),
            mount_yaw_deg=0.0,
            mount_mirror_y=True,
            port=args.lower_port,
        ),
    ]
    multi_radar = MultiRadar(configs)
    planner_config = PlannerConfig(
        enable_free_flight=True,
        free_flight_speed_cm_s=20.0,
        max_speed_cm_s=50.0,
        obstacle_stop_distance_cm=args.stop_distance_cm,
        obstacle_slow_distance_cm=args.slow_distance_cm,
        forward_corridor_half_width_cm=args.corridor_half_width_cm,
    )
    planner = LocalPlanner(config=planner_config)

    try:
        multi_radar.start()
        logger.info("[DUAL-RADAR] 双雷达已启动，等待数据就绪...")

        # 等待连接 + 诊断输出
        wait_start = time.perf_counter()
        while not multi_radar.connected:
            time.sleep(1.0)
            elapsed = time.perf_counter() - wait_start
            for radar in multi_radar.radars:
                stats = radar.get_radar_latency_stats()
                logger.info(
                    f"[WAIT {elapsed:.0f}s] {radar.name}: "
                    f"connected={radar.connected} "
                    f"bytes_read={stats['serial_bytes_read']} "
                    f"frames_ok={stats['serial_frames_ok']} "
                    f"crc_errors={stats['crc_errors']} "
                    f"samples={stats['samples']} "
                    f"parse_buf={stats['parse_buffer_bytes']}B"
                )
            if elapsed > 10.0:
                logger.error("[DUAL-RADAR] 连接超时(>10s)，可能某颗雷达无数据")
                break
        if not multi_radar.connected:
            logger.warning("[DUAL-RADAR] 仅部分雷达连接，继续运行...")

        # 采样窗口（~0.5s 汇总一次，即 ~50 帧 @10ms sleep）
        loop_count = 0
        loop_times = []       # 每帧耗时 ms
        upper_counts = []     # 上雷达过滤后点数
        lower_counts = []     # 下雷达过滤后点数
        blocked_counts = []   # 机身屏蔽点数
        obs_distances = []    # 前方障碍物距离

        while True:
            t0 = time.perf_counter()

            # ① 获取融合点云（仅一次坐标变换）
            all_points = multi_radar.get_obstacle_points_body_cm(
                max_distance_cm=args.max_distance_cm
            )

            # ② 屏蔽机身范围
            filtered_points = _body_mask(
                all_points,
                x_half=args.body_x_half_cm,
                y_half=args.body_y_half_cm,
            )

            # ③ 各雷达点数统计（直接读 map.data，不做坐标变换）
            raw_upper = int(np.count_nonzero(multi_radar.radars[0].map.data != -1))
            raw_lower = int(np.count_nonzero(multi_radar.radars[1].map.data != -1))

            # ④ 避障决策
            command = planner.plan(obstacles_body_cm=filtered_points, target=None)
            obstacle_cm = planner._nearest_forward_obstacle_cm(filtered_points)

            # ⑤ 发送飞控指令
            if fc is not None and not args.dry_run:
                fc.send_realtime_control_data(
                    vel_x=round(command.vx_cm_s),
                    vel_y=round(command.vy_cm_s),
                    vel_z=round(command.vz_cm_s),
                    yaw=round(command.yaw_rate_deg_s),
                )

            t1 = time.perf_counter()

            # ── 累积采样 ──
            loop_count += 1
            loop_times.append((t1 - t0) * 1000.0)
            upper_counts.append(raw_upper)
            lower_counts.append(raw_lower)
            blocked_counts.append(len(all_points) - len(filtered_points))
            if obstacle_cm is not None:
                obs_distances.append(obstacle_cm)

            # ── 每 50 帧 (~0.5s) 汇总输出 ──
            if loop_count >= 50:
                t_arr = np.array(loop_times)
                u_arr = np.array(upper_counts)
                l_arr = np.array(lower_counts)
                b_arr = np.array(blocked_counts)

                effective_hz = 1000.0 / t_arr.mean() if t_arr.mean() > 0 else 0
                cpu_pct = t_arr.mean() / 10.0 * 100.0  # 相对 10ms 周期

                fc_str = ""
                if fc is not None:
                    s = fc.state
                    fc_str = f"FC[mode={s.mode.value} bat={s.bat.value:.1f}V pit={s.pit.value:.1f}] "

                obs_str = f"前={np.mean(obs_distances):.0f}cm" if obs_distances else "前=无"

                logger.info(
                    f"{fc_str}| "
                    f"融合={u_arr.mean():.0f}+{l_arr.mean():.0f}点 "
                    f"机身={b_arr.mean():.0f} | {obs_str} "
                    f"vx={command.vx_cm_s:.0f} | "
                    f"loop={t_arr.mean():.1f}/{t_arr.max():.1f}/{t_arr.min():.1f}ms "
                    f"(avg/max/min) "
                    f"eff={effective_hz:.0f}Hz cpu≈{cpu_pct:.0f}%"
                )

                if args.debug_dump and len(filtered_points) > 0:
                    fwd = filtered_points[filtered_points[:, 0] > 10]
                    fwd_corridor = fwd[abs(fwd[:, 1]) < args.corridor_half_width_cm]
                    if len(fwd_corridor) > 0:
                        dists = np.linalg.norm(fwd_corridor, axis=1)
                        logger.info(
                            f"[DEBUG] 前方走廊={len(fwd_corridor)}点 "
                            f"最近={dists.min():.0f}cm 最远={dists.max():.0f}cm"
                        )

                loop_count = 0
                loop_times.clear()
                upper_counts.clear()
                lower_counts.clear()
                blocked_counts.clear()
                obs_distances.clear()

            time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("[DUAL-RADAR] 用户中断")
    finally:
        if fc is not None:
            fc.send_realtime_control_data(0, 0, 0, 0)
            fc.close()
        multi_radar.stop()
        logger.info("[DUAL-RADAR] 安全退出")


if __name__ == "__main__":
    main()
