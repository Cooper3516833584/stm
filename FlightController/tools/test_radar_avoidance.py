"""
单雷达在线避障链路地面测试。

验证闭环: D500雷达 → 点云 → 障碍物检测 → 飞控速度指令。

用法:
    # 仅测试雷达端（不连飞控，不发指令）
    python FlightController/tools/test_radar_avoidance.py --no-fc --dry-run

    # 雷达 + 飞控，但不发送实际指令
    python FlightController/tools/test_radar_avoidance.py --dry-run

    # 完整链路（雷达 + 飞控 + 发送指令）
    python FlightController/tools/test_radar_avoidance.py
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


def main() -> None:
    _setup_path()

    from FlightController.Components.FCConnector import FCConnectConfig, connect_fc
    from FlightController.Components.LDRadar_Driver import LD_Radar
    from FlightController.Solutions.LocalPlanner import LocalPlanner, PlannerConfig
    from FlightController.Solutions.Safety import (
        Command as SafeCommand,
        RadarFieldConfig,
        RadarObstacleField,
        SafetyArbiter,
        SafetyConfig,
        flight_status_from_fc,
        flight_health_from_sources,
        send_command_safely,
    )
    import numpy as np
    from loguru import logger

    parser = argparse.ArgumentParser(
        description="单雷达在线避障链路地面测试",
    )
    parser.add_argument(
        "--port",
        default="/dev/ttySTM4",
        help="雷达串口路径 (默认: /dev/ttySTM4)",
    )
    parser.add_argument(
        "--fc-port",
        default=None,
        help="飞控串口路径 (默认: 自动探测)",
    )
    parser.add_argument(
        "--no-fc",
        action="store_true",
        help="不连接飞控，仅测试雷达端",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="不发送实际飞控指令（但仍连接飞控读取状态）",
    )
    parser.add_argument(
        "--enable-flight",
        action="store_true",
        help="Explicitly allow non-zero velocity commands to be sent to FC",
    )
    parser.add_argument(
        "--max-distance-cm",
        type=float,
        default=300.0,
        help="障碍物检测最大距离/cm (默认: 300)",
    )
    parser.add_argument(
        "--stop-distance-cm",
        type=float,
        default=80.0,
        help="急停距离/cm (默认: 80)",
    )
    parser.add_argument(
        "--slow-distance-cm",
        type=float,
        default=150.0,
        help="减速距离/cm (默认: 150)",
    )
    parser.add_argument(
        "--cruise-speed-cm-s",
        type=float,
        default=30.0,
        help="巡航速度/cm/s (默认: 30)",
    )
    parser.add_argument(
        "--corridor-half-width-cm",
        type=float,
        default=50.0,
        help="前方走廊半宽/cm (默认: 50)",
    )
    parser.add_argument(
        "--min-distance-cm",
        type=float,
        default=0.0,
        help="障碍物最小检测距离/cm，过滤雷达自反射噪点 (默认: 0)",
    )
    parser.add_argument(
        "--debug-dump",
        action="store_true",
        help="每次循环打印前方走廊内的原始点云数据 (用于诊断)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="性能分析模式: 测量每级管道的延迟并报告数据新鲜度",
    )
    parser.add_argument(
        "--loop-hz",
        type=float,
        default=10.0,
        help="主循环频率/Hz (默认: 10)",
    )
    parser.add_argument(
        "--radar-timeout-s",
        type=float,
        default=0.5,
        help="雷达真实帧超时 watchdog/s (默认: 0.5)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="将日志同时写入文件 (用于 tail -f 实时监控，绕过SSH缓冲)",
    )
    parser.add_argument(
        "--raw-latency",
        action="store_true",
        help="用雷达包自带时间戳估算真实串口/解析积压，区别于 Map_Circle 数据年龄",
    )
    parser.add_argument(
        "--raw-latency-stdout",
        action="store_true",
        help="将 RAW_LATENCY 诊断行用 print(..., flush=True) 直接输出，绕过 loguru 文件队列",
    )
    args = parser.parse_args()

    # ---------- 日志文件输出 (绕过SSH缓冲) ----------
    if args.log_file:
        logger.add(args.log_file, level="DEBUG", format="{time} | {level: <8} | {message}", enqueue=True)
        logger.info(f"日志文件输出已启用 (异步): {args.log_file}")

    logger.info(f"正在连接雷达 {args.port} ...")
    radar = LD_Radar(
        name="Avoidance_Test",
        index=0,
        mount_xy_cm=(0.0, 0.0),
        mount_yaw_deg=0.0,
    )
    try:
        radar.start(com=args.port, radar_type="D500")
    except (RuntimeError, OSError) as e:
        logger.error(f"雷达串口启动失败: {e}")
        return

    # ---------- 初始化飞控 ----------
    fc = None
    if not args.no_fc:
        logger.info("正在连接飞控...")
        try:
            fc = connect_fc(FCConnectConfig(port=args.fc_port, mode=2, timeout_s=10.0))
            state = fc.state
            logger.info(
                f"FC 已连接 | mode={state.mode.value} unlock={state.unlock.value} "
                f"bat={state.bat.value:.1f}V alt={state.alt_add.value}cm"
            )
        except Exception as e:
            logger.error(f"飞控连接失败: {e}")
            fc = None

    # ---------- 初始化规划器 ----------
    planner_config = PlannerConfig(
        enable_free_flight=True,
        free_flight_speed_cm_s=args.cruise_speed_cm_s,
        max_speed_cm_s=50.0,
        obstacle_stop_distance_cm=args.stop_distance_cm,
        obstacle_slow_distance_cm=args.slow_distance_cm,
        forward_corridor_half_width_cm=args.corridor_half_width_cm,
        min_obstacle_distance_cm=args.min_distance_cm,
    )
    planner = LocalPlanner(config=planner_config)
    radar_field = RadarObstacleField(
        RadarFieldConfig(
            max_distance_cm=args.max_distance_cm,
            forward_corridor_half_width_cm=args.corridor_half_width_cm,
            min_obstacle_distance_cm=args.min_distance_cm,
        )
    )
    safety = SafetyArbiter(
        SafetyConfig(
            radar_timeout_s=0.5,
            obstacle_stop_distance_cm=args.stop_distance_cm,
            obstacle_slow_distance_cm=args.slow_distance_cm,
        )
    )
    send_enabled = bool(args.enable_flight and fc is not None)
    if not send_enabled:
        logger.info("[SAFETY] dry-run active; use --enable-flight to send non-zero FC velocity")

    logger.info(
        f"规划器参数: stop<{args.stop_distance_cm}cm "
        f"slow<{args.slow_distance_cm}cm "
        f"cruise={args.cruise_speed_cm_s}cm/s "
        f"corridor=±{args.corridor_half_width_cm}cm"
    )

    # ---------- 等待雷达预热 ----------
    logger.info("等待雷达预热 3 秒...")
    time.sleep(3)

    # 降低 Map_Circle 超时: 10Hz 扫描周期=100ms, 超时设为 150ms 足够
    radar.map.timeout_time = 0.15
    logger.info(f"Map_Circle 超时已调整为 {radar.map.timeout_time}s")

    # 运行时间 vs 延迟采样点 (用于检测延迟增长)
    _uptime_delay_samples: list[tuple[float, float, float]] = []

    if not radar.connected:
        logger.error("雷达未连接！请检查 TX 引脚和 PWM 供电。")
        radar.stop()
        if fc is not None:
            fc.close()
        return

    # ---------- 主循环 ----------
    period = 1.0 / max(args.loop_hz, 0.1)
    logger.info(f"开始避障主循环 @ {args.loop_hz}Hz (周期 {period:.3f}s)")
    logger.info("按 Ctrl+C 停止...")

    loop_count = 0
    _last_update_count = 0
    _last_profile_time = time.perf_counter()
    _last_forward_dist: float | None = None
    _step_start_time: float | None = None
    try:
        while True:
            t_start = time.perf_counter()

            # 1. 检查雷达连接状态
            if not radar.connected:
                logger.warning("雷达断连！")
                if fc is not None and args.enable_flight:
                    health = flight_health_from_sources(
                        fc=fc,
                        radar=radar,
                        radar_timeout_s=args.radar_timeout_s,
                    )
                    send_command_safely(
                        fc,
                        SafeCommand.zero("radar_disconnected"),
                        safety,
                        health,
                        dry_run=not send_enabled,
                    )
                time.sleep(0.5)
                continue

            # 2. 获取机体坐标系下的障碍点云
            t_before_get = time.perf_counter()
            obstacles = radar.get_points_body_cm(
                max_distance_cm=args.max_distance_cm
            )
            radar_field.update(obstacles, t_start)
            t_after_get = time.perf_counter()

            # 3. 计算前方最近障碍物距离
            t_before_plan = time.perf_counter()
            forward_dist = radar_field.nearest_forward_obstacle_cm()
            t_after_plan = time.perf_counter()

            # 3b. debug: dump 前方走廊内的原始点云
            if args.debug_dump and obstacles.size > 0:
                pts = np.asarray(obstacles, dtype=float).reshape(-1, 2)
                half_w = args.corridor_half_width_cm
                min_d = args.min_distance_cm
                forward_pts = pts[
                    (pts[:, 0] > min_d) & (np.abs(pts[:, 1]) < half_w)
                ]
                all_forward = pts[pts[:, 0] > 0]
                logger.info(
                    f"[#{loop_count:04d}] DUMP: 总点云={len(pts)} | "
                    f"前方全部(x>0)={len(all_forward)}点 | "
                    f"前方走廊(x>{min_d} & |y|<{half_w})={len(forward_pts)}点"
                )
                if forward_pts.size > 0:
                    sorted_idx = np.argsort(forward_pts[:, 0])
                    closest = forward_pts[sorted_idx][:10]
                    for i, (x, y) in enumerate(closest):
                        logger.info(f"  -> 第{i+1}近: x={x:.1f}cm, y={y:.1f}cm, dist={np.hypot(x,y):.1f}cm")

            # 3b. 步进响应延迟测量: 障碍物距离发生显著变化时记录
            dist_now = forward_dist
            if _last_forward_dist is None or dist_now is None:
                state_changed = _last_forward_dist is not None or dist_now is not None
            else:
                state_changed = abs(dist_now - _last_forward_dist) > 15.0
            if state_changed:
                old_str = f"{_last_forward_dist:.0f}cm" if _last_forward_dist is not None else "无"
                new_str = f"{dist_now:.0f}cm" if dist_now is not None else "无"
                logger.info(f"[LATENCY] 障碍物变化: {old_str} → {new_str} | 帧#{loop_count}")
                _last_forward_dist = dist_now

            # 4. 规划避障决策
            local_command = planner.plan(obstacles_body_cm=radar_field.points_body_cm, target=None)
            desired_command = SafeCommand(
                local_command.vx_cm_s,
                local_command.vy_cm_s,
                local_command.vz_cm_s,
                local_command.yaw_rate_deg_s,
                local_command.reason,
            )
            health = flight_health_from_sources(
                fc=fc,
                radar=radar,
                radar_timeout_s=args.radar_timeout_s,
            )
            safety_result = safety.filter(
                desired_command,
                flight=flight_status_from_fc(fc),
                radar_connected=bool(radar.connected and radar.is_fresh(max_age_s=args.radar_timeout_s)),
                radar_age_s=health.radar_max_age_s,
                radar_field=radar_field,
                enable_flight=send_enabled,
            )
            command = safety_result.command
            forward_dist = safety_result.nearest_forward_obstacle_cm
            send_decision = send_command_safely(
                fc,
                command,
                safety,
                health,
                dry_run=not send_enabled,
            )

            # 5. 指令已通过 SafetyArbiter + send_command_safely() 闸门

            # 6. 日志输出
            dist_str = f"{forward_dist:.0f}cm" if forward_dist is not None else "无"
            if loop_count % 10 == 0:
                age_s = radar.get_last_frame_age_s()
                age_str = "None" if age_s is None else f"{age_s * 1000:.0f}ms"
                fc_state_str = ""
                if fc is not None:
                    try:
                        s = fc.state
                        fc_state_str = (
                            f"FC[mode={s.mode.value} unlock={s.unlock.value} "
                            f"bat={s.bat.value:.1f}V alt={s.alt_add.value}cm]"
                        )
                    except Exception:
                        fc_state_str = "FC[state_read_error]"

                logger.info(
                    f"[#{loop_count:04d}] {fc_state_str} | "
                    f"点云={len(obstacles)}点 | 前方={dist_str} | "
                    f"指令=(vx={command.vx_cm_s:.0f}, vy={command.vy_cm_s:.0f}, "
                    f"vz={command.vz_cm_s:.0f}, yaw={command.yaw_rate_deg_s:.0f}) | "
                    f"原因={command.reason}"
                )
                logger.info(f"RADAR_HEALTH last_frame_age={age_str} connected={radar.connected}")
            else:
                logger.debug(
                    f"[#{loop_count:04d}] 前方={dist_str} "
                    f"vx={command.vx_cm_s:.0f} reason={command.reason}"
                )

            # 6b. profile: 计时分析
            if (args.profile or args.raw_latency) and loop_count % 30 == 0:
                now = time.perf_counter()
                elapsed_profile = now - _last_profile_time
                _last_profile_time = now

                # 数据新鲜度: 只统计有效数据(data!=-1)的时间戳
                ts = radar.map.time_stamp
                data = radar.map.data
                valid_mask = (ts > 0) & (data != -1)
                valid_ts = ts[valid_mask]
                data_age_min = 999.0
                data_age_max = 0.0
                if len(valid_ts) > 0:
                    ages = now - valid_ts
                    data_age_min = float(np.min(ages))
                    data_age_max = float(np.max(ages))

                # 串口吞吐量: update_count 变化率
                uc = radar.map.update_count
                uc_rate = (uc - _last_update_count) / max(elapsed_profile, 0.001)
                _last_update_count = uc

                # 有效点云率: 理论最小值 vs 实际
                rpm = radar.map.rotation_spd
                pps_expected = rpm / 60 * 360 / 12  # 理论包数/秒

                # 单次 pipeline 耗时
                get_time_ms = (t_after_get - t_before_get) * 1000
                plan_time_ms = (t_after_plan - t_before_plan) * 1000
                total_time_ms = (time.perf_counter() - t_start) * 1000

                crc_err = radar._crc_errors
                crc_rate = crc_err / max(elapsed_profile, 0.001)
                throughput_pct = uc_rate / max(pps_expected, 1) * 100
                raw_latency_text = ""
                if args.raw_latency:
                    raw_stats = radar.get_radar_latency_stats(reset_interval=True)
                    raw_latency_text = (
                        f"雷达帧年龄: 当前={raw_stats['latest_ms']:.0f}ms "
                        f"区间峰值={raw_stats['interval_max_ms']:.0f}ms "
                        f"全局峰值={raw_stats['max_ms']:.0f}ms | "
                        f"设备钟速={raw_stats['device_rate_pct']:.2f}% "
                        f"钟差={raw_stats['clock_drift_ms']:.0f}ms | "
                        f"串口buf峰值={raw_stats['in_waiting_peak']}B "
                        f"解析buf={raw_stats['parse_buffer_bytes']}B "
                        f"有效帧={raw_stats['serial_frames_ok']}"
                    )

                if args.profile:
                    logger.info(
                        f"[PROFILE] 吞吐={uc_rate:.0f}/{pps_expected:.0f}包/s ({throughput_pct:.0f}%) | "
                        f"CRC错误={crc_err}次 ({crc_rate:.1f}/s) | "
                        f"数据年龄: 最新={data_age_min*1000:.0f}ms 最旧={data_age_max*1000:.0f}ms | "
                        f"耗时: get={get_time_ms:.1f}ms plan={plan_time_ms:.1f}ms total={total_time_ms:.1f}ms"
                    )
                if raw_latency_text:
                    logger.info(f"[RAW_LATENCY] {raw_latency_text}")
                    if args.raw_latency_stdout:
                        print(
                            f"{time.strftime('%Y-%m-%d %H:%M:%S')} [RAW_LATENCY] {raw_latency_text}",
                            flush=True,
                        )
                if args.profile and data_age_max > 1.0:
                    logger.warning(
                        f"[PROFILE] ⚠ 数据年龄偏大 (最旧={data_age_max:.1f}s)! "
                        f"障碍物变化需等待 {data_age_max:.1f}s 才能被检测到"
                    )

                # 采样运行时间 vs 延迟 (每10个 profile 输出一次趋势)
                if args.profile:
                    _uptime_delay_samples.append((radar.start_time, data_age_max, data_age_min))
                if args.profile and len(_uptime_delay_samples) >= 10:
                    run_times = [now - s[0] for s in _uptime_delay_samples]
                    max_ages = [s[1] for s in _uptime_delay_samples]
                    min_ages = [s[2] for s in _uptime_delay_samples]
                    # 趋势检测: 比较前后半段的平均值
                    if len(run_times) >= 4:
                        n = len(max_ages)
                        mid = n // 2
                        first_avg = sum(max_ages[:mid]) / mid
                        second_avg = sum(max_ages[mid:]) / (n - mid)
                        if first_avg > 0.001:
                            change = (second_avg - first_avg) / first_avg
                        else:
                            change = 0.0
                        trend = "↑增长" if change > 0.15 else ("↓下降" if change < -0.15 else "→稳定")
                        logger.info(
                            f"[DELAY_TREND] 运行{run_times[-1]:.0f}s | "
                            f"最旧: {max_ages[0]*1000:.0f}→{max_ages[-1]*1000:.0f}ms (趋势{trend}) | "
                            f"最新: {min_ages[0]*1000:.0f}→{min_ages[-1]*1000:.0f}ms"
                        )
                    _uptime_delay_samples.clear()

            loop_count += 1

            # 7. 控制循环频率
            elapsed = time.perf_counter() - t_start
            remaining = period - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        logger.info("收到中断信号，正在安全退出...")
    finally:
        # 发送零速度
        if fc is not None and args.enable_flight:
            logger.info("发送零速度指令...")
            health = flight_health_from_sources(
                fc=fc,
                radar=radar,
                radar_timeout_s=args.radar_timeout_s,
            )
            send_command_safely(
                fc,
                SafeCommand.zero("shutdown"),
                safety,
                health,
                dry_run=not send_enabled,
            )
            time.sleep(0.1)

        radar.stop()
        if fc is not None:
            fc.close()

        logger.info(f"测试结束 | 共运行 {loop_count} 个循环")


if __name__ == "__main__":
    main()
