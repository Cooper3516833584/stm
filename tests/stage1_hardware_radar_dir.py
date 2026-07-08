"""Stage 1 (Hardware): Radar physical direction verification on STM32MP257.

Run this ON THE DRONE (not PC):
    cd ~/Desktop/ObstacleAvoidanceDrone
    source /usr/local/UFC_venv/bin/activate
    PYTHONPATH=. python tests/stage1_hardware_radar_dir.py

Requires: D500 radar connected to /dev/ttySTM4 (upper) and /dev/ttySTM9 (lower).
Does NOT require flight controller.

SAFETY: This script only reads radar data. No commands are sent.
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

# ── Helper: angle in body frame ─────────────────────────────────────────

def body_angle_deg(points: np.ndarray) -> np.ndarray:
    """atan2(y, x) in degrees. + = left, - = right."""
    return np.rad2deg(np.arctan2(points[:, 1], points[:, 0]))


def classify_quadrant(points: np.ndarray) -> dict:
    """Classify points into quadrants in body frame.

    Returns dict with keys: front, front_left, left, back_left,
    back, back_right, right, front_right
    """
    angles = body_angle_deg(points)
    dists = np.linalg.norm(points, axis=1)

    result = {}
    bins = [
        ("front",        -15,   15),
        ("front_right",   15,   75),
        ("right",         75,  105),
        ("back_right",   105,  165),
        ("back",         165,  180),
        ("back",        -180, -165),
        ("back_left",   -165, -105),
        ("left",        -105,  -75),
        ("front_left",   -75,  -15),
    ]

    for name, lo, hi in bins:
        mask = (angles >= lo) & (angles < hi) & (dists > 0)
        count = np.sum(mask)
        nearest = float(dists[mask].min()) if count > 0 else float("inf")
        result[name] = {"count": count, "nearest_cm": nearest}

    return result


def print_direction_report(points: np.ndarray, label: str):
    """Print a human-readable direction report."""
    q = classify_quadrant(points)
    print(f"\n  --- {label} ---")
    print(f"  Total points: {len(points)}")

    # Front is most important
    front_pts = q["front"]["count"]
    fl_pts = q["front_left"]["count"]
    fr_pts = q["front_right"]["count"]
    left_pts = q["left"]["count"] + q["back_left"]["count"]
    right_pts = q["right"]["count"] + q["back_right"]["count"]

    print(f"  Front:       {front_pts:4d} pts  nearest={q['front']['nearest_cm']:.0f} cm")
    print(f"  Front-left:  {fl_pts:4d} pts  nearest={q['front_left']['nearest_cm']:.0f} cm")
    print(f"  Front-right: {fr_pts:4d} pts  nearest={q['front_right']['nearest_cm']:.0f} cm")
    print(f"  Left side:   {left_pts:4d} pts")
    print(f"  Right side:  {right_pts:4d} pts")

    # Direction balance check
    if front_pts > 50 and (fl_pts > 0 or fr_pts > 0):
        if fl_pts > fr_pts * 1.5:
            print(f"  >>> More points on LEFT -- if these are obstacles, expect RIGHT turn <<<")
        elif fr_pts > fl_pts * 1.5:
            print(f"  >>> More points on RIGHT -- if these are obstacles, expect LEFT turn <<<")


# ── Manual test procedure ────────────────────────────────────────────────

def run_continuous_monitor(upper_port="/dev/ttySTM4", lower_port="/dev/ttySTM9"):
    """Run continuous direction monitoring. Press Ctrl+C to stop."""
    from FlightController.Components.LDRadar_Driver import LD_Radar

    print("=" * 65)
    print("Stage 1 -- Radar Direction Monitor")
    print("=" * 65)
    print()
    print("This tool shows which body-frame quadrant each radar sees obstacles.")
    print()
    print("Manual verification procedure:")
    print("  1. Stand directly in front of the radar, ~1m away")
    print("     -> Watch 'Front' count increase, angle near 0deg")
    print()
    print("  2. Move to radar's LEFT side (your left when facing forward)")
    print("     -> Watch 'Front-left' or 'Left' increase, angle POSITIVE (+Y)")
    print()
    print("  3. Move to radar's RIGHT side")
    print("     -> Watch 'Front-right' or 'Right' increase, angle NEGATIVE (-Y)")
    print()
    print("  4. Place a box at ~1.5m, 45deg left of center")
    print("     -> Verify it shows in 'front-left' quadrant (x>0, y>0)")
    print()
    print("  5. Place a box at ~1.5m, 45deg right of center")
    print("     -> Verify it shows in 'front-right' quadrant (x>0, y<0)")
    print()

    # Start upper radar
    print(f"Starting upper radar on {upper_port}...")
    upper = LD_Radar(name="upper", index=0, mount_xy_cm=(0.0, 0.0),
                     mount_yaw_deg=0.0, mount_mirror_y=False)
    upper.start(com=upper_port)
    time.sleep(1)

    lower = None
    if lower_port is not None and os.path.exists(lower_port):
        print(f"Starting lower radar on {lower_port}...")
        lower = LD_Radar(name="lower", index=1, mount_xy_cm=(0.96, 0.15),
                         mount_yaw_deg=0.0, mount_mirror_y=True)
        lower.start(com=lower_port)
        time.sleep(1)

    print()
    print("Waiting for radar data (2 seconds)...")
    time.sleep(2)

    print()
    print("=" * 65)
    print("MONITORING -- move obstacles and watch the report")
    print("Press Ctrl+C to stop")
    print("=" * 65)

    try:
        frame = 0
        while True:
            frame += 1
            report_lines = [f"\n=== Frame {frame} ==="]

            # Upper radar
            pts_u = upper.get_points_body_cm(max_distance_cm=300)
            if len(pts_u) > 0:
                body_angles_u = body_angle_deg(pts_u)
                report_lines.append(f"\n[UPPER] {len(pts_u)} pts")
                report_lines.append(f"  angle range: {body_angles_u.min():+.0f} to {body_angles_u.max():+.0f} deg")
                report_lines.append(f"  x range: {pts_u[:, 0].min():+.0f} to {pts_u[:, 0].max():+.0f} cm")
                report_lines.append(f"  y range: {pts_u[:, 1].min():+.0f} to {pts_u[:, 1].max():+.0f} cm")

                # Quick stats
                front_mask = (np.abs(body_angles_u) < 30) & (pts_u[:, 0] > 10)
                left_mask = (body_angles_u > 10) & (pts_u[:, 0] > 10)
                right_mask = (body_angles_u < -10) & (pts_u[:, 0] > 10)

                if np.any(front_mask):
                    d = pts_u[front_mask, 0].min()
                    report_lines.append(f"  FRONT nearest: {d:.0f} cm")
                if np.any(left_mask):
                    report_lines.append(f"  LEFT points: {np.sum(left_mask)}")
                if np.any(right_mask):
                    report_lines.append(f"  RIGHT points: {np.sum(right_mask)}")

                # Key diagnostic: left/right balance
                if np.any(left_mask) and np.any(right_mask):
                    ratio = np.sum(left_mask) / max(1, np.sum(right_mask))
                    report_lines.append(f"  L/R ratio: {ratio:.2f} (1.0 = balanced)")
            else:
                report_lines.append("\n[UPPER] NO POINTS")

            # Lower radar
            if lower is not None:
                pts_l = lower.get_points_body_cm(max_distance_cm=300)
                if len(pts_l) > 0:
                    body_angles_l = body_angle_deg(pts_l)
                    report_lines.append(f"\n[LOWER] {len(pts_l)} pts")
                    report_lines.append(f"  angle range: {body_angles_l.min():+.0f} to {body_angles_l.max():+.0f} deg")

                    front_l = (np.abs(body_angles_l) < 30) & (pts_l[:, 0] > 10)
                    if np.any(front_l):
                        report_lines.append(f"  FRONT nearest: {pts_l[front_l, 0].min():.0f} cm")
                else:
                    report_lines.append("\n[LOWER] NO POINTS")

            # Print report
            print("\n".join(report_lines))

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        upper.stop()
        if lower is not None:
            lower.stop()
        print("Radars stopped.")
        print()
        print("=" * 65)
        print("Verification Checklist:")
        print("=" * 65)
        print("[ ] Obstacle in FRONT -> appears in 'FRONT' (angle ~0deg)")
        print("[ ] Obstacle on LEFT  -> appears with POSITIVE body angle (+Y)")
        print("[ ] Obstacle on RIGHT -> appears with NEGATIVE body angle (-Y)")
        print()
        print("If all three checked: radar coordinates are CORRECT.")
        print("The mirror effect is from the yaw sign bug in _yaw_command().")


def run_single_snapshot(upper_port="/dev/ttySTM4", lower_port="/dev/ttySTM9"):
    """Take a single snapshot and print detailed direction report."""
    from FlightController.Components.LDRadar_Driver import LD_Radar

    print("Taking radar snapshot (2s warmup)...")
    upper = LD_Radar(name="upper", index=0, mount_xy_cm=(0.0, 0.0),
                     mount_yaw_deg=0.0, mount_mirror_y=False)
    upper.start(com=upper_port)
    time.sleep(2)

    pts_u = upper.get_points_body_cm(max_distance_cm=300)
    upper.stop()

    print_direction_report(pts_u, "UPPER RADAR")

    if lower_port is not None and os.path.exists(lower_port):
        lower = LD_Radar(name="lower", index=1, mount_xy_cm=(0.96, 0.15),
                         mount_yaw_deg=0.0, mount_mirror_y=True)
        lower.start(com=lower_port)
        time.sleep(2)
        pts_l = lower.get_points_body_cm(max_distance_cm=300)
        lower.stop()
        print_direction_report(pts_l, "LOWER RADAR")

    print()
    print("Now run the same test with an obstacle (person/box) at a known position.")
    print("Example: stand 1m directly in front -> verify FRONT quadrant gets many points.")


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 1: Radar physical direction verification")
    parser.add_argument("--upper-port", default="/dev/ttySTM4")
    parser.add_argument("--lower-port", default="/dev/ttySTM9")
    parser.add_argument("--snapshot", action="store_true",
                        help="Take a single snapshot instead of continuous monitoring")
    parser.add_argument("--no-lower", action="store_true",
                        help="Skip lower radar (if not connected)")
    args = parser.parse_args()

    lower = None if args.no_lower else args.lower_port

    if args.snapshot:
        run_single_snapshot(args.upper_port, lower)
    else:
        run_continuous_monitor(args.upper_port, lower)
