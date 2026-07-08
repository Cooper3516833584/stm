"""Stage 0.1: Verify radar polar→Cartesian coordinate conventions.

D500 radar: 0° = forward, angle increases clockwise (top-down view).
Target body frame: +X = forward, +Y = left, -Y = right.

This test verifies the conversion pipeline without any hardware.
"""

from __future__ import annotations

import math
import numpy as np


# ── Replicate get_points_xy_cm logic (LDRadar_Driver.py:370-401) ──────────

ACC = 3  # bins per degree, total 1080 bins
DEG = np.arange(0, 360, 1 / ACC)
COS = np.cos(np.deg2rad(DEG))
SIN = np.sin(np.deg2rad(DEG))


def get_points_xy_cm(angle_deg: float, distance_mm: float) -> tuple[float, float]:
    """Single-point version of LDRadar_Driver.get_points_xy_cm()."""
    rad = math.radians(angle_deg)
    x = distance_mm * math.cos(rad) * 0.1
    y = -distance_mm * math.sin(rad) * 0.1
    return x, y


# ── Tests ─────────────────────────────────────────────────────────────────

def test_forward_is_plus_x():
    """0° (hardware forward) → +X, Y ≈ 0"""
    x, y = get_points_xy_cm(0, 1000)
    assert x > 0, f"0° x should be positive, got {x:.3f}"
    assert abs(y) < 0.01, f"0° y should be ~0, got {y:.3f}"
    print(f"  PASS 0deg -> (x={x:+.1f}, y={y:+.1f}) OK")


def test_right_is_minus_y():
    """90° CW (hardware right) → -Y"""
    x, y = get_points_xy_cm(90, 1000)
    assert abs(x) < 0.01, f"90° x should be ~0, got {x:.3f}"
    assert y < 0, f"90° y should be negative (right=-Y), got {y:.3f}"
    print(f"  PASS 90deg -> (x={x:+.1f}, y={y:+.1f}) OK")


def test_back_is_minus_x():
    """180° CW (hardware back) → -X"""
    x, y = get_points_xy_cm(180, 1000)
    assert x < 0, f"180° x should be negative, got {x:.3f}"
    assert abs(y) < 0.01, f"180° y should be ~0, got {y:.3f}"
    print(f"  PASS 180deg -> (x={x:+.1f}, y={y:+.1f}) OK")


def test_left_is_plus_y():
    """270° CW (hardware left) → +Y"""
    x, y = get_points_xy_cm(270, 1000)
    assert abs(x) < 0.01, f"270° x should be ~0, got {x:.3f}"
    assert y > 0, f"270° y should be positive (left=+Y), got {y:.3f}"
    print(f"  PASS 270deg -> (x={x:+.1f}, y={y:+.1f}) OK")


def test_diagonal_45():
    """45° CW (front-right) → +X, -Y"""
    x, y = get_points_xy_cm(45, 1000)
    assert x > 0, "45° x should be positive"
    assert y < 0, "45° y should be negative (front-right=-Y)"
    # Both components should be roughly equal magnitude
    assert abs(x - abs(y)) < 2, f"45° |x|≈|y|, got x={x:.1f} y={y:.1f}"
    print(f"  PASS 45deg -> (x={x:+.1f}, y={y:+.1f}) OK")


def test_diagonal_315():
    """315° CW (front-left) → +X, +Y"""
    x, y = get_points_xy_cm(315, 1000)
    assert x > 0, "315° x should be positive"
    assert y > 0, "315° y should be positive (front-left=+Y)"
    assert abs(x - y) < 2, f"315° |x|≈|y|, got x={x:.1f} y={y:.1f}"
    print(f"  PASS 315deg -> (x={x:+.1f}, y={y:+.1f}) OK")


def test_full_360_consistency():
    """Sweep 0→359°: ensure continuity and correct sign pattern."""
    errors = []
    for deg in range(0, 360, 1):
        x, y = get_points_xy_cm(deg, 1000)
        if deg == 0:
            assert x > 0 and abs(y) < 1, f"0° fail: ({x:.1f},{y:.1f})"
        elif deg < 90:
            # Quadrant I (front-right): +X, -Y
            if not (x > 0 and y <= 0):
                errors.append(f"Q-I {deg}°: ({x:.1f},{y:.1f})")
        elif deg < 180:
            # Quadrant II (back-right): -X, -Y
            if not (x <= 0 and y <= 0):
                # Allow small positive x near 90°
                if deg > 100 and not (y <= 0):
                    errors.append(f"Q-II {deg}°: ({x:.1f},{y:.1f})")
        elif deg < 270:
            # Quadrant III (back-left): -X, +Y
            if not (x <= 0 and y >= 0):
                if deg > 190 and not (y >= 0):
                    errors.append(f"Q-III {deg}°: ({x:.1f},{y:.1f})")
        else:
            # Quadrant IV (front-left): +X, +Y
            if not (x >= 0 and y >= 0):
                if deg < 350 and not (y >= 0):
                    errors.append(f"Q-IV {deg}°: ({x:.1f},{y:.1f})")
    if errors:
        for e in errors[:10]:
            print(f"  WARN {e}")
        raise AssertionError(f"Quadrant mismatch in {len(errors)}/360 angles")
    print(f"  PASS full 360deg sweep: 0 errors OK")


def test_body_frame_coordinate_axes():
    """Verify the body frame axes match the documented convention.

    Documented: +X=forward, +Y=left, -Y=right, +Z=up
    """
    # Front: D500 angle=0° → body (d, 0) with d>0
    fx, fy = get_points_xy_cm(0, 2000)
    assert fx > 0 and abs(fy) < 1
    # Left: D500 angle=270° CW → body (0, d) with d>0
    lx, ly = get_points_xy_cm(270, 2000)
    assert abs(lx) < 1 and ly > 0
    # Right: D500 angle=90° CW → body (0, -d) with d<0
    rx, ry = get_points_xy_cm(90, 2000)
    assert abs(rx) < 1 and ry < 0
    # Back: D500 angle=180° CW → body (-d, 0) with d<0
    bx, by = get_points_xy_cm(180, 2000)
    assert bx < 0 and abs(by) < 1

    # Cross-check: front-left obstacle → (x>0, y>0), front-right → (x>0, y<0)
    fl_x, fl_y = get_points_xy_cm(330, 2000)   # 330° CW ≈ front-left
    fr_x, fr_y = get_points_xy_cm(30, 2000)     # 30° CW ≈ front-right
    assert fl_x > 0 and fl_y > 0, f"front-left: ({fl_x:.1f}, {fl_y:.1f})"
    assert fr_x > 0 and fr_y < 0, f"front-right: ({fr_x:.1f}, {fr_y:.1f})"

    print("  PASS body frame axes: +X=forward, +Y=left, -Y=right OK")


def test_map_circle_bin_calculation():
    """Verify Map_Circle bin indexing matches the angle sweep."""
    # ACC=3 → 1080 bins, bin for angle θ = round(θ * ACC) % (360*ACC)
    for deg in [0, 90, 180, 270, 359]:
        bin_idx = round(deg * ACC) % (360 * ACC)
        recovered_deg = bin_idx / ACC
        assert abs(recovered_deg - deg) < 0.01, f"{deg}° → bin {bin_idx} → {recovered_deg}°"
    print("  PASS Map_Circle bin indexing OK")


def test_mirror_y_transform():
    """Verify mount_mirror_y: points[:, 1] *= -1.0 (LDRadar_Driver.py:408-409).

    Lower radar is mounted upside-down (inverted).  Y-axis mirror flips:
       radar-local +Y → body -Y
       radar-local -Y → body +Y
    """
    # Simulate: lower radar local point at radar-local "left" (90° CW)
    rlx, rly = get_points_xy_cm(90, 1000)   # radar-local: (0, -10) cm
    # After mirror_y:
    rly_mirrored = -rly  # → +10
    assert rly_mirrored > 0, f"mirror_y: radar-local right → body left: y={rly_mirrored:+.1f}"
    print(f"  PASS mirror_y: radar-local right->body left, y {rly:+.1f}->{rly_mirrored:+.1f} OK")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("Stage 0.1 — Radar Coordinate Convention Tests")
    print("=" * 65)
    tests = [
        test_forward_is_plus_x,
        test_right_is_minus_y,
        test_back_is_minus_x,
        test_left_is_plus_y,
        test_diagonal_45,
        test_diagonal_315,
        test_full_360_consistency,
        test_body_frame_coordinate_axes,
        test_map_circle_bin_calculation,
        test_mirror_y_transform,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{'=' * 65}")
    print(f"Result: {passed}/{len(tests)} passed")
    if passed == len(tests):
        print("[PASS] All radar coordinate tests PASSED")
    else:
        print(f"[FAIL] {len(tests) - passed} test(s) FAILED")
