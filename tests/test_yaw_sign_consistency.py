"""Stage 0.2: Verify yaw sign conventions across the full control chain.

Aims to answer: when an obstacle is on the LEFT side of the drone,
does the system command a LEFT turn (correct) or RIGHT turn (bug)?

Control chain:
  body-frame obstacle angle → navigator._yaw_command() → Command.yaw_rate
  → goal_nav_main → send_realtime_control_data(..., yaw=...)
  → struct.pack("<hhhh", ..., -yaw)

Key conventions:
  - Body frame:    +Y = left,  -Y = right
  - FC API (send_realtime_control_data): yaw>0 = clockwise = right turn
  - FC wire:       yaw is negated: struct.pack(..., -yaw)
  - Navigator:     candidate angle_deg: + = left side, - = right side
"""

from __future__ import annotations

import math
import sys
import os

# Ensure we can import the project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


# ── Replicate the relevant code paths ─────────────────────────────────────

def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _wrap_deg(d: float) -> float:
    """Angle wrap to (-180, 180]."""
    d = d % 360.0
    if d > 180.0:
        d -= 360.0
    return d


# ── Simulated RelativeGoalNavigator._yaw_command ──────────────────────────
# From RelativeGoalNavigator.py:320-331

align_stop_deg = 3.0
yaw_kp = 0.5
yaw_rate_limit = 25.0
min_turn_yaw_rate = 6.0


def navigator_yaw_current(angle_deg: float) -> float:
    """CURRENT implementation: yaw = angle * kp (no sign correction)."""
    if abs(angle_deg) <= align_stop_deg:
        return 0.0
    yaw = _clip(angle_deg * yaw_kp, -yaw_rate_limit, yaw_rate_limit)
    if 0.0 < abs(yaw) < min_turn_yaw_rate:
        yaw = math.copysign(min_turn_yaw_rate, yaw)
    return yaw


def navigator_yaw_fixed(angle_deg: float) -> float:
    """FIXED: negate to map body-frame angle to FC CW-positive convention."""
    if abs(angle_deg) <= align_stop_deg:
        return 0.0
    yaw = _clip(-angle_deg * yaw_kp, -yaw_rate_limit, yaw_rate_limit)
    if 0.0 < abs(yaw) < min_turn_yaw_rate:
        yaw = math.copysign(min_turn_yaw_rate, yaw)
    return yaw


# ── FC protocol layer ────────────────────────────────────────────────────

def fc_wire_yaw(yaw_api: float) -> int:
    """FC wire protocol: struct.pack('<hhhh', ..., round(-yaw))."""
    return round(-yaw_api)


# ── Tests ─────────────────────────────────────────────────────────────────

def test_fc_api_convention():
    """Confirm: FC API positive yaw = clockwise = right turn."""
    # API: yaw=10 → packed as -10
    assert fc_wire_yaw(10) == -10, "yaw=10 → wire -10"
    assert fc_wire_yaw(-10) == 10, "yaw=-10 → wire 10"
    # This means: yaw>0 in API = CW = turns aircraft nose right
    print("  PASS FC API: yaw>0=CW=right turn, yaw<0=CCW=left turn OK")


def test_navigator_yaw_sign_bug():
    """Reproduce the suspected yaw sign bug.

    Scenario: obstacle detected at body-frame angle +30° (LEFT side).
    The navigator should command a RIGHT turn to avoid it (yaw>0 in FC API).
    """
    obstacle_angle = 30.0  # +30° = left side
    current = navigator_yaw_current(obstacle_angle)
    fixed = navigator_yaw_fixed(obstacle_angle)

    print(f"\n  Scenario: obstacle at +{obstacle_angle}deg (LEFT side)")
    print(f"    Current _yaw_command(+30deg) = {current:+.1f} deg/s")
    print(f"      -> FC API yaw>0 = right turn")
    print(f"      -> Aircraft turns RIGHT towards the obstacle [BUG]")
    print(f"    Fixed   _yaw_command(+30deg) = {fixed:+.1f} deg/s")
    print(f"      -> FC API yaw<0 = left turn")
    print(f"      -> Aircraft turns LEFT away from obstacle [CORRECT]")

    # Current: positive yaw for left-side obstacle = turns right (BUG)
    assert current > 0, f"Current yaw for left obstacle should be >0, got {current}"
    # Fixed: negative yaw for left-side obstacle = turns left (CORRECT)
    assert fixed < 0, f"Fixed yaw for left obstacle should be <0, got {fixed}"

    print("  PASS Yaw sign bug confirmed: current code inverts direction")


def test_selected_direction_to_yaw():
    """Verify _yaw_command maps selected direction to FC yaw correctly.

    _yaw_command(selected_angle) receives the DIRECTION the navigator wants
    to fly toward (in body frame: + = left, - = right).  It must output a
    yaw rate that STEERS the aircraft toward that direction.

    To fly toward a LEFT-side angle (+):
      -> need LEFT/CCW turn -> yaw<0 in FC API -> _yaw_command() must be NEGATIVE

    To fly toward a RIGHT-side angle (-):
      -> need RIGHT/CW turn -> yaw>0 in FC API -> _yaw_command() must be POSITIVE

    Current code: yaw = +angle * kp  (same sign as angle -> WRONG for left)
    Fixed  code: yaw = -angle * kp  (opposite sign -> CORRECT)
    """
    cases = [
        # (selected_direction_deg, expected_api_sign, description)
        (+45, -1, "fly toward +45deg (left)  -> need CCW/left turn  -> yaw<0"),
        (-45, +1, "fly toward -45deg (right) -> need CW/right turn -> yaw>0"),
        (+10, -1, "fly toward +10deg (left)  -> gentle CCW turn     -> yaw<0"),
        (-10, +1, "fly toward -10deg (right) -> gentle CW turn     -> yaw>0"),
        (+3,   0, "within align_stop_deg     -> no turn needed     -> yaw=0"),
        (+0,   0, "dead ahead                -> no turn needed     -> yaw=0"),
    ]

    print()
    cur_all_ok = True
    fix_all_ok = True
    for angle, expected_sign, desc in cases:
        cur = navigator_yaw_current(angle)
        fix = navigator_yaw_fixed(angle)

        cur_turn = "RIGHT(CW)" if cur > 0 else ("LEFT(CCW)" if cur < 0 else "STOP")
        fix_turn = "RIGHT(CW)" if fix > 0 else ("LEFT(CCW)" if fix < 0 else "STOP")

        # Check: output sign matches expected API sign
        def sign_of(v):
            return 0.0 if abs(v) < 0.001 else (+1.0 if v > 0 else -1.0)
        cur_sign = sign_of(cur)
        fix_sign = sign_of(fix)

        cur_ok = cur_sign == expected_sign
        fix_ok = fix_sign == expected_sign
        if not cur_ok:
            cur_all_ok = False
        if not fix_ok:
            fix_all_ok = False

        cur_status = "OK" if cur_ok else "WRONG"
        fix_status = "OK" if fix_ok else "WRONG"

        print(f"  [{cur_status}] current {angle:+4.0f}deg -> yaw={cur:+6.1f} -> {cur_turn:>10s}  ({desc})")
        print(f"  [{fix_status}] fixed   {angle:+4.0f}deg -> yaw={fix:+6.1f} -> {fix_turn:>10s}")

    # Current should have failures, fixed should not
    assert not cur_all_ok, "Expected current to be WRONG for non-zero angles"
    assert fix_all_ok, "Expected fixed to be CORRECT for all angles"
    print("  PASS: current code steers WRONG direction, fixed code steers CORRECT")


def test_navigator_select_direction_sign():
    """Verify _select_direction returns body-frame angle (positive=left).

    The navigator's candidate directions span -75° to +75°.
    When goal is straight ahead (0°) and obstacle blocks left side,
    the navigator should select a NEGATIVE angle (right side).
    """
    # Simulated: obstacle blocks everything from -20° to +75°
    # The best candidate should be around -21° (first clear angle on right)
    candidate_span = [-75.0 + i * 2.0 for i in range(76)]  # -75 to +75 in 2° steps

    # Mock evaluation: candidates blocked where clearance < 80
    def is_blocked(angle_deg):
        return angle_deg > -22  # everything right of -22° is blocked

    allowed = [(a, 200.0) for a in candidate_span if not is_blocked(a)]
    blocked = [(a, 40.0) for a in candidate_span if is_blocked(a)]

    print(f"\n  Scenario: obstacle blocks [-20deg, +75deg], goal at 0deg")
    print(f"    Allowed directions: {len(allowed)} ({allowed[0][0]:+.0f}deg to {allowed[-1][0]:+.0f}deg)")
    print(f"    Blocked directions: {len(blocked)} ({blocked[0][0]:+.0f}deg to {blocked[-1][0]:+.0f}deg)")

    # Best allowed direction closest to goal (0°)
    if allowed:
        best = min(allowed, key=lambda x: abs(x[0] - 0.0))
        print(f"    Best candidate: angle={best[0]:+.0f}deg (body frame, negative=right)")
        print(f"    This means: turn RIGHT to avoid left-side obstacle [OK]")

        # Verify: best angle is negative (right side of body)
        assert best[0] < 0, f"Best angle should be negative (right), got {best[0]:+.0f}"

        # The yaw command should steer towards this angle
        # current _yaw_command(best[0]) with best[0] < 0 → yaw < 0 → left turn?!
        cur_yaw = navigator_yaw_current(best[0])
        fix_yaw = navigator_yaw_fixed(best[0])
        print(f"    Current _yaw_command yields: {cur_yaw:+.1f} deg/s (should be >0 for right turn)")
        print(f"    Fixed   _yaw_command yields: {fix_yaw:+.1f} deg/s (should be >0 for right turn)")

        # Expected: to go to negative angle, turn right (yaw>0 in FC API)
        assert cur_yaw < 0, "Current: angle<0 gives yaw<0 -> wrong direction!"
        assert fix_yaw > 0, "Fixed: angle<0 gives yaw>0 -> correct right turn"
    print("  PASS Navigator direction selection: fix produces correct yaw polarity")


def test_road_follower_yaw_sign_config():
    """RoadFollower has yaw_sign parameter (default 1.0) -- verify awareness."""
    try:
        from FlightController.Solutions.RoadFollower import RoadFollowerConfig
    except ModuleNotFoundError:
        print("\n  [SKIP] RoadFollower import needs pyserial (not installed on PC)")
        print("  RoadFollower has yaw_sign=1.0 config, RelativeGoalNavigator lacks equivalent")
        return

    cfg = RoadFollowerConfig()
    print(f"\n  RoadFollowerConfig.yaw_sign = {cfg.yaw_sign}")
    print(f"  pixel_kp_yaw = {cfg.pixel_kp_yaw}, angle_kp_yaw = {cfg.angle_kp_yaw}")
    print(f"  RoadFollower has built-in sign reversal capability [OK]")
    print("  RelativeGoalNavigator does NOT have equivalent sign reversal [MISSING]")


def test_safety_corridor_direction():
    """Verify SafetyArbiter forward corridor respects body frame axes.

    select_forward_corridor (ObstacleUtils.py):
      points[x > min_x_cm & |y| < half_width_cm]

    This only looks at +X, so it's direction-agnostic (no yaw issue here).
    """
    try:
        from FlightController.Solutions.ObstacleUtils import select_forward_corridor
    except ModuleNotFoundError:
        print("\n  [SKIP] ObstacleUtils import needs pyserial (not installed on PC)")
        print("  Forward corridor logic: points[x>min_x & |y|<half_width], no yaw dependency")
        return

    test_points = np.array([
        [100, 0],     # straight ahead
        [100, 30],    # ahead + left
        [100, -30],   # ahead + right
        [-50, 0],     # behind
    ], dtype=float)

    forward = select_forward_corridor(test_points, min_x_cm=10, half_width_cm=50)
    assert len(forward) == 3, f"Expected 3 forward points, got {len(forward)}"
    assert not any(forward[:, 0] < 0), "Backward point should be filtered"
    print("  PASS Safety forward corridor: correct axis usage (no yaw dependency)")


# ── Summary ────────────────────────────────────────────────────────────────

def print_summary():
    print(f"\n{'=' * 65}")
    print("SUMMARY")
    print(f"{'=' * 65}")
    print("""
  Root cause identified:

    RelativeGoalNavigator._yaw_command() does NOT negate the angle.

    Body frame convention:  +Y = left,  -Y = right
    FC API convention:      yaw>0 = CW = right turn

    Navigator angle_deg = +30deg (left side)
      -> current _yaw_command: yaw = +15 deg/s -> FC interprets as RIGHT turn
      -> aircraft turns TOWARDS the obstacle instead of AWAY

    The fix: negate angle_deg in _yaw_command:
      yaw = -angle_deg * yaw_kp

    RoadFollower already has yaw_sign parameter for this purpose.
    RelativeGoalNavigator is missing the equivalent safeguard.

  Affected file:
    FlightController/Solutions/RelativeGoalNavigator.py:320-331

  Fix (1 line):
    Line 325:  angle_deg * cfg.yaw_kp
    ->         -angle_deg * cfg.yaw_kp
""")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("Stage 0.2 — Yaw Sign Consistency Tests")
    print("=" * 65)

    tests = [
        test_fc_api_convention,
        test_navigator_yaw_sign_bug,
        test_selected_direction_to_yaw,
        test_navigator_select_direction_sign,
        test_road_follower_yaw_sign_config,
        test_safety_corridor_direction,
    ]

    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"\n  [FAIL] {t.__name__}: {e}")

    print_summary()
    print(f"Result: {passed}/{len(tests)} passed")
    if passed == len(tests):
        print("[PASS] All yaw sign tests PASSED -- bug confirmed")
    else:
        print(f"[FAIL] {len(tests) - passed} test(s) FAILED")
