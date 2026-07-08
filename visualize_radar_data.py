"""PC-side radar point cloud visualizer for data recorded by record_data.py.

Reads radar_bins/*.npz (raw 1080-bin distances), radar_points/*.npz (radar-local XY),
and radar.jsonl metadata. Reconstructs the body-frame point cloud exactly as the drone
sees it, then renders a top-down PNG sequence and optionally an MP4 video.

Coordinate conventions (matching the drone's body frame):
    +X = forward (up on screen)
    +Y = left   (left on screen)

Usage:
    python visualize_radar_data.py D:/drone/radar_data/20260611_195147_record
    python visualize_radar_data.py D:/drone/radar_data/20260611_195147_record --video
    python visualize_radar_data.py D:/drone/radar_data/20260611_195147_record --every-n 5 --video

Visual elements rendered per frame
----------------------------------

+-----------------------------+------------------------------------------+------------------------------+
| Element                     | Appearance                               | Meaning                      |
+=============================+==========================================+==============================+
| Upper radar points          | Gold dots (core r=2px, glow r=25px)   | Upper D500 1080-bin returns  |
|                             | color = (30, 235, 255) BGR               | mapped to body frame,        |
|                             |                                          | self-reflections masked      |
+-----------------------------+------------------------------------------+------------------------------+
| Lower radar points          | Electric-blue dots (core r=1,glow r=15)  | Lower D500 1080-bin returns  |
|                             | color = (255, 150, 40) BGR               | Y-mirrored -> body frame,    |
|                             |                                          | offset (0.96, 0.15) cm       |
+-----------------------------+------------------------------------------+------------------------------+
| Point glow / halo           | Gaussian-blurred large circles behind    | Close points merge into      |
|                             | each point, blended additively           | continuous glow lines        |
+-----------------------------+------------------------------------------+------------------------------+
| Distance rings              | Solid concentric circles, labelled       | 50 cm minor (thin),          |
|                             | every 50 cm from centre                  | 100 cm major (thick + label) |
|                             | label font scale 2.08, thickness 4       |                              |
+-----------------------------+------------------------------------------+------------------------------+
| Drone body                  | Rounded-square (square 33.5cm side       | |x| < 25 cm, |y| < 25 cm      |
|                             | + circles r=8.35cm at corners),           | self-reflection exclusion    |
|                             | bronze fill + gold outline               | zone; nose = forward         |
+-----------------------------+------------------------------------------+------------------------------+
| Forward arrow               | White filled thin arrow (~8px shaft)     | +X (forward) direction       |
|                             | color = (240, 240, 240) BGR              | "FWD" label alongside        |
+-----------------------------+------------------------------------------+------------------------------+
| Crosshair                   | Interrupted lines through origin         | (0, 0) of body frame         |
|                             | 4 px gap at centre                       |                              |
+-----------------------------+------------------------------------------+------------------------------+
| Top HUD bar                 | Semi-transparent dark panel              | LOOP NNNNN  T+ elapsed       |
|                             |                                          | total pts  upper/lower counts|
+-----------------------------+------------------------------------------+------------------------------+
| Bottom legend bar           | Semi-transparent dark panel              | upper/lower colour swatch    |
|                             |                                          | range Ncm | grid 50cm         |
+-----------------------------+------------------------------------------+------------------------------+

Data pipeline per frame
-----------------------

1. Load pts.npz -> radar-local XY arrays (upper, lower) in cm.
2. Apply Y-mirror to lower radar (mount_mirror_y=True).
3. Translate to body frame: upper += (0, 0); lower += (0.96, 0.15) cm.
4. Filter: distance > 300 cm -> discard; |x| < 25 & |y| < 25 -> discard.
5. Render into 2240x2240 px top-down view (+-280 cm range, 4 px/cm).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# ── constants matching the drone hardware ───────────────────────────
ACC = 3
TOTAL_BINS = 360 * ACC  # 1080

UPPER_MIRROR = False
UPPER_TX_CM  = 0.0
UPPER_TY_CM  = 0.0

LOWER_MIRROR = True
LOWER_TX_CM  = 0.96
LOWER_TY_CM  = 0.15

BODY_X_HALF_CM = 25.0
BODY_Y_HALF_CM = 25.0
MAX_DISTANCE_CM = 300.0

# Visualisation
SCALE_PX_PER_CM = 4
VIEW_RANGE_CM   = 280
CANVAS          = int(VIEW_RANGE_CM * 2 * SCALE_PX_PER_CM)

# ── palette (BGR) ───────────────────────────────────────────────────
BG_DEEP     = (18, 12, 10)
RING_MAJOR  = (50, 44, 38)
RING_MINOR  = (34, 30, 26)
BODY_FILL   = (55, 50, 44)
BODY_EDGE   = (115, 105, 90)
AXIS_COLOR  = (75, 68, 60)
ARROW_COLOR = (240, 240, 240)  # white thin arrow
UPPER_COLOR = (30, 235, 255)   # saturated gold
LOWER_COLOR = (255, 150, 40)   # saturated electric blue
TEXT_MAIN   = (210, 200, 190)
TEXT_DIM    = (130, 125, 115)

# ── body geometry ───────────────────────────────────────────────────
# Rounded square: square side = 50*0.67=33.5cm, corner circle r=50*0.167=8.35cm
BODY_SQ_HALF_CM = 50.0 * 0.335   # 16.75 cm
BODY_CR_CM      = 50.0 * 0.167   # 8.35 cm


# ── coordinate helpers ───────────────────────────────────────────────

def _radar_to_body(pts_xy: np.ndarray, mirror_y: bool,
                   tx_cm: float, ty_cm: float) -> np.ndarray:
    if pts_xy.size == 0:
        return pts_xy
    p = pts_xy.copy()
    if mirror_y:
        p[:, 1] *= -1.0
    p[:, 0] += tx_cm
    p[:, 1] += ty_cm
    return p


def _filter_body_frame(pts: np.ndarray) -> np.ndarray:
    if pts.size == 0:
        return pts
    d = np.linalg.norm(pts, axis=1)
    keep = d <= MAX_DISTANCE_CM
    body = (np.abs(pts[:, 0]) < BODY_X_HALF_CM) & (np.abs(pts[:, 1]) < BODY_Y_HALF_CM)
    return pts[keep & ~body]


def _pixel(x_cm: float, y_cm: float, cx: int, cy: int):
    return int(cx - y_cm * SCALE_PX_PER_CM), int(cy - x_cm * SCALE_PX_PER_CM)


# ── rendering helpers ────────────────────────────────────────────────

def _rings(img, cx, cy):
    for r_cm in range(50, VIEW_RANGE_CM + 1, 50):
        r_px = int(r_cm * SCALE_PX_PER_CM)
        major = (r_cm % 100 == 0)
        col = RING_MAJOR if major else RING_MINOR
        cv2.circle(img, (cx, cy), r_px, col, 2 if major else 1, cv2.LINE_AA)
        if major:
            cv2.putText(img, str(r_cm), (cx + 10, cy - r_px + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.08, TEXT_DIM, 4, cv2.LINE_AA)


def _body(img, cx, cy):
    """Rounded square: square of side 33.5 cm + circles r=8.35 cm at corners.
    Only the outer boundary of the union is drawn/filled."""
    sq_half = BODY_SQ_HALF_CM
    cr = BODY_CR_CM
    cr_px = int(cr * SCALE_PX_PER_CM)

    mask = np.zeros((CANVAS, CANVAS), dtype=np.uint8)

    sq_corners = np.array([
        _pixel( sq_half, -sq_half, cx, cy),
        _pixel( sq_half,  sq_half, cx, cy),
        _pixel(-sq_half,  sq_half, cx, cy),
        _pixel(-sq_half, -sq_half, cx, cy),
    ], dtype=np.int32)
    cv2.fillPoly(mask, [sq_corners], 255)

    for sx, sy in [(sq_half, sq_half), (sq_half, -sq_half),
                   (-sq_half, sq_half), (-sq_half, -sq_half)]:
        cpx, cpy = _pixel(sx, sy, cx, cy)
        cv2.circle(mask, (cpx, cpy), cr_px, 255, -1, cv2.LINE_AA)

    # fill
    body_fill = np.array(BODY_FILL, dtype=np.uint8)
    img[mask > 0] = body_fill

    # outline
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, BODY_EDGE, 2, cv2.LINE_AA)

    # nose dot at forward-most point (top of front circle)
    nose = _pixel(sq_half + cr, 0, cx, cy)
    cv2.circle(img, nose, 5, ARROW_COLOR, -1, cv2.LINE_AA)


def _arrow(img, cx, cy):
    """White filled thin arrow, ~8 px shaft width, FWD label alongside."""
    pts = np.array([
        _pixel(40, 0, cx, cy),       # head tip
        _pixel(20, 3, cx, cy),       # right wing tip
        _pixel(20, 1, cx, cy),       # right shaft
        _pixel(-10, 1, cx, cy),      # shaft rear right
        _pixel(-10, -1, cx, cy),     # shaft rear left
        _pixel(20, -1, cx, cy),      # left shaft
        _pixel(20, -3, cx, cy),      # left wing tip
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], ARROW_COLOR)
    cv2.polylines(img, [pts], True, ARROW_COLOR, 1, cv2.LINE_AA)
    # FWD label alongside the arrow head
    tx, ty = _pixel(48, -6, cx, cy)
    cv2.putText(img, "FWD", (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, ARROW_COLOR, 2, cv2.LINE_AA)


def _scatter_with_glow(img, pts, cx, cy, color, core_r, glow_r):
    """Draw points with soft glow using 1/4-scale downsampled blur for speed.

    The glow layer is rendered and blurred on a canvas 4× smaller, then
    upscaled and blended.  This makes the 5× larger radii cost roughly the
    same as the original small radii."""
    if len(pts) == 0:
        return

    DOWN = 4
    small_sz = CANVAS // DOWN
    small_gr = max(1, glow_r // DOWN)

    glow_small = np.zeros((small_sz, small_sz, 3), dtype=np.uint8)
    for pt in pts:
        px, py = _pixel(pt[0], pt[1], cx, cy)
        sx, sy = px // DOWN, py // DOWN
        if 0 <= sx < small_sz and 0 <= sy < small_sz:
            cv2.circle(glow_small, (sx, sy), small_gr, color, -1, cv2.LINE_AA)

    ks = max(3, small_gr) | 1
    glow_small = cv2.GaussianBlur(glow_small, (ks, ks), 0)
    glow_full = cv2.resize(glow_small, (CANVAS, CANVAS), interpolation=cv2.INTER_LINEAR)
    cv2.addWeighted(glow_full, 0.35, img, 1.0, 0, img)

    # core dots on top (full-res)
    for pt in pts:
        px, py = _pixel(pt[0], pt[1], cx, cy)
        if 0 <= px < CANVAS and 0 <= py < CANVAS:
            cv2.circle(img, (px, py), core_r, color, -1, cv2.LINE_AA)


def _panel(img, x, y, w, h, alpha=0.42):
    over = img.copy()
    cv2.rectangle(over, (x, y), (x + w, y + h), (20, 16, 12), -1)
    cv2.addWeighted(over, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), (65, 60, 52), 1, cv2.LINE_AA)


# ── frame renderer ───────────────────────────────────────────────────

def render_frame(pts_upper: np.ndarray, pts_lower: np.ndarray,
                 loop: int, time_s: float, out_path: str) -> None:
    img = np.full((CANVAS, CANVAS, 3), BG_DEEP, dtype=np.uint8)
    cx = cy = CANVAS // 2
    m = 8

    _rings(img, cx, cy)
    _body(img, cx, cy)
    _scatter_with_glow(img, pts_lower, cx, cy, LOWER_COLOR, core_r=1, glow_r=15)
    _scatter_with_glow(img, pts_upper, cx, cy, UPPER_COLOR, core_r=2, glow_r=25)
    _arrow(img, cx, cy)

    # crosshair
    ch = 14
    cv2.line(img, (cx - ch, cy), (cx - 4, cy), AXIS_COLOR, 1, cv2.LINE_AA)
    cv2.line(img, (cx + 4, cy), (cx + ch, cy), AXIS_COLOR, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - ch), (cx, cy - 4), AXIS_COLOR, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy + 4), (cx, cy + ch), AXIS_COLOR, 1, cv2.LINE_AA)

    # ── top HUD bar ──
    _panel(img, m, m, CANVAS - 2 * m, 36)
    total = len(pts_upper) + len(pts_lower)
    cv2.putText(img, f"LOOP {loop:05d}",
                (m + 12, m + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, TEXT_MAIN, 2, cv2.LINE_AA)
    cv2.putText(img, f"T+{time_s:.1f}s",
                (m + 210, m + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(img, f"{total} pts",
                (CANVAS - 320, m + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_MAIN, 1, cv2.LINE_AA)
    cv2.circle(img, (CANVAS - 188, m + 18), 5, UPPER_COLOR, -1)
    cv2.putText(img, str(len(pts_upper)), (CANVAS - 176, m + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.circle(img, (CANVAS - 120, m + 18), 5, LOWER_COLOR, -1)
    cv2.putText(img, str(len(pts_lower)), (CANVAS - 108, m + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_DIM, 1, cv2.LINE_AA)

    # ── bottom legend ──
    by = CANVAS - m - 28
    _panel(img, m, by - 6, CANVAS - 2 * m, 34)
    cv2.circle(img, (m + 14, by + 10), 5, UPPER_COLOR, -1)
    cv2.putText(img, "upper", (m + 26, by + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, TEXT_MAIN, 1, cv2.LINE_AA)
    cv2.circle(img, (m + 100, by + 10), 5, LOWER_COLOR, -1)
    cv2.putText(img, "lower", (m + 112, by + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, TEXT_MAIN, 1, cv2.LINE_AA)
    cv2.putText(img, f"range {VIEW_RANGE_CM}cm  |  grid 50cm",
                (CANVAS - 360, by + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_DIM, 1, cv2.LINE_AA)

    cv2.imwrite(out_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])


# ── polar panel ──────────────────────────────────────────────────────

def render_polar_panel(bins_dict: dict[str, np.ndarray], out_path: str) -> None:
    SIZE = 900
    img = np.full((SIZE, SIZE, 3), BG_DEEP, dtype=np.uint8)
    cx = cy = SIZE // 2
    mr = SIZE // 2 - 30

    for r_cm in (100, 200, 300):
        r_px = int(r_cm / MAX_DISTANCE_CM * mr)
        cv2.circle(img, (cx, cy), r_px, RING_MINOR, 1, cv2.LINE_AA)
        cv2.putText(img, str(r_cm), (cx + 5, cy - r_px - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_DIM, 1, cv2.LINE_AA)

    for deg in range(0, 360, 30):
        rad = np.deg2rad(deg)
        ex = int(cx + mr * np.sin(rad))
        ey = int(cy - mr * np.cos(rad))
        cv2.line(img, (cx, cy), (ex, ey), RING_MINOR, 1, cv2.LINE_AA)
        label = f"{deg}" if deg <= 180 else f"{deg - 360}"
        cv2.putText(img, label, (ex - 14, ey - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_DIM, 1, cv2.LINE_AA)

    pal = dict(upper=UPPER_COLOR, lower=LOWER_COLOR)
    for name, bins in bins_dict.items():
        if bins is None:
            continue
        ok = bins != -1
        idx = np.where(ok)[0]
        if idx.size == 0:
            continue
        degs = idx / ACC  # type: ignore[operator]
        dists_mm = bins[ok]
        for ang, mm in zip(degs, dists_mm):
            if mm <= 0 or mm > MAX_DISTANCE_CM * 10:
                continue
            rad = np.deg2rad(ang)
            r_px = mm / 10 / MAX_DISTANCE_CM * mr
            px = int(cx + r_px * np.sin(rad))
            py = int(cy - r_px * np.cos(rad))
            if 0 <= px < SIZE and 0 <= py < SIZE:
                cv2.circle(img, (px, py), 1, pal.get(name, (150, 150, 150)), -1)

    cv2.putText(img, "FWD", (cx - 16, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, ARROW_COLOR, 2, cv2.LINE_AA)
    cv2.line(img, (cx, cy - mr - 10), (cx, cy), ARROW_COLOR, 1, cv2.LINE_AA)

    cv2.imwrite(out_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])


# ── video compile ────────────────────────────────────────────────────

def _compile_video(frames_dir: Path, fps: int) -> None:
    frames = sorted(frames_dir.glob("pointcloud_*.png"))
    if not frames:
        print("[WARN] no pointcloud frames to compile")
        return
    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    vpath = str(frames_dir / "pointcloud.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(vpath, fourcc, fps, (w, h))
    for f in frames:
        writer.write(cv2.imread(str(f)))
    writer.release()
    print(f"[INFO] video -> {vpath}")


# ── main ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize recorded radar data as top-down point cloud images")
    parser.add_argument("record_dir",
                        help="Path to recorded session directory")
    parser.add_argument("-o", "--output-dir", default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--every-n", type=int, default=1)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--polar", action="store_true")
    parser.add_argument("--no-pointcloud", action="store_true")
    args = parser.parse_args()

    rec = Path(args.record_dir)
    jsonl_path = rec / "radar.jsonl"
    bins_dir = rec / "radar_bins"
    pts_dir  = rec / "radar_points"

    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else rec / "visualization"
    out_dir.mkdir(parents=True, exist_ok=True)
    polar_dir = out_dir / "polar"
    if args.polar:
        polar_dir.mkdir(exist_ok=True)

    entries: list[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for li, line in enumerate(fh):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[WARN] skipping truncated line {li}")

    print(f"[INFO] {len(entries)} snapshots in {jsonl_path}")
    if args.max_frames > 0:
        entries = entries[:args.max_frames]

    rendered_pc = rendered_polar = skipped = corrupted = 0

    for idx, entry in enumerate(entries):
        if idx % args.every_n != 0:
            continue

        loop = entry["loop"]
        time_s = entry.get("time_perf_s", 0.0)

        pts_file = Path(entry.get("points_file", ""))
        pts_path = pts_dir / pts_file.name if pts_file.name else None
        bins_file = Path(entry.get("bins_file", ""))
        bins_path = bins_dir / bins_file.name if bins_file.name else None

        if pts_path is None or not pts_path.exists():
            skipped += 1
            continue

        if not args.no_pointcloud:
            try:
                data = np.load(str(pts_path))
                upper_xy = data["upper"]
                lower_xy = data["lower"]
                data.close()
            except (OSError, EOFError, ValueError) as e:
                corrupted += 1
                if corrupted <= 5:
                    print(f"[WARN] corrupt {pts_path.name}: {e}")
                continue

            upper_body = _radar_to_body(upper_xy, UPPER_MIRROR, UPPER_TX_CM, UPPER_TY_CM)
            lower_body = _radar_to_body(lower_xy, LOWER_MIRROR, LOWER_TX_CM, LOWER_TY_CM)
            upper_f = _filter_body_frame(upper_body)
            lower_f = _filter_body_frame(lower_body)

            pc_path = out_dir / f"pointcloud_{loop:07d}.png"
            render_frame(upper_f, lower_f, loop, time_s, str(pc_path))
            rendered_pc += 1

        if args.polar and bins_path is not None and bins_path.exists():
            try:
                bdata = np.load(str(bins_path))
            except (OSError, EOFError, ValueError):
                continue
            bins_dict = {k: bdata[k] for k in bdata.files}
            bdata.close()
            pol_path = polar_dir / f"polar_{loop:07d}.png"
            render_polar_panel(bins_dict, str(pol_path))
            rendered_polar += 1

        if (rendered_pc + rendered_polar) % 200 == 0 and (rendered_pc + rendered_polar) > 0:
            print(f"  ... {rendered_pc} pc + {rendered_polar} polar  "
                  f"({idx + 1}/{len(entries)} scanned)")

    if skipped:
        print(f"[WARN] {skipped} entries skipped (missing pts.npz)")
    if corrupted:
        print(f"[WARN] {corrupted} entries skipped (corrupt .npz)")

    print(f"\n[DONE] pc={rendered_pc}  polar={rendered_polar}  -> {out_dir}")

    if args.video:
        _compile_video(out_dir, args.video_fps)
        if args.polar:
            _compile_video(polar_dir, args.video_fps)


if __name__ == "__main__":
    main()
