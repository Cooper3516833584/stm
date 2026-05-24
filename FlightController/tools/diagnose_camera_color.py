"""Camera white balance / color cast diagnostic.

Place a white sheet of paper in front of the camera, then run:

    PYTHONPATH=. python -u FlightController/tools/diagnose_camera_color.py --index 7
"""

import argparse

import numpy as np


def main() -> None:
    import cv2

    parser = argparse.ArgumentParser(description="Camera color cast diagnostic")
    parser.add_argument("--index", type=int, default=7, help="cv2 camera index")
    parser.add_argument("--frames", type=int, default=10, help="number of frames to average")
    parser.add_argument("--save", default=None, help="save captured frame to path (e.g. /media/sdcard/white_ref.jpg)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera index {args.index}")
        return

    # Discard first few frames (auto-exposure settling)
    for _ in range(5):
        cap.read()

    # Collect frames
    frames = []
    for i in range(args.frames):
        ok, frame = cap.read()
        if not ok:
            print(f"WARN: frame {i} read failed")
            continue
        frames.append(frame.astype(np.float64))
    cap.release()

    if not frames:
        print("ERROR: no frames captured")
        return

    # Average across frames to reduce noise
    avg = np.mean(frames, axis=0)
    h, w = avg.shape[:2]

    # Only sample the center 60% of the image (avoid vignetting at edges)
    cx, cy = w // 2, h // 2
    crop_w, crop_h = int(w * 0.6), int(h * 0.6)
    x1, y1 = cx - crop_w // 2, cy - crop_h // 2
    x2, y2 = x1 + crop_w, y1 + crop_h
    roi = avg[y1:y2, x1:x2, :]

    # Per-channel statistics
    b_mean = np.mean(roi[:, :, 0])
    g_mean = np.mean(roi[:, :, 1])
    r_mean = np.mean(roi[:, :, 2])
    b_std = np.std(roi[:, :, 0])
    g_std = np.std(roi[:, :, 1])
    r_std = np.std(roi[:, :, 2])

    # Overall brightness
    gray_mean = (b_mean + g_mean + r_mean) / 3.0

    # Color balance ratios (normalized to green = 1.0)
    rg_ratio = r_mean / g_mean if g_mean > 0 else 0
    bg_ratio = b_mean / g_mean if g_mean > 0 else 0

    print(f"=== 前视摄像头 /dev/video{args.index} 色彩分析 ===")
    print(f"分析区域: {crop_w}×{crop_h} (画面中心 60%)")
    print(f"采样帧数: {len(frames)}")
    print()
    print(f"           B        G        R")
    print(f"  mean:  {b_mean:7.1f}  {g_mean:7.1f}  {r_mean:7.1f}")
    print(f"  std:   {b_std:7.1f}  {g_std:7.1f}  {r_std:7.1f}")
    print(f"  ratio:  {bg_ratio:7.3f}    1.000    {rg_ratio:7.3f}  (B/G | G/G | R/G)")
    print(f"  整体平均亮度: {gray_mean:.1f} / 255")

    # Diagnose color cast
    tol = 0.05
    issues = []
    if rg_ratio > 1.0 + tol:
        issues.append(f"偏红 (R/G={rg_ratio:.3f})")
    elif rg_ratio < 1.0 - tol:
        issues.append(f"偏青 (R/G={rg_ratio:.3f})")

    if bg_ratio > 1.0 + tol:
        issues.append(f"偏蓝 (B/G={bg_ratio:.3f})")
    elif bg_ratio < 1.0 - tol:
        issues.append(f"偏黄 (B/G={bg_ratio:.3f})")

    print()
    if issues:
        print(f"⚠ 检测到偏色: {'; '.join(issues)}")
    else:
        print("✓ 白平衡无明显偏色 (R/G 和 B/G 均在 {:.2f}-{:.2f} 范围内)".format(1.0 - tol, 1.0 + tol))

    # Correction suggestion
    if rg_ratio != 1.0 or bg_ratio != 1.0:
        kr = 1.0 / max(rg_ratio, 0.01)
        kb = 1.0 / max(bg_ratio, 0.01)
        print(f"  建议白平衡修正系数: R×{kr:.2f}  B×{kb:.2f}  G×1.00")
        print(f"  即: cv2.cvtColor 后做 img[:,:,0]*{kb:.2f}  img[:,:,2]*{kr:.2f}")
        print()
        print("  或者用 cv2.xphoto.createSimpleWB() 自动白平衡:")
        print("    wb = cv2.xphoto.createSimpleWB()")
        print("    corrected = wb.balanceWhite(frame)")

    # Save if requested
    if args.save:
        cv2.imwrite(args.save, avg.astype(np.uint8))
        print(f"  平均帧已保存至: {args.save}")


if __name__ == "__main__":
    main()
