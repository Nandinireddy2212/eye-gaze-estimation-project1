"""
heatmap.py — ProctorEye
========================
Exact notebook method:
  1. Store raw camera eye coords (cam_x, cam_y)
  2. Scale to screen using calibration iris bounds
  3. KDE plot with rocket_r colormap, bw_adjust=0.8
  4. Blend on real screenshot background
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

sns.set(style="white")
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


def scale_cam_to_screen(cam_xs, cam_ys,
                         cam_x_min, cam_x_max,
                         cam_y_min, cam_y_max,
                         screen_w, screen_h):
    """
    Scale camera iris coordinates to screen coordinates
    using the calibration bounds — exact notebook method.

    cam_x_min/max = iris x range seen during calibration
    cam_y_min/max = iris y range seen during calibration
    """
    cam_x_range = max(cam_x_max - cam_x_min, 1)
    cam_y_range = max(cam_y_max - cam_y_min, 1)

    screen_xs = []
    screen_ys = []

    for cx, cy in zip(cam_xs, cam_ys):
        # Normalize to 0-1 based on calibration range
        nx = 1.0 - (cx - cam_x_min) / cam_x_range  # flip X for mirrored webcam
        ny = (cy - cam_y_min) / cam_y_range

        # Clamp with small margin
        nx = max(-0.1, min(1.1, nx))
        ny = max(-0.1, min(1.1, ny))

        # Scale to screen
        sx = int(nx * screen_w)
        sy = int(ny * screen_h)

        screen_xs.append(sx)
        screen_ys.append(sy)

    return screen_xs, screen_ys


def get_calibration_bounds(calib_points):
    """
    Extract iris coordinate bounds from calibration data.
    Returns (x_min, x_max, y_min, y_max) in camera coords.
    """
    if not calib_points or len(calib_points) < 2:
        return 100, 500, 50, 400   # fallback defaults

    xs = [p['iris_x'] for p in calib_points if 'iris_x' in p]
    ys = [p['iris_y'] for p in calib_points if 'iris_y' in p]

    if not xs or not ys:
        return 100, 500, 50, 400

    margin_x = (max(xs) - min(xs)) * 0.1
    margin_y = (max(ys) - min(ys)) * 0.1

    return (min(xs) - margin_x, max(xs) + margin_x,
            min(ys) - margin_y, max(ys) + margin_y)


def generate_heatmap_screen(screen_xs, screen_ys,
                             screen_w=1920, screen_h=1080,
                             filename="heatmap_screen.png"):
    """
    KDE heatmap — exact notebook Cell 14 method.
    rocket_r colormap, bw_adjust=0.8
    """
    if len(screen_xs) < 5:
        return None

    my_dpi = 118   # exact from notebook
    fig_w  = screen_w  / my_dpi
    fig_h  = screen_h  / my_dpi

    plt.figure(figsize=(fig_w, fig_h), dpi=my_dpi)
    plt.axis('off')

    try:
        sns.kdeplot(
            x=screen_xs,
            y=screen_ys,
            fill=True,
            cmap='rocket_r',   # exact from notebook
            bw_adjust=0.8,     # exact from notebook
            warn_singular=False
        )
    except Exception:
        plt.scatter(screen_xs, screen_ys,
                    s=8, alpha=0.7, c='red')

    plt.gca().invert_yaxis()   # screen Y goes downward
    plt.tight_layout(pad=0)

    out_path = os.path.join(REPORTS_DIR, filename)
    plt.savefig(out_path,
                dpi=my_dpi * 2,
                bbox_inches='tight',
                pad_inches=0)
    plt.close()
    return filename


def generate_heatmap_overlay(screen_xs, screen_ys,
                              bg_image_path=None,
                              screen_w=1920, screen_h=1080,
                              filename="heatmap_overlay.png"):
    """
    Overlay heatmap on exam page screenshot.
    Exact notebook Cell 14 method:
      addWeighted(bg, 0.6, heatmap, 0.4, 0)
    """
    if len(screen_xs) < 5:
        return None

    my_dpi = 118

    # ── Step 1: KDE heatmap ───────────────────────────────────
    temp_file = os.path.join(REPORTS_DIR, f"_temp_{filename}")
    plt.figure(figsize=(screen_w/my_dpi, screen_h/my_dpi), dpi=my_dpi)
    plt.axis('off')

    try:
        sns.kdeplot(
            x=screen_xs,
            y=screen_ys,
            fill=True,
            cmap='rocket_r',
            bw_adjust=0.8,
            warn_singular=False
        )
    except Exception:
        plt.scatter(screen_xs, screen_ys,
                    s=8, alpha=0.7, c='red')

    plt.gca().invert_yaxis()
    plt.tight_layout(pad=0)
    plt.savefig(temp_file,
                dpi=my_dpi * 2,
                bbox_inches='tight',
                pad_inches=0)
    plt.close()

    # ── Step 2: Background ────────────────────────────────────
    if bg_image_path and os.path.exists(bg_image_path):
        img1 = cv2.imread(bg_image_path)
        img1 = cv2.resize(img1, (screen_w, screen_h))
    else:
        img1 = np.ones((screen_h, screen_w, 3), dtype=np.uint8) * 245
        cv2.putText(img1, "Exam Page",
                    (screen_w//2 - 100, screen_h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (180,180,180), 3)

    # ── Step 3: Load heatmap + resize ────────────────────────
    img2 = cv2.imread(temp_file)
    if img2 is None:
        return None
    img2 = cv2.resize(img2, (screen_w, screen_h))

    # ── Step 4: Blend — exact notebook values ─────────────────
    dst = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)

    out_path = os.path.join(REPORTS_DIR, filename)
    cv2.imwrite(out_path, dst)

    if os.path.exists(temp_file):
        os.remove(temp_file)

    return filename


def generate_scatter(screen_xs, screen_ys,
                     bg_image_path=None,
                     screen_w=1920, screen_h=1080,
                     filename="scatter.png"):
    """Scatter plot — notebook Cell 14 method"""
    if len(screen_xs) < 2:
        return None

    my_dpi = 118
    plt.figure(figsize=(screen_w/my_dpi, screen_h/my_dpi), dpi=my_dpi)
    plt.scatter(screen_xs, screen_ys, s=8, alpha=0.7, c='red')
    plt.axis('off')
    plt.tight_layout(pad=0)

    out_path = os.path.join(REPORTS_DIR, filename)
    plt.savefig(out_path, dpi=my_dpi*2,
                bbox_inches='tight', pad_inches=0)
    plt.close()
    return filename


def generate_both_heatmaps(gaze_points,
                            calib_points=None,
                            exam_screenshot_path=None,
                            screen_w=1920, screen_h=1080,
                            frame_w=640,   frame_h=480,
                            session_id="session"):
    """
    Main function called from Flask.
    gaze_points = list of (cam_x, cam_y) — raw camera coords
    calib_points = calibration data to get iris bounds
    """
    if not gaze_points or len(gaze_points) < 5:
        print(f"⚠️ Not enough gaze points: {len(gaze_points) if gaze_points else 0}")
        return None, None

    cam_xs = [p[0] for p in gaze_points]
    cam_ys = [p[1] for p in gaze_points]

    print(f"📊 Gaze points: {len(cam_xs)}")
    print(f"   Cam X range: {min(cam_xs):.0f}–{max(cam_xs):.0f}")
    print(f"   Cam Y range: {min(cam_ys):.0f}–{max(cam_ys):.0f}")

    # ── Use actual data range for scaling ─────────────────────
    # Camera coords range during exam — most reliable
    padding_x = (max(cam_xs) - min(cam_xs)) * 0.2 or 20
    padding_y = (max(cam_ys) - min(cam_ys)) * 0.2 or 20
    x_min = min(cam_xs) - padding_x
    x_max = max(cam_xs) + padding_x
    y_min = min(cam_ys) - padding_y
    y_max = max(cam_ys) + padding_y
    print(f"   Using data bounds: x={x_min:.1f}–{x_max:.1f}  y={y_min:.1f}–{y_max:.1f}")

    # ── Scale camera → screen ──────────────────────────────────
    screen_xs, screen_ys = scale_cam_to_screen(
        cam_xs, cam_ys,
        x_min, x_max,
        y_min, y_max,
        screen_w, screen_h
    )

    print(f"   Screen X range: {min(screen_xs)}–{max(screen_xs)}")
    print(f"   Screen Y range: {min(screen_ys)}–{max(screen_ys)}")

    # ── Generate heatmaps ──────────────────────────────────────
    screen_file = generate_heatmap_screen(
        screen_xs, screen_ys,
        screen_w, screen_h,
        filename=f"heatmap_screen_{session_id}.png"
    )
    overlay_file = generate_heatmap_overlay(
        screen_xs, screen_ys,
        bg_image_path=exam_screenshot_path,
        screen_w=screen_w,
        screen_h=screen_h,
        filename=f"heatmap_overlay_{session_id}.png"
    )

    return screen_file, overlay_file


# ── Test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing heatmap with notebook method...")
    np.random.seed(42)

    # Simulate camera iris positions (typical range 100-250, 60-150)
    reading  = [(int(np.random.normal(160, 15)),
                 int(np.random.normal(90,  10)))
                for _ in range(80)]
    options  = [(int(np.random.normal(140, 12)),
                 int(np.random.normal(110, 8)))
                for _ in range(40)]
    distract = [(int(np.random.normal(200, 20)),
                 int(np.random.normal(80,  15)))
                for _ in range(20)]

    all_pts = reading + options + distract
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]

    # Simulate calibration bounds
    calib = [
        {'iris_x': 120, 'iris_y': 70},
        {'iris_x': 200, 'iris_y': 70},
        {'iris_x': 160, 'iris_y': 95},
        {'iris_x': 120, 'iris_y': 120},
        {'iris_x': 200, 'iris_y': 120},
    ]

    f1, f2 = generate_both_heatmaps(
        gaze_points=list(zip(xs, ys)),
        calib_points=calib,
        screen_w=1920, screen_h=1080,
        session_id="test"
    )
    print(f"Screen  → reports/{f1}")
    print(f"Overlay → reports/{f2}")
    print("✅ Check reports/ folder!")