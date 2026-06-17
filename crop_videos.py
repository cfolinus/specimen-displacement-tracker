"""Crop all videos in 'No audio/' and save to 'Cropped/'.

The user picks a crop region once (drag or type) from the first video's first
frame; that same box is applied to every video via ffmpeg's crop filter.

Tunables (INPUT_DIR, OUTPUT_DIR, CROP_BOX) are set just below.
"""

import subprocess
import time
from pathlib import Path

import cv2

# --- Tunable settings -----------------------------------------------------------
INPUT_DIR = Path("input_videos/2026-06-15 Videos/Split")
OUTPUT_DIR = INPUT_DIR.parent / "Cropped"

# Crop box as (x, y, width, height) in pixels, applied to every video.
# Set to None to pick it interactively from the first video's first frame; the
# chosen box is printed so it can be pasted here for future runs.
CROP_BOX = None


def select_crop_box(frame):
    """Let the user pick a crop box by dragging on `frame` or typing coordinates.

    Returns (x0, y0, w, h) as ints.
    """
    height, width = frame.shape[:2]
    print(f"\nFrame size: {width} x {height}")
    choice = input("Select crop by [d]ragging on the image or [t]yping coordinates? [d/t]: ").strip().lower()

    if choice == "t":
        x0 = int(input("  x (left edge): ").strip())
        y0 = int(input("  y (top edge): ").strip())
        w  = int(input("  width: ").strip())
        h  = int(input("  height: ").strip())
    else:
        window = "Drag to select crop box — press ENTER or SPACE to confirm (ESC to cancel)"
        x0, y0, w, h = cv2.selectROI(window, frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(window)
        x0, y0, w, h = int(x0), int(y0), int(w), int(h)

    if w <= 0 or h <= 0:
        raise ValueError("Crop box must have positive width and height.")

    # ffmpeg's crop filter requires even dimensions for most codecs.
    w -= w % 2
    h -= h % 2

    print(f"\nCrop box: CROP_BOX = ({x0}, {y0}, {w}, {h})  # (x, y, width, height)")
    print("Paste this as CROP_BOX above to reuse it for future runs.\n")

    return x0, y0, w, h


def get_crop_box_from_first_video(input_path):
    """Read the first frame of `input_path` and let the user pick a crop box."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame of: {input_path}")

    print(f"Using first frame of '{input_path.name}' for crop selection.")
    return select_crop_box(frame)


def crop_video(input_path, output_path, crop_box):
    """Crop one video with ffmpeg's crop filter and save to output_path."""
    x0, y0, w, h = crop_box
    crop_filter = f"crop={w}:{h}:{x0}:{y0}"

    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", crop_filter,
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-an",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"ffmpeg failed on {input_path.name}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_paths = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".mp4", ".mov"}
    )
    total = len(input_paths)

    if total == 0:
        print(f"No video files found in: {INPUT_DIR}")
        return

    print(f"Found {total} video(s) in {INPUT_DIR}")

    crop_box = CROP_BOX if CROP_BOX is not None else get_crop_box_from_first_video(input_paths[0])

    for i, input_path in enumerate(input_paths, start=1):
        output_path = OUTPUT_DIR / (input_path.stem + ".mp4")
        remaining = total - i
        print(f"[{i}/{total}] {input_path.name}  ({remaining} remaining after this)")

        start = time.time()
        crop_video(input_path, output_path, crop_box)
        elapsed = time.time() - start

        print(f"  -> {output_path} ({elapsed:.1f}s)")

    print(f"\nDone. {total} video(s) saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
