"""Edit specimen videos: strip audio, crop, boost saturation, save copies.

Processes every .mp4 in INPUT_DIR and writes edited copies to OUTPUT_DIR,
leaving originals untouched. See "README - video trimming.md" for usage.

Pipeline (OpenCV only; audio is dropped for free since cv2 never writes it):
    select crop box (once, shared by all videos) -> open -> set up writer
    -> per-frame (crop + saturate) -> release

Tunables (SATURATION_SCALE, INPUT_DIR, OUTPUT_DIR, CROP_BOX) are set just below.
"""

import time
from pathlib import Path

import cv2
import numpy as np

# --- Tunable settings (the things we'll revisit later live here) ----------------
SATURATION_SCALE = 1.8    # multiply the HSV S channel by this (>1 = more saturated)

INPUT_DIR = Path("input_videos/2026-06-15 Videos")
OUTPUT_DIR = INPUT_DIR / "Edited"

# Crop box as (x, y, width, height) in pixels, applied to every video in the batch.
# Set to None to pick it interactively from the first video's first frame; the
# chosen box is printed in this format so it can be pasted here for future runs.
CROP_BOX = None


def select_crop_box(frame):
    """Ask the user for a crop box, by dragging on `frame` or typing coordinates.

    Returns (x0, y0, w, h) as ints. Prints the result in a copy-pasteable format
    so it can be reused as CROP_BOX in future runs.
    """
    height, width = frame.shape[:2]
    print(f"\nFrame size: {width} x {height}")
    choice = input("Select crop box by [d]ragging on the image or [t]yping coordinates? [d/t]: ").strip().lower()

    if choice == "t":
        x0 = int(input("  x (left): ").strip())
        y0 = int(input("  y (top): ").strip())
        w = int(input("  width: ").strip())
        h = int(input("  height: ").strip())
    else:
        # Any input other than "t" (including the default empty Enter) opens
        # the drag-to-select GUI.
        window = "Drag to select crop box, then press ENTER/SPACE (ESC to cancel)"
        x0, y0, w, h = cv2.selectROI(window, frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(window)
        x0, y0, w, h = int(x0), int(y0), int(w), int(h)

    if w <= 0 or h <= 0:
        raise ValueError("Crop box must have positive width and height.")

    # The mp4v codec requires even width/height, silently truncating odd ones.
    # Round down here so the printed/reused box always matches the actual output.
    w -= w % 2
    h -= h % 2

    print("\nCrop box selected:")
    print(f"  CROP_BOX = ({x0}, {y0}, {w}, {h})  # (x, y, width, height)")
    print("Paste this as CROP_BOX in edit_video.py to reuse it for future runs.\n")

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

    print(f"Using first frame of '{input_path.name}' for crop box selection.")
    return select_crop_box(frame)


def boost_saturation(frame_bgr, scale=SATURATION_SCALE):
    """Return a copy of the BGR frame with the S channel scaled by `scale`.

    Work in float to avoid uint8 overflow, clip to 0-255, convert back to BGR.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def edit_video(input_path, output_path, crop_box):
    """Process one video file: crop + saturate every frame, write to output_path.

    `crop_box` is (x0, y0, w, h) in pixels, applied to every frame.
    The input is opened read-only; the original is never modified.
    """
    x0, y0, cw, ch = crop_box

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if x0 + cw > width or y0 + ch > height:
        cap.release()
        raise ValueError(
            f"Crop box {crop_box} exceeds frame size ({width}x{height}) for {input_path.name}"
        )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (cw, ch))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            cropped = frame[y0:y0 + ch, x0:x0 + cw]
            out_frame = boost_saturation(cropped)
            writer.write(out_frame)
    finally:
        cap.release()
        writer.release()


def main():
    """Edit every .mp4 in INPUT_DIR, sharing one crop box across the whole batch."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Non-recursive glob: files already inside OUTPUT_DIR ("Edited/") are skipped.
    # Match extension case-insensitively (e.g. some cameras write ".MP4").
    input_paths = sorted(
        p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"
    )
    total = len(input_paths)

    if total == 0:
        print(f"No .mp4 files found in: {INPUT_DIR}")
        return

    crop_box = CROP_BOX if CROP_BOX is not None else get_crop_box_from_first_video(input_paths[0])

    for i, input_path in enumerate(input_paths, start=1):
        output_path = OUTPUT_DIR / input_path.name
        remaining = total - i
        print(f"[{i}/{total}] Processing: {input_path.name} ({remaining} remaining after this)")

        start = time.time()
        edit_video(input_path, output_path, crop_box)
        elapsed = time.time() - start

        print(f"  -> saved to {output_path} ({elapsed:.1f}s)")

    print(f"Done. Processed {total} video(s).")


if __name__ == "__main__":
    main()
