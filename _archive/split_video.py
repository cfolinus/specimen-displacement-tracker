"""Split video(s) into segments at specified timestamps.

Usage:
    python split_video.py                    # use hardcoded VIDEO_PATH and TIMESTAMPS below
    python split_video.py splits.xlsx        # use Excel sheet (columns: Video, Timestamps)

Excel sheet format:
    Video       — filename stem in INPUT_DIR, e.g. "C0650"
    Timestamps  — comma-separated split points, e.g. "0:35, 1:11, 1:51, 2:28"
"""

import json
import subprocess
import sys
import time
from pathlib import Path

# --- Tunable settings -----------------------------------------------------------
INPUT_DIR  = Path("input_videos/2026-06-15 Videos/Cropped")
OUTPUT_DIR = Path("input_videos/2026-06-15 Videos/Split")

# Hardcoded job (used when no xlsx is passed)
VIDEO_PATH = INPUT_DIR / "C0650.mp4"
TIMESTAMPS = "0:35, 1:11, 1:51, 2:28"

# Encoding: "copy" is fast but cuts at keyframes; "encode" is frame-accurate.
ENCODE_MODE = "copy"
# ---------------------------------------------------------------------------------


def parse_timestamps(ts_string: str) -> list[float]:
    """Parse a comma-separated string of M:SS or H:MM:SS timestamps into seconds."""
    result = []
    for token in ts_string.split(","):
        token = token.strip()
        if not token:
            continue
        parts = [float(p) for p in token.split(":")]
        if len(parts) == 2:
            seconds = parts[0] * 60 + parts[1]
        elif len(parts) == 3:
            seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
        else:
            raise ValueError(f"Unrecognised timestamp format: {token!r}")
        result.append(seconds)
    return result


def get_duration(video_path: Path) -> float:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", str(video_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {video_path.name}:\n{result.stderr}")
    return float(json.loads(result.stdout)["format"]["duration"])


def split_video(video_path: Path, timestamps: list[float], output_dir: Path) -> None:
    duration = get_duration(video_path)
    stem = video_path.stem
    boundaries = [0.0] + sorted(timestamps) + [duration]
    total = len(boundaries) - 1

    codec_args = (
        ["-c:v", "copy"] if ENCODE_MODE == "copy"
        else ["-c:v", "libx264", "-crf", "18", "-preset", "fast"]
    )

    print(f"\n{stem}: {total} segment(s)")

    for i in range(total):
        start, end = boundaries[i], boundaries[i + 1]
        out_path = output_dir / f"{stem} - {i + 1}.mp4"

        result = subprocess.run(
            ["ffmpeg", "-y", "-ss", str(start), "-to", str(end),
             "-i", str(video_path), *codec_args, "-an", str(out_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError(f"ffmpeg failed on segment {i + 1} of {video_path.name}")

        print(f"  [{i + 1}/{total}] {start:.1f}s – {end:.1f}s ({end - start:.1f}s)  -> {out_path.name}")


def load_jobs_from_excel(xlsx_path: str) -> list[tuple[str, str]]:
    import pandas as pd

    df = pd.read_excel(xlsx_path, header=0, usecols=["Video", "Timestamps"], dtype=str)
    df = df.dropna()
    return [(r.Video.strip(), r.Timestamps.strip()) for r in df.itertuples()]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 1:
        xlsx_path = sys.argv[1]
        print(f"Loading jobs from {xlsx_path!r}")
        jobs = load_jobs_from_excel(xlsx_path)
        print(f"{len(jobs)} video(s) to split\n")

        for video_stem, ts_string in jobs:
            candidates = [
                p for p in INPUT_DIR.glob(f"{video_stem}.*")
                if p.suffix.lower() in {".mp4", ".mov"}
            ]
            if not candidates:
                print(f"  SKIP  {video_stem!r} — not found in {INPUT_DIR}")
                continue

            t0 = time.time()
            split_video(candidates[0], parse_timestamps(ts_string), OUTPUT_DIR)
            print(f"  Done in {time.time() - t0:.1f}s")
    else:
        print("Using hardcoded job")
        t0 = time.time()
        split_video(VIDEO_PATH, parse_timestamps(TIMESTAMPS), OUTPUT_DIR)
        print(f"\nDone in {time.time() - t0:.1f}s — saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
