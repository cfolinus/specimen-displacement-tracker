"""
Split and rename video files based on a Video log Excel sheet.

Edit the settings block below, then run:
    python rename_and_split_videos.py

Required Excel columns:
    "Original video name"  — source filename stem (e.g. C0650)
    "Should be renamed?"   — only rows marked True are processed
    "Renamed base file"    — base name for output clips (e.g. Tension - Specimen D10 - 2)
    "Trials"               — trial range as [start, end] (e.g. [1, 5])
    "Timestamps"           — comma-separated split points (e.g. 0:30, 1:00, 1:30, 2:00)

Example for C0650 with Trials=[1,5] and four timestamps:
    → Tension - Specimen D10 - 2 - Trial 1.mp4
    → Tension - Specimen D10 - 2 - Trial 2.mp4
    → Tension - Specimen D10 - 2 - Trial 3.mp4
    → Tension - Specimen D10 - 2 - Trial 4.mp4
    → Tension - Specimen D10 - 2 - Trial 5.mp4
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Settings — edit these before running
# ---------------------------------------------------------------------------

XLSX_PATH = "input_videos/2026-06-15 Videos/Video log - sandbox.xlsx"

# Input videos are read from this folder (relative to the xlsx file's directory).
# Set to "" to look in the same folder as the xlsx file.
INPUT_SUBDIR = ""

# Output clips are written here (relative to the xlsx file's directory).
OUTPUT_SUBDIR = "Split"

# Set to True to delete each source video after its clips are created.
DELETE_ORIGINALS = False

# Set to True to skip output files that already exist; False to overwrite them.
SKIP_EXISTING = True

# Encoding: "copy" is fast but cuts at keyframes; "encode" is frame-accurate.
ENCODE_MODE = "copy"

# ---------------------------------------------------------------------------


def parse_timestamps(ts_string: str) -> list[float]:
    """Parse a comma-separated string of M:SS or H:MM:SS timestamps into seconds."""
    result = []
    for token in ts_string.split(","):
        token = token.strip()
        if not token:
            continue
        parts = [float(p) for p in token.split(":")]
        if len(parts) == 2:
            result.append(parts[0] * 60 + parts[1])
        elif len(parts) == 3:
            result.append(parts[0] * 3600 + parts[1] * 60 + parts[2])
        else:
            raise ValueError(f"Unrecognised timestamp format: {token!r}")
    return result


def parse_trial_range(trials_str: str) -> tuple[int, int]:
    """Parse '[start, end]' into (start, end) integers."""
    stripped = trials_str.strip().lstrip("[").rstrip("]")
    parts = [p.strip() for p in stripped.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected '[start, end]' format, got: {trials_str!r}")
    return int(float(parts[0])), int(float(parts[1]))


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


def find_video(stem: str, video_dir: Path) -> Path | None:
    """Find a video file by stem, accepting .mp4 or .MP4."""
    for ext in (".mp4", ".MP4", ".mov", ".MOV"):
        candidate = video_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def split_and_rename(
    video_path: Path,
    timestamps: list[float],
    trial_start: int,
    output_base: str,
    output_dir: Path,
    delete_original: bool,
) -> None:
    duration = get_duration(video_path)
    boundaries = [0.0] + sorted(timestamps) + [duration]
    n_chunks = len(boundaries) - 1
    trial_end = trial_start + n_chunks - 1

    codec_args = (
        ["-c:v", "copy"] if ENCODE_MODE == "copy"
        else ["-c:v", "libx264", "-crf", "18", "-preset", "fast"]
    )

    print(f"\n{video_path.name}: {n_chunks} chunk(s)  →  trials {trial_start}–{trial_end}")

    for i in range(n_chunks):
        start, end = boundaries[i], boundaries[i + 1]
        trial_num = trial_start + i
        out_name = f"{output_base} - Trial {trial_num}.mp4"
        out_path = output_dir / out_name

        if out_path.exists():
            if SKIP_EXISTING:
                print(f"  SKIP  [{trial_num}] {out_name!r} — already exists")
                continue
            print(f"  OVER  [{trial_num}] {out_name!r} — overwriting")

        result = subprocess.run(
            ["ffmpeg", "-y", "-ss", str(start), "-to", str(end),
             "-i", str(video_path), *codec_args, "-an", str(out_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError(f"ffmpeg failed on trial {trial_num} of {video_path.name}")

        print(f"  OK    [{trial_num}] {start:.1f}s – {end:.1f}s ({end - start:.1f}s)  →  {out_name!r}")

    if delete_original:
        video_path.unlink()
        print(f"  DEL   {video_path.name}")


def load_jobs(xlsx_path: str) -> list[dict]:
    import pandas as pd

    df = pd.read_excel(xlsx_path, header=0, dtype=str)

    required = {"Original video name", "Should be renamed?", "Renamed base file", "Trials", "Timestamps"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: Excel sheet is missing columns: {missing}")
        sys.exit(1)

    mask = df["Should be renamed?"].str.strip().str.lower() == "true"
    df = df[mask]

    jobs = []
    for _, row in df.iterrows():
        stem = str(row["Original video name"]).strip()
        base = str(row["Renamed base file"]).strip()
        trials_str = str(row["Trials"]).strip()
        ts_str = str(row["Timestamps"]).strip()

        if any(v in ("nan", "") for v in [stem, base, trials_str, ts_str]):
            print(f"  SKIP  row with stem={stem!r} — missing required fields")
            continue

        jobs.append({
            "stem": stem,
            "base": base,
            "trials_str": trials_str,
            "ts_str": ts_str,
        })

    return jobs


def main():
    xlsx_dir = Path(os.path.abspath(XLSX_PATH)).parent
    video_dir = xlsx_dir / INPUT_SUBDIR if INPUT_SUBDIR else xlsx_dir
    output_dir = xlsx_dir / OUTPUT_SUBDIR

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Excel:            {XLSX_PATH}")
    print(f"Video dir:        {video_dir}")
    print(f"Output dir:       {output_dir}")
    print(f"Delete originals: {DELETE_ORIGINALS}")
    print(f"Skip existing:    {SKIP_EXISTING}\n")

    jobs = load_jobs(XLSX_PATH)
    print(f"{len(jobs)} video(s) to process")

    for job in jobs:
        stem = job["stem"]
        video_path = find_video(stem, video_dir)

        if video_path is None:
            print(f"\n  SKIP  {stem!r} — not found in {video_dir}")
            continue

        try:
            timestamps = parse_timestamps(job["ts_str"])
            trial_start, trial_end = parse_trial_range(job["trials_str"])
        except ValueError as e:
            print(f"\n  SKIP  {stem!r} — parse error: {e}")
            continue

        expected_chunks = trial_end - trial_start + 1
        actual_chunks = len(timestamps) + 1
        if expected_chunks != actual_chunks:
            print(
                f"\n  WARN  {stem!r} — Trials {job['trials_str']} implies {expected_chunks} chunks "
                f"but {len(timestamps)} timestamps give {actual_chunks} chunks; proceeding anyway"
            )

        t0 = time.time()
        split_and_rename(
            video_path=video_path,
            timestamps=timestamps,
            trial_start=trial_start,
            output_base=job["base"],
            output_dir=output_dir,
            delete_original=DELETE_ORIGINALS,
        )
        print(f"  Done in {time.time() - t0:.1f}s")

    print("\nAll done.")


if __name__ == "__main__":
    main()
