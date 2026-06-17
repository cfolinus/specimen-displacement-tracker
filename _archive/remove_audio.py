"""Remove audio from all videos in a folder using ffmpeg (stream copy, no re-encode).

Reads every .mp4 / .MP4 / .MOV / .mov in INPUT_DIR and writes a silent copy to OUTPUT_DIR,
leaving originals untouched.
"""

import subprocess
import time
from pathlib import Path

INPUT_DIR = Path("input_videos/2026-06-15 Videos")
OUTPUT_DIR = INPUT_DIR / "No audio"

VIDEO_EXTENSIONS = {".mp4", ".mov"}


def remove_audio(input_path: Path, output_path: Path) -> None:
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_path), "-c:v", "copy", "-an", str(output_path)],
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
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    total = len(input_paths)

    if total == 0:
        print(f"No video files found in: {INPUT_DIR}")
        return

    print(f"Found {total} video(s) in {INPUT_DIR}")

    for i, input_path in enumerate(input_paths, start=1):
        output_path = OUTPUT_DIR / (input_path.stem + ".mp4")
        print(f"\n[{i}/{total}] {input_path.name}")

        start = time.time()
        remove_audio(input_path, output_path)
        elapsed = time.time() - start
        print(f"  -> {output_path} ({elapsed:.1f}s)")

    print(f"\nDone. {total} video(s) saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
