"""
Rename video files based on a Video log Excel sheet.

Usage:
    python rename_videos.py <xlsx_path> [video_dir]

    xlsx_path  — path to the Video log Excel file
    video_dir  — folder containing the videos (default: same folder as xlsx_path)

The script reads two columns from the sheet:
    "Original video name"  — source filename stem (e.g. C0650)
    "Renamed base file"    — desired output filename stem (e.g. Tension - Specimen D10 - 2)

Only rows where "Should be renamed?" is True and "Renamed?" is blank are processed.
The renamed file lands in the same folder as the original.
"""

import os
import sys


def resolve_path(path: str) -> str | None:
    """Return the actual path on disk, trying both .mp4 and .MP4 extensions."""
    if os.path.isfile(path):
        return path
    root, ext = os.path.splitext(path)
    if ext.lower() == ".mp4":
        alt = root + (".MP4" if ext == ".mp4" else ".mp4")
        if os.path.isfile(alt):
            return alt
    return None


def load_mapping_from_excel(xlsx_path: str, video_dir: str) -> list[tuple[str, str]]:
    import pandas as pd

    df = pd.read_excel(xlsx_path, header=0, dtype=str)

    required = {"Original video name", "Renamed base file", "Should be renamed?"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: Excel sheet is missing columns: {missing}")
        sys.exit(1)

    mask = df["Should be renamed?"].str.strip().str.lower() == "true"
    df = df[mask]

    mapping = []
    for _, row in df.iterrows():
        stem = str(row["Original video name"]).strip()
        new_stem = str(row["Renamed base file"]).strip()
        if not stem or not new_stem or stem == "nan" or new_stem == "nan":
            continue

        # Try both extensions
        for ext in (".mp4", ".MP4"):
            original_path = os.path.join(video_dir, stem + ext)
            if os.path.isfile(original_path):
                break
        else:
            original_path = os.path.join(video_dir, stem + ".mp4")  # will show as not found

        # Preserve the actual extension of the found file
        resolved = resolve_path(original_path)
        actual_ext = os.path.splitext(resolved)[1] if resolved else ".mp4"
        new_name = new_stem + actual_ext

        mapping.append((original_path, new_name))

    return mapping


def rename_videos(mapping: list[tuple[str, str]]) -> None:
    for original_path, new_name in mapping:
        original_path = os.path.normpath(original_path)

        resolved = resolve_path(original_path)
        if resolved is None:
            print(f"  SKIP  {os.path.basename(original_path)!r} — file not found")
            continue
        original_path = resolved

        directory = os.path.dirname(original_path)
        new_path = os.path.join(directory, new_name)

        if os.path.exists(new_path):
            print(f"  SKIP  {original_path!r} — destination already exists: {new_path!r}")
            continue

        os.rename(original_path, new_path)
        print(f"  OK    {os.path.basename(original_path)!r}  →  {new_name!r}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    xlsx_path = sys.argv[1]
    video_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(xlsx_path))

    print(f"Excel:     {xlsx_path}")
    print(f"Video dir: {video_dir}\n")

    mapping = load_mapping_from_excel(xlsx_path, video_dir)
    print(f"Renaming {len(mapping)} file(s)...\n")
    rename_videos(mapping)


if __name__ == "__main__":
    main()
