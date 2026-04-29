# OpenCV Dot Tracker for Instron Tensile Tests

Automated displacement measurement tool for Instron tensile test videos. Tracks two Sharpie marker dots on elastic specimens frame-by-frame using OpenCV, replacing manual tracking in physics software (e.g., Tracker).

> **Current scope:** designed for elastic specimens with two black Sharpie marker dots being stretched **vertically** in an Instron tensile tester. The specimen must be brighter than the Instron jaws, and the dots must be vertically separated with at least ~50px between them.

---

## Quick Install

**1. Install Python 3.8+** from [python.org](https://www.python.org/downloads/) (check "Add Python to PATH" during install)

**2. Install dependencies** — open a terminal and run:
```
pip install opencv-python numpy matplotlib pillow xlsxwriter pywin32
```
(`pywin32` is Windows-only and enables "Copy plot to clipboard" — everything else works without it.)

**3. Run the tracker:**
- Double-click `Launch Tracker.bat`, or
- Run `python app.py` from this folder

---

## Usage

1. Place `.MOV` (or `.mp4`, `.avi`) video files in `input_videos/`
   - Filenames **must** contain the initial dot separation distance, e.g. `Instron - side - 1 49.9mm.MOV`
2. Click **Add from input_videos/** or **Add Videos...** to load files
3. Choose a frame skip rate and click **Run All**
4. Once complete (✓ appears), click a video to review it
5. In the **Data** tab, click **Clean Outliers** then **Export Cleaned CSV**

---

## Features

- **Automated dot detection** — finds dots on specimen using annular contrast filtering and specimen region isolation (jaw detection)
- **Sub-pixel tracking** — adaptive blob-finding centroid refinement resists drift during large specimen stretching
- **Batch processing** — processes multiple videos sequentially in a background thread
- **Video review** — scrub through any completed video with annotated crosshair overlays
- **Displacement output** — pixel-to-mm calibration from filename-encoded initial distance; outputs displacement relative to frame 0
- **Data cleaning** — rolling median + MAD outlier removal
- **CSV export** — auto-saves to `output_data/` on completion; manual export also available

---

## Validation

The `validation/` folder contains:
- `build_comparison.py` — generates an Excel comparison between manual Tracker output and OpenCV output
- `Tracker vs openCV Comparison.xlsx` — validation result from a representative test

Validated accuracy: **<0.5% mean difference**, **R² ≈ 0.9999** against manual Tracker ground truth.

---

## Technical Documentation

See [`TECHNICAL.md`](TECHNICAL.md) for a full explanation of how the tracking pipeline works.
