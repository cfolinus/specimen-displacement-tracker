# OpenCV Dot Tracker for Instron Tensile Tests

Automated displacement/position measurement tool for Instron tensile test and mechanism videos. Tracks marker dots frame-by-frame using OpenCV, replacing manual tracking in physics software (e.g., Tracker).

## Supported test types

The tracker selects which kind of video it's looking at from a **keyword in the filename** (case-insensitive). Every video **must** contain one of these keywords — there is no default. A file with no recognized keyword is **skipped** (marked ✗) with a message listing the valid types, rather than being silently run through the wrong detector:

| Test type | Filename keyword | What it tracks | Output |
|---|---|---|---|
| **tensile** | filename contains `tensile` | 2 dark Sharpie dots on a bright specimen, stretched **vertically** between Instron jaws. Specimen must be brighter than the jaws; dots must be ≥50px apart vertically. | px and/or mm (if filename has `<number>mm`), inter-dot distance/displacement |
| **roller** | filename contains `roller` | 1–2 bright magenta paint-pen dots on a grey mechanism | px and/or mm, per-dot displacement |
| **hinge** | filename contains `hinge` (but not `hinge colored`/`hinge-colored`/`hinge_colored` or `hinge tension`/`hinge-tension`/`hinge_tension`) | Two rows of up to 4 small black marker dots, one row per coupler bar of a hinge mechanism, tracked **independently**, labeled `set1_1...set1_4/set2_1...set2_4` (3 dots accepted per set if the faint 4th can't be found) | px only (no mm calibration, no inter-dot distance) |
| **hinge_colored** | filename contains `hinge colored`, `hinge-colored`, or `hinge_colored` | 2 green + 2 yellow paint dots on a hinge mechanism, tracked **independently**, labeled `green1/green2/yellow1/yellow2` | px only (no mm calibration, no inter-dot distance) |
| **hinge_tension** | filename contains `hinge tension`, `hinge-tension`, or `hinge_tension` | Dark marker dots inside two user-defined **rotated-strip ROIs** (one per coupler bar / gauge region). ROIs are set interactively before running via the **Set ROIs…** button. | px only (no mm calibration, no inter-dot distance) |

Tracking overlays (crosshairs/circles) are drawn in **bright blue**.

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
   - **Every filename must contain a test-type keyword** — `tensile`, `roller`, `hinge`, `hinge colored`, or `hinge tension` — or the video is skipped. e.g. `Tensile - Instron - side - 1 49.9mm.MOV`, `Roller test 3.MP4`, `Hinge sample 12.mp4`, `Hinge colored sample 12.mp4`, `Hinge tension sample 3.mp4`
   - `hinge` alone selects the black-dot detector; `hinge colored` (or `hinge-colored`/`hinge_colored`) selects the green/yellow paint-dot detector; `hinge tension` (or `hinge-tension`/`hinge_tension`) selects the rotated-strip ROI detector
   - For **tensile**/**roller** videos with mm calibration, the filename **must** also contain the initial dot separation distance, e.g. `Tensile - Instron - side - 1 49.9mm.MOV`
2. Click **Add from input_videos/** or **Add Videos...** to load files
3. For **hinge_tension** videos, select the video in the list and click **Set ROIs…** to define the two rotated-strip ROIs interactively (left-click two points per ROI, press ENTER to confirm, R to redo)
4. Choose a frame skip rate and click **Run All** (or **Run Selected** to process only the highlighted video)
5. Once complete (✓ appears), click a video to review it
6. In the **Data** tab, click **Clean Outliers** then **Export Cleaned CSV**

---

## Features

- **Automated dot detection** — tensile (dark Sharpie dots via annular contrast filtering + jaw-based specimen isolation), roller (bright magenta paint dots via HSV), hinge (two rows of small black marker dots via multi-threshold contrast + collinear-group fitting), hinge_colored (green/yellow paint dots via HSV, with merged-dot splitting), and hinge_tension (dark dots inside user-defined rotated-strip ROIs)
- **Sub-pixel tracking** — adaptive blob-finding centroid refinement resists drift during large specimen stretching or mechanism motion
- **Batch processing** — processes multiple videos sequentially in a background thread; **Run All** processes all listed videos, **Run Selected** processes only the highlighted video
- **Rotated-strip ROI selection** — for `hinge_tension` videos, the **Set ROIs…** button opens an interactive first-frame window to define two tight rotated-rectangle search zones that exclude surrounding bolt hardware
- **Video review** — scrub through any completed video with annotated crosshair overlays (bright blue)
- **Displacement output** — pixel-to-mm calibration from filename-encoded initial distance (tensile/roller); outputs displacement relative to frame 0
- **Multi-dot tracking** — up to 8 dots tracked independently per video; hinge mode labels dots by set/position (`set1_1`...`set1_4`, `set2_1`...`set2_4`), hinge_colored mode labels dots by color (`green1`, `green2`, `yellow1`, `yellow2`)
- **Output variable selection** — checkboxes to choose which columns appear in the CSV: pixel position, scaled mm position, per-dot displacement, inter-dot displacement, inter-dot distance
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
