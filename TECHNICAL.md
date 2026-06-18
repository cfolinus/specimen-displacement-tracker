# Technical Documentation

## Overview

This tracker automates displacement measurement from Instron test videos across five test types: `tensile`, `roller`, `hinge`, `hinge_colored`, and `hinge_tension`. The test type is auto-detected from the filename. Each type uses a different initial dot detector; all types share the same template-matching tracker and centroid refinement pipeline.

The pipeline has five main stages:

1. Initial dot detection (per test type)
2. Template extraction
3. Frame-by-frame template matching
4. Centroid refinement (per test type)
5. Output and calibration

---

## File Structure

```
specimen-displacement-tracker/
├── app.py              — GUI application (tkinter)
├── tracker_core.py     — Core tracking engine (OpenCV)
├── Launch Tracker.bat  — Windows one-click launcher
├── input_videos/       — Place .MOV files here
├── output_data/        — Auto-saved CSV results
└── validation/
    ├── build_comparison.py              — Generates comparison Excel sheet
    └── Tracker vs openCV Comparison.xlsx
```

---

## tracker_core.py

### `VALID_TEST_TYPES` and `detect_test_type(filename)`

`VALID_TEST_TYPES = ('tensile', 'roller', 'hinge', 'hinge_colored', 'hinge_tension')`

`detect_test_type` infers the test type from the filename stem (lowercased). Detection order matters — more specific aliases are checked before shorter substrings to avoid false matches:

1. `roller` — filename contains `roller`
2. `hinge_colored` — filename contains `hinge colored`, `hinge-colored`, or `hinge_colored`
3. `hinge_tension` — filename contains `hinge tension`, `hinge-tension`, or `hinge_tension`
4. `hinge` — filename contains `hinge` (plain, not matched by 2 or 3)
5. `tensile` — filename contains `tensile`

Returns `None` if no keyword matches. `VideoTracker.open()` treats `None` as a hard error and marks the video as failed rather than guessing.

---

### `extract_initial_distance_mm(filename)`

Parses the filename to extract the initial dot separation in mm using a regex pattern matching `<number>mm` (e.g., `49.9mm`). This value is used as the pixel-to-mm calibration reference (tensile and roller only).

---

### `find_specimen_region(gray)`

Detects the bright specimen region between the two dark Instron jaw clamps.

**How it works:**
1. Counts dark pixels (intensity < 60) per row across the full frame width
2. Smooths the count with a 10-row moving average
3. Classifies any row where >30% of pixels are dark as a "jaw row"
4. Finds contiguous jaw bands taller than 30px
5. Identifies the gap between adjacent jaw bands that has the highest average brightness in its center strip — this is the specimen

**Why this matters:** Without jaw detection, the tracker could latch onto dark rig hardware (bolts, clamp edges) that score higher contrast than the actual dots, especially in zoomed-out video frames.

**Returns:** `(y_min, y_max)` of the specimen region, or `None` if fewer than 2 jaw bands are found.

---

### `find_initial_dots(gray)`

Detects exactly two Sharpie marker dots in the first frame.

**How it works:**

**Step 1 — Restrict search area:**  
Limits detection to the specimen region (from `find_specimen_region`) plus a small margin. Also excludes the leftmost and rightmost quarters of the frame, since dots are always placed near the specimen centerline.

**Step 2 — Multi-threshold blob detection:**  
Iterates threshold values from 90 to 170 (step 5). At each threshold, inverts the image (dark pixels become white), applies morphological opening to remove noise, and extracts contours. Contours are filtered by:
- Area: 15–500 px²
- Aspect ratio: 0.15–6.0 (excludes extreme elongation)

**Step 3 — Annular contrast filter:**  
For each candidate blob centroid, computes:
- `surround_mean`: mean intensity in an annulus from radius 8–25px around the centroid
- `center_mean`: mean intensity in a 7×7px patch at the centroid
- `contrast = surround_mean - center_mean`

Only candidates where `surround_mean > 140` (bright surroundings) **and** `contrast > 40` (clearly darker than surroundings) pass. This rejects dark hardware features that sit on dark backgrounds.

**Step 4 — Clustering:**  
Candidates within 15px of each other (the same dot detected at multiple thresholds) are merged into a single cluster. Each cluster's score is `max_contrast × detection_count`.

**Step 5 — Pair selection:**  
Among the top 8 clusters by score, selects the pair with the highest combined score where the two dots are at least 50px apart vertically. Returns them sorted top-to-bottom.

**Returns:** `[(x1, y1), (x2, y2)]` or `None`.

---

### `find_initial_dots(frame_bgr, test_type='tensile')`

Dispatcher that calls the appropriate detector for the given test type:

- `tensile` → `find_initial_dots_tensile(gray)` — annular contrast filter on grayscale (described above)
- `roller` → `find_initial_dots_roller(frame_bgr)` — HSV-based magenta blob detection
- `hinge_colored` → `find_initial_dots_color(frame_bgr)` — HSV-based green/yellow dot detection with merged-dot splitting; returns labeled `(point, color)` pairs
- `hinge` → `find_initial_dots_hinge_black(frame_bgr)` — multi-threshold dark blob detection with collinear-group fitting; returns labeled `(point, 'set1'/'set2')` pairs
- `hinge_tension` → `find_initial_dots_test(frame_bgr, roi1, roi2)` — dark dot detection restricted to two rotated-strip ROIs; returns labeled `(point, 'set1'/'set2')` pairs

---

### `track_dot_template(gray, template, last_pos, search_radius=60)`

Tracks a single dot in a new frame using Normalized Cross-Correlation (NCC).

**How it works:**
1. Defines a search window centered on the dot's last known position, extended by `search_radius` in each direction
2. Runs `cv2.matchTemplate` with `TM_CCOEFF_NORMED` (values range −1 to 1; 1 = perfect match)
3. Rejects the result if the peak correlation score is below 0.25

**Why NCC:** Normalized cross-correlation is invariant to uniform brightness changes, which matters because the specimen stretches and the dot appearance changes (it elongates and may fade as the specimen deforms).

**Returns:** `(cx, cy), score` or `(None, score)` on failure.

---

### `refine_centroid(frame_bgr, gray, pos, test_type, patch_size=30, color=None)`

Dispatcher that calls the right centroid refinement for the given test type:

- `tensile` → `refine_centroid_dark(gray, pos, patch_size=30)` — local adaptive contrast (background − pixel), connected-component weighted centroid
- `roller` → `refine_centroid_bright(frame_bgr, pos, patch_size=30)` — saturation-weighted centroid in the magenta hue range
- `hinge_colored` → `refine_centroid_color(frame_bgr, pos, color, patch_size=10)` — saturation-weighted centroid in the dot's own hue range; patch capped at 10px to avoid bleeding into adjacent dots
- `hinge` → `refine_centroid_dark(gray, pos, patch_size=12)` — same as tensile dark refinement; patch capped at 12px
- `hinge_tension` → `refine_centroid_dark(gray, pos, patch_size=15)` — same as tensile dark refinement; patch capped at 15px

#### `refine_centroid_dark` detail

**Problem it solves:** As the specimen stretches, dots elongate and the template match may land slightly off-center. Simple intensity thresholding fails because overall frame brightness varies. This finds the dark anomaly relative to the local background regardless of absolute intensity.

**How it works:**
1. Extracts a patch of size `2×patch_size` around the template position
2. Computes a heavy Gaussian blur (σ=12) of the patch — this estimates the local background intensity at each pixel, as if the dot weren't there
3. Computes `contrast_map = blurred − patch` — positive values indicate pixels darker than their surroundings (i.e., the dot)
4. Thresholds to keep only pixels with contrast > 35% of the peak contrast in the patch
5. Finds connected components in the mask and selects the component closest to the patch center
6. Computes an intensity-weighted centroid within that component

**Returns:** `(cx, cy)` in full-frame coordinates.

---

### `VideoTracker` class

Stateful, frame-by-frame tracker designed for GUI integration. Handles 1 to 8 dots per video across all test types.

**Constructor:** `VideoTracker(video_path, frame_skip=1, initial_distance_mm=None, test_type=None, rois=None)`  
`rois` is a list of `((x1,y1), (x2,y2), half_width)` tuples used by `hinge_tension`; ignored for all other types.

**Key state:**
- `dots`: list of current dot positions `[(x, y), ...]` in full-frame float coordinates
- `colors`: parallel list of dot identity labels — `None` for tensile/roller, `'green'`/`'yellow'` for hinge_colored, `'set1'`/`'set2'` for hinge and hinge_tension
- `templates`: rolling NCC templates (updated every 15 frames), one per dot
- `ref_templates`: original first-frame templates (fallback if rolling template fails)
- `n_dots`: number of dots tracked (1–8)
- `results`: list of `(time_s, inter_dot_distance_or_None)` tuples; distance is `None` when `n_dots != 2`
- `positions`: list of `[(x,y), ...]` snapshots, parallel to `results`
- `frame_indices`: list of frame numbers, parallel to `results`
- `redetections`: count of successful re-locks via the full initial detector

**`open()`:**  
Opens the video, reads the first frame, dispatches to the correct initial detector (passing ROIs for `hinge_tension`), refines each detected centroid, extracts templates, records the first data point at t=0, and returns an annotated preview frame. Returns `None` and sets `self.error` if the test type is unrecognized or no dots are detected.

**`step()`:**  
Processes one step (skipping `frame_skip − 1` frames):
1. Tries each dot's rolling template → falls back to its reference template if score < 0.25
2. Rejects the whole frame if any dot makes a jump > 80% of `search_radius` (treated as a false match)
3. If template tracking has failed for ≥2 consecutive frames, runs the full initial detector (`_try_redetect`) to re-acquire dots after a large jump; on success, resets templates and counters
4. On success: refines all centroids, updates rolling templates every 15 frames, appends result
5. On failure: repeats the last known distance and positions; terminates after 60 consecutive failures

**Template rolling:** Updated every 15 frames to track gradual appearance changes (dot elongation, fading). The reference template is kept as a fallback to recover from brief tracking loss.

**`_dot_label(i)`:** Returns a human-readable label for dot index `i`: `'green1'`/`'yellow2'` for hinge_colored, `'set1_1'`/`'set2_4'` for hinge and hinge_tension (1-indexed within group), `'dot1'`/`'dot2'` otherwise.

---

### `annotate_frame(frame, dots, dist_val, unit)`

Draws tracking overlays on a frame for display:
- Blue crosshairs (±18px) and circles (r=12) at each dot position in `dots`
- Cyan line connecting the dots (only when exactly 2 dots)
- Distance label in mm or px (only when exactly 2 dots and `dist_val` is not `None`)

All drawn on an overlay copy, then blended at 50% opacity using `cv2.addWeighted` so the dot remains visible under the crosshair.

---

## app.py

### Toolbar controls

- **Add Videos… / Add from input_videos/** — load video files into the list
- **Clear List** — reset the video list and all stored ROIs/results
- **Set ROIs…** — enabled only when the selected video is `hinge_tension` type and not currently processing; opens an interactive first-frame window to define two rotated-strip ROIs (see below)
- **Frame skip** — dropdown: Every frame / Every 2nd frame / Every 4th frame
- **Run All** — process all listed videos in a background thread
- **Run Selected** — process only the currently highlighted video in a background thread
- **Stop** — signal the worker to stop after the current video

**Output variable checkboxes** (row below toolbar): select which columns appear in the exported CSV — pixel position (x, y), scaled mm position (x, y), per-dot displacement, inter-dot displacement, inter-dot distance (mm).

### Rotated-strip ROI selection (`_set_rois`)

Opens a scaled-down copy of the first frame in an OpenCV window. For each of two ROIs the user left-clicks two points defining the strip centerline; the strip extends `ROI_HALF_WIDTH = 25` native pixels on each side perpendicular to that line. Controls: left-click to place a point, **R** to redo the current ROI, **ENTER** to confirm and advance, **ESC** to cancel.

Confirmed ROIs are stored in `self.test_rois[vid_idx]` as a tuple of `((pt1, pt2, half_width), ...)`. The listbox label gains a `[ROI✓]` prefix. A ROI overlay is drawn on the first-frame preview whenever the video is selected. The ROIs are passed to `VideoTracker(rois=...)` at run time.

### Threading model

Processing runs in a background `daemon` thread (`_worker`). The worker sends messages to the UI via a `queue.Queue`. The UI polls the queue every 30ms using `tk.after`. This prevents the GUI from freezing during processing.

Message types:
- `MsgProgress(vid_idx, frame_idx, total_frames, frame_bgr)` — progress update; frame sent every 20 processed frames to limit UI load
- `MsgDone(vid_idx, tracker)` — video finished; full tracker object with all results
- `MsgError(vid_idx, error_msg)` — detection or file error
- `MsgAllDone` — all videos finished

### Video review

Completed videos can be scrubbed frame-by-frame. A `cv2.VideoCapture` is held open for the currently-reviewed video. The scrub bar (`ttk.Scale`) calls `_show_review_frame(frame_idx)` which:
1. Seeks to the requested frame with `cv2.CAP_PROP_POS_FRAMES`
2. Finds the closest tracked position using `np.searchsorted` on the stored `frame_indices`
3. Calls `annotate_frame` with that position and re-renders

A `_scrub_blocked` flag prevents the scrub callback from firing during `configure(to=...)` calls (which would otherwise seek to frame 0 spuriously).

### Outlier cleaning (`clean_data`)

Two-pass MAD-based filter:

**Pass 1 — Velocity outliers:**  
Computes frame-to-frame velocity `dd/dt`. In a rolling window of 51 frames, computes the median velocity and MAD (Median Absolute Deviation). Points where velocity deviates by more than 5× MAD are removed.

**Pass 2 — Position outliers:**  
On the velocity-filtered data, computes a rolling window median of the distance values. Points deviating more than 5× MAD from the local median are removed.

MAD is preferred over standard deviation because it is robust to the outliers being removed — a single large spike does not inflate the threshold.

### Pixel-to-mm calibration

On `VideoTracker.open()`, the pixel distance between the two detected dot centroids is computed. If the filename contains an initial distance in mm (e.g., `49.9mm`), then:

```
px_per_mm = initial_pixel_distance / initial_distance_mm
```

All subsequent distances are divided by `px_per_mm`. If no distance is found in the filename, distances are reported in pixels.

---

## validation/build_comparison.py

Generates `Tracker vs openCV Comparison.xlsx` comparing manual Tracker measurements against OpenCV output.

**Process:**
1. Loads manual Tracker CSV (2 header rows, columns: time, length in meters)
2. Loads OpenCV CSV (time, pixel distance)
3. Calibrates OpenCV pixel values to mm using `tracker_mm[0] / opencv_px[0]`
4. Time-matches: for each Tracker timestamp, finds the nearest OpenCV timestamp
5. Computes % difference column
6. Writes a formatted table, statistics panel (RMSE, R², correlation, mean/median/max % difference), and two embedded Excel charts (distance overlay + % difference over time)
