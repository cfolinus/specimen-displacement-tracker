# Implementation Plan — Multi-Dot "Hinge" Tracking

**Status: implemented and validated.** This doc reflects the final, as-built
implementation in `tracker_core.py` / `app.py` (two details changed from the
original draft during validation — see §2/§2a/§3 notes below).

Engineer-level detail for adding a reusable color-dot (`hinge`) test type that
tracks four colored dots (2 green, 2 yellow) independently, labeled by color.

Target video for validation: `input_videos/Hinge - Trimmed sample video - imovie.mp4`
(434×550, ~60 fps, 1798 frames). Full run: 1798/1798 frames tracked, 0 redetections,
0 lost-tracking events.

---

## 0. Background: how detection/tracking is wired today

- `detect_test_type(filename)` → `'tensile'` | `'roller'` (string), from filename keywords.
- `find_initial_dots(frame_bgr, test_type)` dispatches to:
  - `find_initial_dots_tensile(gray)` — dark blobs on bright specimen.
  - `find_initial_dots_roller(bgr)` — bright magenta dots (HSV).
  - Returns a list of `(x, y)` positions, bottom-to-top, or `None`.
- `refine_centroid(frame_bgr, gray, pos, test_type, patch_size)` dispatches to
  `refine_centroid_dark` / `refine_centroid_bright`.
- `VideoTracker` stores `self.dots` (list of `(x,y)`), `self.n_dots`, parallel
  history in `self.positions`, and tracks each dot independently via grayscale
  template matching (`track_dot_template`) + redetect fallback (`_try_redetect`).
  This is already N-dot capable (1–4). Inter-dot distance is computed **only when
  `n_dots == 2`**, so a 4-dot video naturally produces positions-only output.
- `save_csv` builds headers/rows with generic `dot{i+1}` labels.
- `app.py` builds the data-table columns (`_show_data_table`), plot variable list
  (`_variable_labels`), and plot value extraction (`_compute_var`) with `dot{i}` labels.

We will keep the tracking loop untouched and add: a new detector, a new refinement,
a `colors` list on the tracker, and color-aware labels in CSV + UI.

---

## 1. `tracker_core.py` — `detect_test_type`

A keyword is now **mandatory**: `detect_test_type` returns `None` when the
filename matches none of `tensile`/`roller`/`hinge`, and `VideoTracker.open()`
refuses a `None`-typed video (see §6a). There is no silent tensile default — a
misnamed file is skipped rather than run through the wrong detector.

```python
VALID_TEST_TYPES = ('tensile', 'roller', 'hinge')

def detect_test_type(filename):
    name = Path(filename).stem.lower()
    if 'roller' in name:
        return 'roller'
    if 'hinge' in name:
        return 'hinge'
    if 'tensile' in name:
        return 'tensile'
    return None
```

> **Migration note:** existing tensile videos whose names lack the word
> `tensile` (e.g. `Instron - side - 1 49.9mm.MOV`) must be renamed to include it,
> or passed an explicit `test_type='tensile'` to `VideoTracker`.

---

## 2. `tracker_core.py` — `find_initial_dots_color(bgr, max_per_color=2)`

New detector. Returns a list of `((x, y), color)` tuples — **4 entries**:
green dots first (top-to-bottom), then yellow dots (top-to-bottom). Returns `None`
if it can't find at least one dot of either color.

> NOTE: this detector returns `(pos, color)` tuples, unlike the existing detectors
> which return bare `(x, y)`. The dispatch layer (§4) and `VideoTracker.open` (§6)
> handle splitting these into `self.dots` + `self.colors`.

### HSV thresholds (final, as-implemented)

```python
# Green: bright, saturated
GREEN_H = (45, 85);  GREEN_S = 70;  GREEN_V = 70
# Yellow: dimmer, yellow-green; hue sits just below green
YELLOW_H = (33, 44); YELLOW_S = 100; YELLOW_V = 110
```

The split between yellow and green is **H≈44**. Green dots are brighter (higher V),
which is a useful tiebreaker if a blob lands near the boundary.

> **Why YELLOW_S/V were raised from the initial draft (50/60 → 100/110):** a
> metal screw head in the sample frame has a brownish-yellow specular reflection
> at HSV ≈ (20, 88, 92) — low hue, low saturation, low value relative to the real
> yellow dots (H≈37–38, S≈122–148, V≈130–138). The looser starting thresholds
> picked up the screw head as a "yellow" blob with a larger area than one of the
> real dots, displacing it from the top-2 selection. Raising `YELLOW_S` to 100 and
> `YELLOW_V` to 110 — both well below the real dots' values but above the screw's —
> excludes it while keeping comfortable margin for the real dots. If future videos
> show yellow dots dimmer than this, narrow `YELLOW_H` further (e.g. to 35–42)
> before lowering `YELLOW_S`/`YELLOW_V`, to avoid re-admitting screw-like artifacts.

### Per-color blob extraction (factor into a helper)

```python
def _find_color_dots(hsv, h_lo, h_hi, s_min, v_min, max_dots):
    H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    mask = ((H >= h_lo) & (H <= h_hi) & (S >= s_min) & (V >= v_min)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)

    blobs = []  # (cx, cy, area)
    for i in range(1, n):
        a  = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if a < 12 or a > 1500:           # area gate; tune on real data
            continue
        aspect = bw / max(bh, 1)
        if aspect < 0.3 or aspect > 3.3:  # reject long streaks
            continue
        cx, cy = cent[i]
        blobs.append([float(cx), float(cy), int(a), i])

    # Handle the merged-green case: if we found fewer than max_dots but one blob
    # is oversized, erode it until it splits into two (see §2a).
    blobs = _maybe_split_blobs(blobs, lab, max_dots)

    blobs.sort(key=lambda b: b[2], reverse=True)   # largest/most-confident first
    blobs = blobs[:max_dots]
    blobs.sort(key=lambda b: b[1])                 # then order top-to-bottom (y asc)
    return [(b[0], b[1]) for b in blobs]
```

### 2a. Splitting a merged blob (the green pair) — final approach: erosion

The original draft proposed a distance-transform peak split, but on the real
merged-green blob (area≈498, bbox 42×36) the two strongest distance-transform
peaks landed only ~8px apart — both near the blob's "waist" rather than at each
dot's center. **Erosion gives a clean split instead:**

```python
def _maybe_split_blobs(blobs, lab, max_dots):
    if len(blobs) >= max_dots or len(blobs) == 0:
        return blobs

    blobs_sorted = sorted(blobs, key=lambda b: b[2], reverse=True)
    largest = blobs_sorted[0]
    area, label_id = largest[2], largest[3]
    if area < 250:
        return blobs

    mask = (lab == label_id).astype(np.uint8)
    for k in range(1, 5):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        eroded = cv2.erode(mask, kernel)
        n_e, e_lab, e_stats, e_cent = cv2.connectedComponentsWithStats(eroded)
        valid = [i for i in range(1, n_e) if e_stats[i, cv2.CC_STAT_AREA] >= 2]
        if len(valid) == 2:
            new_blobs = [b for b in blobs if b[3] != label_id]
            for i in valid:
                cx, cy = e_cent[i]
                new_blobs.append([float(cx), float(cy), area // 2, label_id])
            return new_blobs
        if len(valid) > 2:
            break

    return blobs
```

Erode with a growing elliptical kernel (3×3, 5×5, 7×7, 9×9) until the mask
separates into exactly 2 components, then use those components' centroids
directly — no distance transform needed. On the sample frame, a 3×3 erosion
(k=1) was sufficient and gave centroids ~29px apart (vs. ~8px for the
distance-transform peaks), matching the true dot separation.

If splitting still yields ≠2 components at any kernel size up to 9×9, return
`blobs` unchanged (the caller tolerates <max_dots per color, but the whole
detector needs ≥1 of each color).

### Assemble result

```python
def find_initial_dots_color(bgr, max_per_color=2):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    greens  = _find_color_dots(hsv, *GREEN_H,  GREEN_S,  GREEN_V,  max_per_color)
    yellows = _find_color_dots(hsv, *YELLOW_H, YELLOW_S, YELLOW_V, max_per_color)
    if not greens and not yellows:
        return None
    out  = [(p, 'green')  for p in greens]
    out += [(p, 'yellow') for p in yellows]
    return out  # list of ((x, y), color)
```

---

## 3. `tracker_core.py` — `refine_centroid_color(bgr, pos, color, patch_size=30)`

Mirror `refine_centroid_bright`, but restrict the saturation-weighted centroid to
the dot's own hue band:

```python
def refine_centroid_color(bgr, pos, color, patch_size=30):
    h, w = bgr.shape[:2]
    half = patch_size
    x, y = int(pos[0]), int(pos[1])
    x1, y1 = max(0, x - half), max(0, y - half)
    x2, y2 = min(w, x + half), min(h, y + half)
    patch = bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return pos
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]; S = hsv[:, :, 1].astype(float)
    if color == 'green':
        hue_ok = ((H >= GREEN_H[0]) & (H <= GREEN_H[1])).astype(float)
    else:
        hue_ok = ((H >= YELLOW_H[0]) & (H <= YELLOW_H[1])).astype(float)
    weights = S * hue_ok
    if weights.sum() < 30:
        return pos
    yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    cx = (xx * weights).sum() / weights.sum() + x1
    cy = (yy * weights).sum() / weights.sum() + y1
    return (cx, cy)
```

> **Patch size caveat (discovered during validation):** the hinge dots sit only
> ~29px apart (green pair) to ~38px apart (yellow pair). With the default
> `patch_size=30` (→ a 60×60px window), the saturation-weighted centroid for one
> green dot included most of its neighbor's pixels too, pulling both refined
> green positions to nearly the same point (~3px apart instead of ~29px). The
> dispatcher (§4) caps `patch_size` at **10** for hinge mode regardless of the
> caller's request, giving a 20×20px window — comfortably larger than a single
> ~8–10px dot but well short of the neighbor.

---

## 4. `tracker_core.py` — dispatch updates

`find_initial_dots` currently returns bare positions. For `hinge` we need colors too.
Two options; pick **(A)** to keep the dispatch signature simple:

**(A) Keep `find_initial_dots` returning positions; expose colors separately.**
Add a sibling that returns the labeled list, and have `VideoTracker.open` call the
color detector directly when `test_type == 'hinge'`. Minimal blast radius:

```python
def find_initial_dots(frame_bgr, test_type='tensile'):
    if test_type == 'roller':
        return find_initial_dots_roller(frame_bgr)
    if test_type == 'hinge':
        labeled = find_initial_dots_color(frame_bgr)
        return [p for p, _c in labeled] if labeled else None
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return find_initial_dots_tensile(gray)
```

`refine_centroid` needs the per-dot color, so extend its signature with an optional
`color` argument used only in hinge mode:

```python
def refine_centroid(frame_bgr, gray, pos, test_type='tensile', patch_size=30, color=None):
    if test_type == 'roller':
        return refine_centroid_bright(frame_bgr, pos, patch_size)
    if test_type == 'hinge':
        # Cap the patch so the centroid doesn't bleed into a neighboring
        # dot of the same color (~29px away); see §3 caveat.
        return refine_centroid_color(frame_bgr, pos, color or 'green', min(patch_size, 10))
    return refine_centroid_dark(gray, pos, patch_size)
```

> `_try_redetect` (re-acquisition) also calls `find_initial_dots` + `refine_centroid`.
> In hinge mode it must preserve color identity. Simplest: re-run
> `find_initial_dots_color`, then match re-detected dots to existing dots **within
> the same color** by nearest-neighbor (don't let a green re-lock onto a yellow).
> See §6 for the redetect note.

---

## 5. `tracker_core.py` — `annotate_frame` (already partly done)

- Overlay color already changed to **bright blue** `(255, 0, 0)` (BGR). No further
  change required for the crosshairs.
- The inter-dot connecting line + distance label are gated to `len(dots) == 2`, so
  they won't render for 4 dots. Leave as-is.

---

## 6. `tracker_core.py` — `VideoTracker`

### `__init__`
Add: `self.colors = []  # parallel to self.dots: e.g. ['green','green','yellow','yellow']`

### `open()`
First, **refuse a video with no recognized test type** (see §1) before any frame
I/O, so a misnamed file fails fast with an actionable message instead of being
mistracked:

```python
if self.test_type is None:
    self.error = (
        f"Could not determine test type for '{self.video_path.name}'. "
        f"Include one of these keywords in the filename: "
        f"{', '.join(VALID_TEST_TYPES)}."
    )
    self.finished = True
    return None
```

This surfaces through the existing error path in `app.py._worker` (the video is
marked ✗ and the message is printed; it is **not** processed).

After `find_initial_dots`, also capture colors. Since we chose dispatch option (A),
call the labeled detector directly in hinge mode so we get colors without changing
the generic return type:

```python
if self.test_type == 'hinge':
    labeled = find_initial_dots_color(frame)
    if not labeled:
        self.error = "Could not detect any colored dots (hinge)"
        self.finished = True
        return None
    detected     = [p for p, _c in labeled]
    self.colors  = [c for _p, c in labeled]
else:
    detected = find_initial_dots(frame, self.test_type)
    self.colors = [None] * (len(detected) if detected else 0)
    ...existing None/empty checks...

self.dots = [refine_centroid(frame, gray, d, self.test_type, color=c)
             for d, c in zip(detected, self.colors)]
```

`self.n_dots = len(self.dots)` (will be up to 4). Calibration/inter-dot block stays
gated on `n_dots == 2`, so nothing happens for hinge — good.

### `step()`
Where it refines tracked positions, pass the matching color:

```python
refined = [refine_centroid(frame, gray, p, self.test_type, 24, color=c)
           for p, c in zip(new_positions, self.colors)]
```

(Do the same in the redetect branch.)

### `_try_redetect(frame, gray)`
In hinge mode, re-detect with `find_initial_dots_color`, then match **per color** to
existing dots by nearest-neighbor so identities don't swap. Keep the existing
distance gate (`max_d`). If a color can't be re-matched, return `None` (treated as a
miss, consistent with current behavior). For non-hinge types, leave the function
unchanged.

### `save_csv`
Replace the generic `dot{i+1}` label with a color-aware label. Add a helper:

```python
def _dot_label(self, i):
    if self.colors and i < len(self.colors) and self.colors[i]:
        # number within color: 1st/2nd green, 1st/2nd yellow
        color = self.colors[i]
        nth = sum(1 for j in range(i + 1) if self.colors[j] == color)
        return f'{color}{nth}'
    return f'dot{i+1}'
```

Use `self._dot_label(i)` everywhere `save_csv` currently builds `f'dot{i+1}'`
(header construction only — row logic is index-based and unchanged). Inter-dot
columns remain gated to `n == 2`.

---

## 7. `app.py` — UI labels

Mirror the same color-aware labeling. Add a small helper near the top of `App` (or a
module function) that derives a label from a tracker + dot index, reusing
`tracker._dot_label(i)` so logic lives in one place:

- **`_show_data_table`**: replace `f'Dot{i+1} X (px)'` etc. with
  `f'{tracker._dot_label(i)} X (px)'` (and Y, mm, dX, dY variants). Row-building loops
  are index-based — no change beyond headers.
- **`_variable_labels`**: build labels from `tracker._dot_label(i-1)` instead of
  `f'Dot{i}'`. Keep the inter-dot entries gated to `n == 2` (won't appear for hinge).
- **`_compute_var`**: this **parses** the label to recover the dot index
  (`label.startswith('Dot')` … `int(label[3:space])`). Update the parser to handle
  color labels. Cleanest: instead of parsing the human label, map the selected label
  back to a dot index via the same `_dot_label` list built once per tracker, then
  reuse the existing X/Y/dX/dY math. Avoid brittle string parsing of `green1` etc.

> Watch-outs in `_compute_var`: the current code keys off the literal prefix `'Dot'`
> and the inter-dot prefixes. With color labels, rewrite the dispatch to:
> 1) if label is `'Time (s)'` → time; 2) if it matches an inter-dot label → inter-dot
> (only for n==2); 3) else find the dot index whose `_dot_label(i)` is a prefix of the
> label, then branch on the suffix (`X (px)`, `Y (px)`, `X (mm)`, `Y (mm)`, `dX …`,
> `dY …`). Keep the suffixes identical to today so the math is unchanged.

---

## 8. Validation steps — all completed ✓

1. **Headless detection check:** loaded the first frame, called
   `find_initial_dots_color` — returned 4 dots: `green` at (316.3, 287.0) and
   (339.1, 306.8), `yellow` at (343.4, 361.0) and (313.2, 384.9), after centroid
   refinement (post the YELLOW threshold tightening and erosion split).
2. **Annotated frame:** `VideoTracker.open()` → four **blue** crosshairs land
   cleanly on the four dots (saved to a temp PNG for review).
3. **Full run (headless, equivalent to GUI Run All):** 1798/1798 frames processed,
   **0 redetections, 0 lost-tracking events**. Sampled frames at t≈5s, 15s, 25s,
   30s show crosshairs holding through a ~90° hinge swing.
4. **CSV export:** headers read
   `time_s,green1_x_px,green1_y_px,green2_x_px,green2_y_px,yellow1_x_px,yellow1_y_px,yellow2_x_px,yellow2_y_px,...`
   — `green1/green2/yellow1/yellow2` confirmed via `tracker._dot_label(i)`.
   mm columns present but empty (no `mm` in filename, as expected); no
   inter-dot/distance columns (n=4).
5. **Plot/table labels:** `_variable_labels` → `['Time (s)', 'Green1 X (px)',
   'Green1 Y (px)', ..., 'Yellow2 dY (px)']`; `_default_y_label` →
   `'Green1 dY (px)'`; `_compute_var` correctly resolves each color label back
   to its dot index and computes X/Y/dX/dY.
6. **Regression:** `detect_test_type` still returns `'tensile'` for
   `Instron - side - 1 49.9mm.MOV` and `Four dots - two colors.MOV`, `'roller'`
   for `Roller.MP4`; `_dot_label(i).capitalize()` still returns `'Dot1'`/`'Dot2'`
   when `colors` are `None` (non-hinge), so existing CSV/plot labels are
   byte-for-byte unchanged.

---

## 9. Tuning notes / risks for the 100+ video rollout

- **Yellow thresholds are now tighter than first drafted** (`H 33–44, S≥100,
  V≥110`) specifically to exclude a screw-head reflection at HSV≈(20,88,92). If
  a future video's yellow dots are dimmer than (37–38, 122–148, 130–138), don't
  just lower S/V — check whether that re-admits screw-like hardware first.
  Narrowing `YELLOW_H` (e.g. 35–42) is safer than lowering `YELLOW_S`/`YELLOW_V`.
- **Green-pair merge + split via erosion** assumes the two green dots, when
  merged, form one blob ≥250px that separates into exactly 2 components within
  4 erosion steps (3×3 → 9×9 ellipse kernels). If a future video's dots are
  larger/closer (merge into a blob that doesn't cleanly bisect), this may need a
  larger kernel range or a different split strategy.
- **Refinement patch is capped at 10px** for hinge mode (§3/§4) because dots are
  only ~29px apart. If future videos have dots spaced very differently, this
  cap may need to become test-type- or video-specific rather than a fixed `10`.
- **Identity stability across frames** relies on per-color nearest-neighbor
  matching in `_try_redetect`. Validated with 0 redetections needed on the
  sample video — re-verify if a future video has fast motion that triggers
  redetects.
- Thresholds and the patch-size cap are centralized as module constants
  (`GREEN_H`, `YELLOW_H`, etc., and the `min(patch_size, 10)` in
  `refine_centroid`) so they can be tuned once for all 100+ videos.
