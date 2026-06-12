# Handover — Multi-Dot "Hinge" Tracking

**Status: implemented and validated** on the sample video (1798/1798 frames tracked,
0 redetections, 0 lost-tracking events).

## Goal
Make the tracker recognize and track **four colored dots** (2 green, 2 yellow) on a
metal hinge mechanism, e.g. `input_videos/Hinge - Trimmed sample video - imovie.mp4`.

## Why it doesn't work today
The software only has two detectors:
- **tensile** (the default) — looks for *dark* Sharpie dots on a *bright* specimen.
- **roller** — looks for *bright magenta* paint dots.

The hinge video has no "roller" or "mm" in its filename, so it falls back to
**tensile**, which hunts for dark blobs. It locks onto dark hinge hardware (screws,
pivots) and completely ignores the bright green/yellow dots. There is **no
color-dot detector at all** — that's the gap we're filling.

> **Update:** the silent tensile default has since been removed. A test-type
> keyword (`tensile`, `roller`, or `hinge`) is now **mandatory** in the filename;
> a file with none is skipped (marked ✗) with a message listing the valid types,
> rather than being mistracked as tensile. Existing tensile videos must now
> include `tensile` in their name (or be passed `test_type='tensile'`).

## What we're adding
A new, reusable **`hinge`** test type, auto-selected for any filename containing
"hinge". It detects the four colored dots by color (HSV), tracks them
independently, and labels them by color.

## Key decisions (confirmed with stakeholder)
1. **Track all four dots independently** — no distances between dots, just positions.
2. **Label dots by color**: `green1, green2, yellow1, yellow2` (ordered top-to-bottom
   within each color). Green/yellow assignment is consistent across videos.
3. **Pixel output only** — no mm calibration for this test type.
4. **Reusable mode** — will run on 100+ similarly-colored "hinge" videos.

## What the source video looks like
- 434×550, ~60 fps, 1798 frames.
- Green dots: bright, well-saturated (HSV H≈51, S≈152, V≈130).
- Yellow dots: **dim and small**, yellow-green (HSV H≈37–38, S≈122–148, V≈130–138).
- Gotchas the detector handles:
  - The two green dots sit close together (~29px) and **merge into one blob** —
    fixed by eroding the merged blob until it splits into two components and
    using their centroids.
  - A metal screw head has a brownish-yellow reflection (HSV H≈20, S≈88, V≈92)
    that falls inside a naive "yellow" range — excluded by tightening the
    yellow thresholds (final: `H 33–44, S≥100, V≥110`) well above the screw's
    values and below the real yellow dots'.
  - The refinement patch had to be capped at 10px (vs. the default 30px) for
    hinge mode, otherwise the saturation-weighted centroid for one green dot
    bled into its neighbor ~29px away and pulled both dots toward the same point.

## The blue crosshairs
Tracking overlay color was changed from green to **bright blue** so it stands out
against the green dots during troubleshooting. (Already done in `tracker_core.py`,
`annotate_frame`.)

## Files affected
- `tracker_core.py` — new detector, refinement, dispatch, per-dot color labels, CSV headers.
- `app.py` — data-table columns, plot variable labels.

## How to verify
Run `python app.py`, add the hinge video, Run All. Expect **four blue crosshairs**
on the four dots, holding through the hinge motion. CSV columns should read
`green1_x_px, green1_y_px, green2_…, yellow1_…, yellow2_…`.

See **IMPLEMENTATION PLAN - multidots.md** for engineer-level detail.
