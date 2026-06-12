# Video Trimming / Pre-processing Script

`edit_video.py` batch-processes raw specimen videos before they're used elsewhere (e.g. with the dot tracker in `app.py`). For every video, it produces an edited **copy** — originals are never modified or deleted.

## What it does

For each `.mp4` video, the script:

1. **Removes audio** — the output has no audio track. (This happens automatically: the script reads/writes video frames only with OpenCV, which never carries audio.)
2. **Crops** every frame to the same pixel box, so all videos in a batch line up consistently (useful when videos are shot with a tripod of same-sized objects).
3. **Increases color saturation** by a fixed multiplier.
4. **Saves the result** as a new file in an `Edited/` subfolder, named the same as the original.

## Folder layout

```
input_videos/2026-06-11 Videos/
├── video1.mp4
├── video2.mp4
├── ...
└── Edited/              <- created automatically
    ├── video1.mp4
    ├── video2.mp4
    └── ...
```

`INPUT_DIR` (top of `edit_video.py`) points at the folder of raw videos; `OUTPUT_DIR` is always `Edited/` inside it.

## How to run it

```
.venv/bin/python edit_video.py
```

The script processes **every `.mp4` file** found directly inside `INPUT_DIR` (not recursively — the `Edited/` subfolder is skipped automatically). For each video it prints progress:

```
[1/12] Processing: video1.mp4 (11 remaining after this)
  -> saved to .../Edited/video1.mp4 (20.1s)
...
Done. Processed 12 video(s).
```

## Choosing the crop box

The crop box is a single pixel rectangle `(x, y, width, height)` applied to **every video in the batch**, so it must be consistent across videos (same camera position/frame size).

- **First run**: leave `CROP_BOX = None` at the top of the file. The script will read the first frame of the first video and ask you to either:
  - **Drag** a box on the image in a popup window (press ENTER/SPACE to confirm, ESC to cancel), or
  - **Type** the coordinates (`x`, `y`, `width`, `height`) at the prompt.
- After you choose, the script prints the box in a ready-to-paste format, e.g.:
  ```
  CROP_BOX = (150, 190, 130, 164)  # (x, y, width, height)
  ```
- **Subsequent runs**: paste that line over `CROP_BOX = None` to reuse the same crop without being prompted again.

Note: width/height are automatically rounded down to even numbers, since the video codec used (`mp4v`) requires even dimensions — the printed value already reflects this.

## Tuning constants

All adjustable settings live near the top of `edit_video.py`:

| Constant | Purpose |
|---|---|
| `SATURATION_SCALE` | Multiplier applied to the HSV saturation channel (e.g. `1.8` = 80% more saturated). |
| `INPUT_DIR` | Folder containing the raw `.mp4` videos. |
| `OUTPUT_DIR` | Where edited copies are written (`Edited/` inside `INPUT_DIR`). |
| `CROP_BOX` | `(x, y, width, height)` pixel crop applied to all videos, or `None` to select interactively. |

## Notes / limitations

- Only `.mp4` files are picked up.
- Output is re-encoded with the `mp4v` codec — there's no control over output compression/quality beyond what OpenCV's default `mp4v` writer provides.
- The crop box is applied in absolute pixels, so all input videos should share the same frame dimensions.
- Processing is roughly real-time-ish per video (~20s for a ~30s clip in testing); for 100+ videos, expect the full batch to take on the order of 30+ minutes.
