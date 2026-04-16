"""
Core dot-tracking logic for Instron tensile test videos.
Detects two dark Sharpie dots on a light specimen and tracks
their separation frame-by-frame using template matching.
"""

import cv2
import numpy as np
import csv
import re
from pathlib import Path


def extract_initial_distance_mm(filename):
    """Extract initial dot distance from filename, e.g. '25.4mm' -> 25.4"""
    match = re.search(r'([\d.]+)\s*mm', filename, re.IGNORECASE)
    return float(match.group(1)) if match else None


def find_specimen_region(gray):
    """
    Detect the specimen (light-colored dogbone) by finding the brightest gap
    between dark horizontal jaw bands. The jaws are large dark horizontal
    structures; the specimen is the bright material between them.

    Returns (y_min, y_max) of the specimen region, or None.
    """
    h, w = gray.shape

    # Count dark pixels per row — jaw rows have many dark pixels spanning the frame
    dark_count = (gray < 60).sum(axis=1).astype(float)
    kernel = np.ones(10) / 10
    dark_smooth = np.convolve(dark_count, kernel, mode='same')

    # Jaw rows: >30% of the row width is dark
    is_jaw = dark_smooth > w * 0.3

    # Find contiguous jaw bands
    bands = []
    in_band = False
    start = 0
    for i in range(h):
        if is_jaw[i] and not in_band:
            start = i
            in_band = True
        elif not is_jaw[i] and in_band:
            bands.append((start, i))
            in_band = False
    if in_band:
        bands.append((start, h))

    # Keep bands taller than 30px
    bands = [(s, e) for s, e in bands if e - s > 30]

    if len(bands) < 2:
        return None

    # The specimen is the brightest gap between adjacent jaw bands
    best = None
    best_brightness = 0
    cx = w // 2
    for i in range(len(bands) - 1):
        gap_top = bands[i][1]
        gap_bot = bands[i + 1][0]
        if gap_bot - gap_top < 80:
            continue
        # Measure brightness in center strip of this gap
        x1 = max(0, cx - 80)
        x2 = min(w, cx + 80)
        gap_region = gray[gap_top:gap_bot, x1:x2]
        brightness = gap_region.mean()
        if brightness > best_brightness:
            best_brightness = brightness
            best = (gap_top, gap_bot)

    return best


def find_initial_dots(gray):
    """
    Find exactly 2 dark Sharpie dots on the light specimen in the first frame.

    Step 1: Detect the specimen region (between the Instron jaws).
    Step 2: Search for small dark blobs with high contrast only within that region.

    Returns [(x1,y1), (x2,y2)] sorted top-to-bottom, or None.
    """
    h, w = gray.shape

    # Determine search bounds — restrict to specimen if detected
    spec = find_specimen_region(gray)
    if spec is not None:
        search_y_min, search_y_max = spec
        # Add a small margin
        search_y_min = max(0, search_y_min - 20)
        search_y_max = min(h, search_y_max + 20)
    else:
        search_y_min, search_y_max = 0, h

    x_margin = w // 4
    search_region = gray[search_y_min:search_y_max, x_margin:w - x_margin]
    offset_x = x_margin
    offset_y = search_y_min

    all_candidates = []

    for thresh_val in range(90, 175, 5):
        _, thresh = cv2.threshold(search_region, thresh_val, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area < 15 or area > 500:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            aspect = bw / max(bh, 1)
            if aspect < 0.15 or aspect > 6.0:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"] + offset_x
            cy = M["m01"] / M["m00"] + offset_y

            ix, iy = int(cx), int(cy)
            inner_r, outer_r = 8, 25
            y1 = max(0, iy - outer_r)
            y2 = min(h, iy + outer_r)
            x1 = max(0, ix - outer_r)
            x2 = min(w, ix + outer_r)
            patch = gray[y1:y2, x1:x2]

            yy, xx = np.ogrid[-(iy - y1):(y2 - iy), -(ix - x1):(x2 - ix)]
            dist = np.sqrt(xx.astype(float)**2 + yy.astype(float)**2)
            annulus_mask = (dist > inner_r) & (dist <= outer_r)

            if annulus_mask.sum() == 0:
                continue

            surround_mean = patch[annulus_mask].mean()
            center_mean = gray[max(0, iy - 3):iy + 4, max(0, ix - 3):ix + 4].mean()
            contrast = surround_mean - center_mean

            if surround_mean > 140 and contrast > 40:
                all_candidates.append((cx, cy, area, contrast, surround_mean))

    if len(all_candidates) < 2:
        return None

    # Cluster candidates within 15px (same dot at multiple thresholds)
    clusters = []
    used = set()
    for i, c1 in enumerate(all_candidates):
        if i in used:
            continue
        cluster = [c1]
        used.add(i)
        for j, c2 in enumerate(all_candidates):
            if j in used:
                continue
            if np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2) < 15:
                cluster.append(c2)
                used.add(j)
        avg_x = np.mean([c[0] for c in cluster])
        avg_y = np.mean([c[1] for c in cluster])
        max_contrast = max(c[3] for c in cluster)
        avg_surround = np.mean([c[4] for c in cluster])
        n_det = len(cluster)
        score = max_contrast * n_det
        clusters.append((avg_x, avg_y, max_contrast, avg_surround, n_det, score))

    if len(clusters) < 2:
        return None

    clusters.sort(key=lambda c: c[5], reverse=True)
    best_pair = None
    best_pair_score = 0

    for i in range(min(len(clusters), 8)):
        for j in range(i + 1, min(len(clusters), 8)):
            dy = abs(clusters[i][1] - clusters[j][1])
            if dy < 50:
                continue
            pair_score = clusters[i][5] + clusters[j][5]
            if pair_score > best_pair_score:
                best_pair_score = pair_score
                c1, c2 = clusters[i], clusters[j]
                if c1[1] < c2[1]:
                    best_pair = [(c1[0], c1[1]), (c2[0], c2[1])]
                else:
                    best_pair = [(c2[0], c2[1]), (c1[0], c1[1])]

    return best_pair


def track_dot_template(gray, template, last_pos, search_radius=60):
    """Track a single dot via normalised cross-correlation."""
    h, w = gray.shape
    th, tw = template.shape

    x1 = max(0, int(last_pos[0] - search_radius - tw // 2))
    y1 = max(0, int(last_pos[1] - search_radius - th // 2))
    x2 = min(w, int(last_pos[0] + search_radius + tw // 2))
    y2 = min(h, int(last_pos[1] + search_radius + th // 2))

    search_area = gray[y1:y2, x1:x2]
    if search_area.shape[0] < th or search_area.shape[1] < tw:
        return None, 0.0

    result = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < 0.25:
        return None, max_val

    cx = x1 + max_loc[0] + tw // 2
    cy = y1 + max_loc[1] + th // 2
    return (cx, cy), max_val


def refine_centroid(gray, pos, patch_size=30):
    """
    Snap to the centroid of the actual dark blob nearest to `pos`.

    Uses local adaptive contrast (local_mean - pixel) to find the dark region
    regardless of how the dot's shape has changed due to stretching. This is
    robust to elongation, fading, and background brightness variation.
    """
    h, w = gray.shape
    half = patch_size
    x, y = int(pos[0]), int(pos[1])
    x1, y1 = max(0, x - half), max(0, y - half)
    x2, y2 = min(w, x + half), min(h, y + half)

    patch = gray[y1:y2, x1:x2].astype(np.float64)
    if patch.size == 0:
        return pos

    # Compute local background: smooth heavily to get the "expected" brightness
    # without the dot, then subtract to find the dark anomaly
    local_bg = cv2.GaussianBlur(patch, (0, 0), sigmaX=12)
    contrast_map = local_bg - patch  # positive where pixel is darker than surroundings

    # Threshold: keep only the clearly-dark region
    peak = contrast_map.max()
    if peak < 10:
        return pos  # no clear dark spot, keep template position

    mask = contrast_map > peak * 0.35

    # Find connected components — pick the one closest to center
    mask_u8 = mask.astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8)

    if n_labels < 2:
        return pos

    # Pick the component closest to the center of the patch (where template said the dot is)
    pcx, pcy = patch.shape[1] / 2, patch.shape[0] / 2
    best_label = None
    best_dist = float('inf')
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 3:
            continue
        dx = centroids[i][0] - pcx
        dy = centroids[i][1] - pcy
        d = dx * dx + dy * dy
        if d < best_dist:
            best_dist = d
            best_label = i

    if best_label is None:
        return pos

    # Compute intensity-weighted centroid within the selected blob
    blob_mask = (labels == best_label)
    weights = contrast_map * blob_mask
    total = weights.sum()
    if total == 0:
        return pos

    yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    cx = (xx * weights).sum() / total + x1
    cy = (yy * weights).sum() / total + y1
    return (cx, cy)


def extract_template(gray, center, patch_size=40):
    """Extract a square template patch around a dot center."""
    h, w = gray.shape
    half = patch_size // 2
    x, y = int(center[0]), int(center[1])
    x1, y1 = max(0, x - half), max(0, y - half)
    x2, y2 = min(w, x + half), min(h, y + half)
    return gray[y1:y2, x1:x2].copy()


class VideoTracker:
    """
    Stateful tracker that processes one frame at a time so the GUI
    can display progress and the annotated frame after each step.
    """

    def __init__(self, video_path, frame_skip=1, initial_distance_mm=None):
        self.video_path = Path(video_path)
        self.frame_skip = frame_skip  # 1 = every frame, 2 = every other, 4 = every 4th
        self.initial_distance_mm = initial_distance_mm
        self.px_per_mm = None

        self.cap = None
        self.fps = 0
        self.total_frames = 0
        self.width = 0
        self.height = 0

        self.pos_top = None
        self.pos_bot = None
        self.template_top = None
        self.template_bot = None
        self.ref_template_top = None
        self.ref_template_bot = None

        self.results = []          # [(time, distance), ...]
        self.positions = []        # [(pos_top, pos_bot), ...] parallel to results
        self.frame_indices = []    # [frame_idx, ...] parallel to results
        self.current_frame_idx = 0
        self.consecutive_failures = 0
        self.finished = False
        self.error = None

        self._template_update_counter = 0

    # ---- public API ----

    def open(self):
        """Open video and detect dots in the first frame. Returns annotated first frame (BGR)."""
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            self.error = f"Cannot open video: {self.video_path.name}"
            self.finished = True
            return None

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ret, frame = self.cap.read()
        if not ret:
            self.error = "Cannot read first frame"
            self.finished = True
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dots = find_initial_dots(gray)
        if dots is None:
            self.error = "Could not detect 2 dots in first frame"
            self.finished = True
            return None

        self.pos_top = refine_centroid(gray, dots[0])
        self.pos_bot = refine_centroid(gray, dots[1])

        init_px = np.sqrt((self.pos_top[0] - self.pos_bot[0])**2 +
                          (self.pos_top[1] - self.pos_bot[1])**2)

        if self.initial_distance_mm is not None:
            self.px_per_mm = init_px / self.initial_distance_mm

        self.template_top = extract_template(gray, self.pos_top, 40)
        self.template_bot = extract_template(gray, self.pos_bot, 40)
        self.ref_template_top = self.template_top.copy()
        self.ref_template_bot = self.template_bot.copy()

        dist = init_px / self.px_per_mm if self.px_per_mm else init_px
        self.results.append((0.0, dist))
        self.positions.append((self.pos_top, self.pos_bot))
        self.frame_indices.append(0)
        self.current_frame_idx = 0

        return self._annotate(frame)

    def step(self):
        """
        Process the next frame (or batch if frame_skip > 1).
        Returns annotated BGR frame, or None when finished / on error.
        """
        if self.finished or self.cap is None:
            return None

        # Skip frames according to frame_skip
        for _ in range(self.frame_skip):
            ret, frame = self.cap.read()
            self.current_frame_idx += 1
            if not ret:
                self.finished = True
                self.cap.release()
                return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        search_radius = 40 + self.consecutive_failures * 20

        new_top, _ = track_dot_template(gray, self.template_top, self.pos_top, search_radius)
        if new_top is None:
            new_top, _ = track_dot_template(gray, self.ref_template_top, self.pos_top, search_radius + 30)

        new_bot, _ = track_dot_template(gray, self.template_bot, self.pos_bot, search_radius)
        if new_bot is None:
            new_bot, _ = track_dot_template(gray, self.ref_template_bot, self.pos_bot, search_radius + 30)

        t = self.current_frame_idx / self.fps

        if new_top is None or new_bot is None:
            self.consecutive_failures += 1
            if self.consecutive_failures > 60:
                self.error = f"Lost tracking at {t:.1f}s"
                self.finished = True
                self.cap.release()
                return None
            self.results.append((t, self.results[-1][1]))
            self.positions.append((self.pos_top, self.pos_bot))
            self.frame_indices.append(self.current_frame_idx)
            return self._annotate(frame)

        new_top = refine_centroid(gray, new_top, 24)
        new_bot = refine_centroid(gray, new_bot, 24)

        jump_top = np.sqrt((new_top[0] - self.pos_top[0])**2 + (new_top[1] - self.pos_top[1])**2)
        jump_bot = np.sqrt((new_bot[0] - self.pos_bot[0])**2 + (new_bot[1] - self.pos_bot[1])**2)

        if jump_top > search_radius * 0.8 or jump_bot > search_radius * 0.8:
            self.consecutive_failures += 1
            self.results.append((t, self.results[-1][1]))
            self.positions.append((self.pos_top, self.pos_bot))
            self.frame_indices.append(self.current_frame_idx)
            return self._annotate(frame)

        self.consecutive_failures = 0
        self.pos_top = new_top
        self.pos_bot = new_bot

        self._template_update_counter += 1
        if self._template_update_counter >= 15:
            self._template_update_counter = 0
            self.template_top = extract_template(gray, self.pos_top, 40)
            self.template_bot = extract_template(gray, self.pos_bot, 40)

        px_dist = np.sqrt((self.pos_top[0] - self.pos_bot[0])**2 +
                          (self.pos_top[1] - self.pos_bot[1])**2)
        dist = px_dist / self.px_per_mm if self.px_per_mm else px_dist
        self.results.append((t, dist))
        self.positions.append((self.pos_top, self.pos_bot))
        self.frame_indices.append(self.current_frame_idx)

        return self._annotate(frame)

    def release(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

    @property
    def progress(self):
        if self.total_frames == 0:
            return 0.0
        return self.current_frame_idx / self.total_frames

    @property
    def unit(self):
        return "mm" if self.px_per_mm else "px"

    @property
    def info_text(self):
        parts = [f"{self.width}x{self.height}",
                 f"{self.fps:.0f} fps",
                 f"{self.total_frames} frames"]
        if self.initial_distance_mm:
            parts.append(f"cal: {self.initial_distance_mm} mm")
        return "  |  ".join(parts)

    def save_csv(self, path, results=None, positions=None,
                 track_pixel_pos=False, track_mm_pos=False,
                 track_dot_disp=False, track_interdot_disp=True,
                 track_interdot_dist=False):
        """
        Write selected tracking variables to CSV.

        Coordinate system: origin at bottom-left of frame (y is flipped).

        Columns (depending on options selected):
          time_s
          top_x_px, top_y_px, bot_x_px, bot_y_px       [pixel_pos]
          top_x_mm, top_y_mm, bot_x_mm, bot_y_mm        [mm_pos]
          top_dx_<unit>, top_dy_<unit>,
          bot_dx_<unit>, bot_dy_<unit>                   [dot_disp]
          displacement_<unit>                            [interdot_disp]
          distance_<unit>                                [interdot_dist]
        """
        if results is None:
            results = self.results
        if positions is None:
            positions = self.positions

        h = self.height
        ppm = self.px_per_mm
        unit = self.unit

        d0 = results[0][1] if results else 0
        top0 = positions[0][0] if positions else None
        bot0 = positions[0][1] if positions else None

        # Build header
        header = ['time_s']
        if track_pixel_pos:
            header += ['dot1_x_px', 'dot1_y_px', 'dot2_x_px', 'dot2_y_px']
        if track_mm_pos:
            header += ['dot1_x_mm', 'dot1_y_mm', 'dot2_x_mm', 'dot2_y_mm']
        if track_dot_disp:
            header += [f'dot1_dx_{unit}', f'dot1_dy_{unit}',
                       f'dot2_dx_{unit}', f'dot2_dy_{unit}']
        if track_interdot_disp:
            header.append(f'displacement_{unit}')
        if track_interdot_dist:
            header.append(f'distance_{unit}')

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i, (t, d) in enumerate(results):
                pt = positions[i][0] if i < len(positions) else None
                pb = positions[i][1] if i < len(positions) else None
                row = [f'{t:.4f}']

                # dot1 = bottom (pb), dot2 = top (pt)
                if track_pixel_pos:
                    if pt is not None and pb is not None:
                        row += [f'{pb[0]:.2f}', f'{h - pb[1]:.2f}',
                                f'{pt[0]:.2f}', f'{h - pt[1]:.2f}']
                    else:
                        row += ['', '', '', '']

                if track_mm_pos:
                    if pt is not None and pb is not None and ppm:
                        row += [f'{pb[0]/ppm:.4f}', f'{(h - pb[1])/ppm:.4f}',
                                f'{pt[0]/ppm:.4f}', f'{(h - pt[1])/ppm:.4f}']
                    else:
                        row += ['', '', '', '']

                if track_dot_disp:
                    if pt is not None and pb is not None and top0 is not None and bot0 is not None:
                        if ppm:
                            row += [f'{(pb[0] - bot0[0])/ppm:.4f}',
                                    f'{((h - pb[1]) - (h - bot0[1]))/ppm:.4f}',
                                    f'{(pt[0] - top0[0])/ppm:.4f}',
                                    f'{((h - pt[1]) - (h - top0[1]))/ppm:.4f}']
                        else:
                            row += [f'{pb[0] - bot0[0]:.2f}',
                                    f'{(h - pb[1]) - (h - bot0[1]):.2f}',
                                    f'{pt[0] - top0[0]:.2f}',
                                    f'{(h - pt[1]) - (h - top0[1]):.2f}']
                    else:
                        row += ['', '', '', '']

                if track_interdot_disp:
                    row.append(f'{d - d0:.4f}')

                if track_interdot_dist:
                    row.append(f'{d:.4f}')

                writer.writerow(row)

    # ---- internal ----

    def _annotate(self, frame):
        """Draw dot markers and distance on the frame."""
        dist_val = self.results[-1][1] if self.results else None
        return annotate_frame(frame, self.pos_top, self.pos_bot, dist_val, self.unit)


def annotate_frame(frame, pos_top, pos_bot, dist_val=None, unit="px"):
    """Draw semi-transparent crosshairs, line, and distance label."""
    vis = frame.copy()
    if pos_top is None or pos_bot is None:
        return vis

    pt1 = (int(pos_top[0]), int(pos_top[1]))
    pt2 = (int(pos_bot[0]), int(pos_bot[1]))

    # Draw crosshairs and circles on an overlay, then blend
    overlay = vis.copy()
    sz = 18
    color = (0, 255, 0)
    for pt in [pt1, pt2]:
        cv2.line(overlay, (pt[0] - sz, pt[1]), (pt[0] + sz, pt[1]), color, 2)
        cv2.line(overlay, (pt[0], pt[1] - sz), (pt[0], pt[1] + sz), color, 2)
        cv2.circle(overlay, pt, 12, color, 2)
    cv2.line(overlay, pt1, pt2, (0, 200, 255), 1, cv2.LINE_AA)

    # Blend at 50% opacity so the dot is visible underneath
    cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)

    # Distance label (drawn at full opacity so it's readable)
    if dist_val is not None:
        label = f"{dist_val:.2f} {unit}"
        mid = ((pt1[0] + pt2[0]) // 2 + 15, (pt1[1] + pt2[1]) // 2)
        cv2.putText(vis, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2, cv2.LINE_AA)

    return vis
