"""
Core dot-tracking logic for Instron tensile and roller test videos.

Supports two test types, auto-detected from the filename:
  - "tensile": two dark Sharpie dots on a bright specimen between jaws
  - "roller":  one or two bright magenta paint-pen dots on a grey mechanism

The tracker stores a variable number of dots (1 or 2) per video and
computes inter-dot distance only when there are at least 2 dots.
"""

import cv2
import numpy as np
import csv
import re
from pathlib import Path


# ── Filename utilities ──────────────────────────────────────────────────────
def extract_initial_distance_mm(filename):
    """Extract initial dot distance from filename, e.g. '25.4mm' -> 25.4"""
    match = re.search(r'([\d.]+)\s*mm', filename, re.IGNORECASE)
    return float(match.group(1)) if match else None


def detect_test_type(filename):
    """Infer test type from filename. Returns 'tensile' or 'roller'."""
    name = Path(filename).stem.lower()
    if 'roller' in name:
        return 'roller'
    # Default to tensile (includes "Tensile", "Instron - side", anything else)
    return 'tensile'


# ── Tensile: specimen region + dark Sharpie dot detection ───────────────────
def find_specimen_region(gray):
    """
    Detect the specimen (light-colored dogbone) by finding the brightest gap
    between dark horizontal jaw bands. Returns (y_min, y_max) or None.
    """
    h, w = gray.shape
    dark_count = (gray < 60).sum(axis=1).astype(float)
    kernel = np.ones(10) / 10
    dark_smooth = np.convolve(dark_count, kernel, mode='same')
    is_jaw = dark_smooth > w * 0.3

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
    bands = [(s, e) for s, e in bands if e - s > 30]
    if len(bands) < 2:
        return None

    best = None
    best_brightness = 0
    cx = w // 2
    for i in range(len(bands) - 1):
        gap_top = bands[i][1]
        gap_bot = bands[i + 1][0]
        if gap_bot - gap_top < 80:
            continue
        x1 = max(0, cx - 80)
        x2 = min(w, cx + 80)
        gap_region = gray[gap_top:gap_bot, x1:x2]
        brightness = gap_region.mean()
        if brightness > best_brightness:
            best_brightness = brightness
            best = (gap_top, gap_bot)
    return best


def find_initial_dots_tensile(gray):
    """
    Find exactly 2 dark Sharpie dots on a light specimen in the first frame.
    Returns [(x_bottom, y_bottom), (x_top, y_top)] or None.
    """
    h, w = gray.shape

    spec = find_specimen_region(gray)
    if spec is not None:
        search_y_min, search_y_max = spec
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

    # Cluster candidates within 15px
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
        n_det = len(cluster)
        score = max_contrast * n_det
        clusters.append((avg_x, avg_y, max_contrast, n_det, score))

    if len(clusters) < 2:
        return None

    clusters.sort(key=lambda c: c[4], reverse=True)
    best_pair = None
    best_pair_score = 0
    for i in range(min(len(clusters), 8)):
        for j in range(i + 1, min(len(clusters), 8)):
            dy = abs(clusters[i][1] - clusters[j][1])
            if dy < 50:
                continue
            pair_score = clusters[i][4] + clusters[j][4]
            if pair_score > best_pair_score:
                best_pair_score = pair_score
                c1, c2 = clusters[i], clusters[j]
                # Return [bottom (larger y), top (smaller y)]
                if c1[1] > c2[1]:
                    best_pair = [(c1[0], c1[1]), (c2[0], c2[1])]
                else:
                    best_pair = [(c2[0], c2[1]), (c1[0], c1[1])]
    return best_pair


# ── Roller: bright magenta paint-pen dot detection ──────────────────────────
def find_initial_dots_roller(bgr, max_dots=2):
    """
    Find 1 or 2 bright magenta paint-pen dots on a grey mechanism.

    Returns list of positions sorted bottom-to-top (dots[0] = dot1 = bottom),
    or None if no confident dot is found.
    """
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Magenta hue range (distinct from workshop red/orange at 0-15)
    mask = ((h_ch >= 155) & (h_ch <= 180) &
            (s_ch > 140) & (v_ch > 80)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    s_f = s_ch.astype(float)
    candidates = []
    for i in range(1, n_labels):
        a = stats[i, cv2.CC_STAT_AREA]
        # Paint-pen dots are small and round; exclude giant background patches
        if a < 50 or a > 700:
            continue
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = bw / max(bh, 1)
        if aspect < 0.6 or aspect > 1.7:
            continue
        cx, cy = cents[i]
        ix, iy = int(cx), int(cy)

        # Annular saturation contrast: paint pen should stand out
        r_out = 35
        r_in = max(8, int(np.sqrt(a / np.pi)) + 3)
        y1, y2 = max(0, iy - r_out), min(h, iy + r_out)
        x1, x2 = max(0, ix - r_out), min(w, ix + r_out)
        s_patch = s_f[y1:y2, x1:x2]
        yy, xx = np.ogrid[-(iy - y1):(y2 - iy), -(ix - x1):(x2 - ix)]
        dist = np.sqrt(xx.astype(float)**2 + yy.astype(float)**2)
        ann = (dist > r_in) & (dist <= r_out)
        if ann.sum() < 10:
            continue
        s_sur = s_patch[ann].mean()
        cen_mask = dist <= 4
        s_cen = s_patch[cen_mask].mean() if cen_mask.sum() > 0 else s_ch[iy, ix]
        sat_contrast = s_cen - s_sur
        if sat_contrast < 90:
            continue

        circularity = min(aspect, 1.0 / max(aspect, 1e-6))
        score = sat_contrast * circularity
        candidates.append((cx, cy, a, sat_contrast, score))

    if not candidates:
        return None

    candidates.sort(key=lambda c: c[4], reverse=True)
    top = candidates[0]
    dots = [(top[0], top[1])]

    # Accept a second dot only if its score is comparable and it's well-separated
    for cand in candidates[1:max_dots]:
        dx = cand[0] - top[0]
        dy = cand[1] - top[1]
        separation = np.sqrt(dx * dx + dy * dy)
        if cand[4] > 0.6 * top[4] and separation > 100:
            dots.append((cand[0], cand[1]))

    # Sort bottom-to-top (larger y = bottom on screen)
    dots.sort(key=lambda p: p[1], reverse=True)
    return dots


# ── Unified initial detection ───────────────────────────────────────────────
def find_initial_dots(frame_bgr, test_type='tensile'):
    """Dispatch to the right detection for the given test type."""
    if test_type == 'roller':
        return find_initial_dots_roller(frame_bgr)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return find_initial_dots_tensile(gray)


# ── Tracking: template matching (shared across test types) ──────────────────
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


# ── Centroid refinement (per test type) ─────────────────────────────────────
def refine_centroid_dark(gray, pos, patch_size=30):
    """
    Snap to the centroid of a dark blob (Sharpie dot) near `pos`.
    Uses local adaptive contrast (background - pixel).
    """
    h, w = gray.shape
    half = patch_size
    x, y = int(pos[0]), int(pos[1])
    x1, y1 = max(0, x - half), max(0, y - half)
    x2, y2 = min(w, x + half), min(h, y + half)

    patch = gray[y1:y2, x1:x2].astype(np.float64)
    if patch.size == 0:
        return pos

    local_bg = cv2.GaussianBlur(patch, (0, 0), sigmaX=12)
    contrast_map = local_bg - patch
    peak = contrast_map.max()
    if peak < 10:
        return pos

    mask = contrast_map > peak * 0.35
    mask_u8 = mask.astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8)
    if n_labels < 2:
        return pos

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

    blob_mask = (labels == best_label)
    weights = contrast_map * blob_mask
    total = weights.sum()
    if total == 0:
        return pos
    yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    cx = (xx * weights).sum() / total + x1
    cy = (yy * weights).sum() / total + y1
    return (cx, cy)


def refine_centroid_bright(bgr, pos, patch_size=30):
    """
    Snap to the centroid of a bright magenta blob (paint pen) near `pos`.
    Uses saturation weighting within the magenta hue range.
    """
    h, w = bgr.shape[:2]
    half = patch_size
    x, y = int(pos[0]), int(pos[1])
    x1, y1 = max(0, x - half), max(0, y - half)
    x2, y2 = min(w, x + half), min(h, y + half)

    patch = bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return pos

    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h_ch = hsv_patch[:, :, 0]
    s_ch = hsv_patch[:, :, 1].astype(float)

    # Weight by saturation but only for magenta hue
    hue_ok = ((h_ch >= 155) & (h_ch <= 180)).astype(float)
    weights = s_ch * hue_ok
    total = weights.sum()
    if total < 50:
        return pos  # not enough magenta found

    yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    cx = (xx * weights).sum() / total + x1
    cy = (yy * weights).sum() / total + y1
    return (cx, cy)


def refine_centroid(frame_bgr, gray, pos, test_type='tensile', patch_size=30):
    """Dispatch to the right refinement for the test type."""
    if test_type == 'roller':
        return refine_centroid_bright(frame_bgr, pos, patch_size)
    return refine_centroid_dark(gray, pos, patch_size)


def extract_template(gray, center, patch_size=40):
    """Extract a square template patch around a dot center."""
    h, w = gray.shape
    half = patch_size // 2
    x, y = int(center[0]), int(center[1])
    x1, y1 = max(0, x - half), max(0, y - half)
    x2, y2 = min(w, x + half), min(h, y + half)
    return gray[y1:y2, x1:x2].copy()


# ── Main tracker class ──────────────────────────────────────────────────────
class VideoTracker:
    """
    Stateful tracker that processes one frame at a time. Handles
    variable dot count (1 or 2) for tensile or roller test videos.
    """

    def __init__(self, video_path, frame_skip=1,
                 initial_distance_mm=None, test_type=None):
        self.video_path = Path(video_path)
        self.frame_skip = frame_skip
        self.initial_distance_mm = initial_distance_mm
        self.test_type = test_type or detect_test_type(self.video_path.name)
        self.px_per_mm = None

        self.cap = None
        self.fps = 0
        self.total_frames = 0
        self.width = 0
        self.height = 0

        # Current state — lists indexed by dot (0 = dot1/bottom, 1 = dot2/top)
        self.dots = []               # [(x, y), ...]
        self.templates = []
        self.ref_templates = []
        self.n_dots = 0

        # History (parallel arrays)
        self.results = []            # [(time, inter_dot_distance_or_None), ...]
        self.positions = []          # [[(x,y), (x,y)], ...] length == n_dots per frame
        self.frame_indices = []
        self.current_frame_idx = 0
        self.consecutive_failures = 0
        self.finished = False
        self.error = None
        self._template_update_counter = 0

    # ---- public API ----
    def open(self):
        """Open video and detect dots in the first frame."""
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
        detected = find_initial_dots(frame, self.test_type)
        if detected is None or len(detected) < 1:
            self.error = f"Could not detect any dots ({self.test_type})"
            self.finished = True
            return None

        # Refine each dot
        self.dots = [refine_centroid(frame, gray, d, self.test_type)
                     for d in detected]
        self.n_dots = len(self.dots)

        # Templates (grayscale template matching works for both dot types)
        self.templates = [extract_template(gray, d, 40) for d in self.dots]
        self.ref_templates = [t.copy() for t in self.templates]

        # Calibration (only if 2 dots and filename provides initial distance)
        dist = None
        if self.n_dots >= 2:
            init_px = self._dot_distance_px()
            if self.initial_distance_mm is not None:
                self.px_per_mm = init_px / self.initial_distance_mm
            dist = init_px / self.px_per_mm if self.px_per_mm else init_px

        self.results.append((0.0, dist))
        self.positions.append(list(self.dots))
        self.frame_indices.append(0)
        self.current_frame_idx = 0
        return self._annotate(frame)

    def step(self):
        """Process the next frame (or batch if frame_skip > 1)."""
        if self.finished or self.cap is None:
            return None

        for _ in range(self.frame_skip):
            ret, frame = self.cap.read()
            self.current_frame_idx += 1
            if not ret:
                self.finished = True
                self.cap.release()
                return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        search_radius = 40 + self.consecutive_failures * 20
        t = self.current_frame_idx / self.fps

        # Track each dot independently
        new_positions = []
        any_fail = False
        for i in range(self.n_dots):
            new_pos, _ = track_dot_template(
                gray, self.templates[i], self.dots[i], search_radius)
            if new_pos is None:
                new_pos, _ = track_dot_template(
                    gray, self.ref_templates[i], self.dots[i], search_radius + 30)
            if new_pos is None:
                any_fail = True
                new_positions.append(None)
            else:
                new_positions.append(new_pos)

        if any_fail:
            self.consecutive_failures += 1
            if self.consecutive_failures > 60:
                self.error = f"Lost tracking at {t:.1f}s"
                self.finished = True
                self.cap.release()
                return None
            self.results.append((t, self.results[-1][1]))
            self.positions.append(list(self.dots))
            self.frame_indices.append(self.current_frame_idx)
            return self._annotate(frame)

        # Refine centroids
        new_positions = [
            refine_centroid(frame, gray, p, self.test_type, 24)
            for p in new_positions
        ]

        # Reject excessive jumps
        max_jump = 0
        for i in range(self.n_dots):
            dx = new_positions[i][0] - self.dots[i][0]
            dy = new_positions[i][1] - self.dots[i][1]
            jump = np.sqrt(dx * dx + dy * dy)
            max_jump = max(max_jump, jump)

        if max_jump > search_radius * 0.8:
            self.consecutive_failures += 1
            self.results.append((t, self.results[-1][1]))
            self.positions.append(list(self.dots))
            self.frame_indices.append(self.current_frame_idx)
            return self._annotate(frame)

        self.consecutive_failures = 0
        self.dots = new_positions

        # Update rolling templates every 15 frames
        self._template_update_counter += 1
        if self._template_update_counter >= 15:
            self._template_update_counter = 0
            self.templates = [extract_template(gray, d, 40) for d in self.dots]

        # Inter-dot distance (only if 2+ dots)
        if self.n_dots >= 2:
            px_dist = self._dot_distance_px()
            dist = px_dist / self.px_per_mm if self.px_per_mm else px_dist
        else:
            dist = None

        self.results.append((t, dist))
        self.positions.append(list(self.dots))
        self.frame_indices.append(self.current_frame_idx)
        return self._annotate(frame)

    def release(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

    # ---- helpers ----
    def _dot_distance_px(self):
        if self.n_dots < 2:
            return 0.0
        dx = self.dots[0][0] - self.dots[1][0]
        dy = self.dots[0][1] - self.dots[1][1]
        return float(np.sqrt(dx * dx + dy * dy))

    def _annotate(self, frame):
        dist_val = self.results[-1][1] if (self.results and self.n_dots >= 2) else None
        return annotate_frame(frame, self.dots, dist_val, self.unit)

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
                 f"{self.total_frames} frames",
                 f"{self.n_dots} dot{'s' if self.n_dots != 1 else ''}",
                 f"{self.test_type}"]
        if self.initial_distance_mm:
            parts.append(f"cal: {self.initial_distance_mm} mm")
        return "  |  ".join(parts)

    def save_csv(self, path, results=None, positions=None,
                 track_pixel_pos=False, track_mm_pos=False,
                 track_dot_disp=False, track_interdot_disp=True,
                 track_interdot_dist=False):
        """
        Write selected tracking variables to CSV. Coordinate origin is
        bottom-left (y is flipped from frame coords).

        Columns for each dot (1..n_dots) are included when the corresponding
        option is selected. Inter-dot columns are skipped when n_dots < 2.
        """
        if results is None:
            results = self.results
        if positions is None:
            positions = self.positions

        h = self.height
        ppm = self.px_per_mm
        unit = self.unit
        n = self.n_dots

        d0 = results[0][1] if (results and results[0][1] is not None) else None
        ref_positions = positions[0] if positions else [None] * n

        # ── Build header ────────────────────────────────────────────────
        header = ['time_s']
        for i in range(n):
            lbl = f'dot{i+1}'
            if track_pixel_pos:
                header += [f'{lbl}_x_px', f'{lbl}_y_px']
        for i in range(n):
            lbl = f'dot{i+1}'
            if track_mm_pos:
                header += [f'{lbl}_x_mm', f'{lbl}_y_mm']
        for i in range(n):
            lbl = f'dot{i+1}'
            if track_dot_disp:
                header += [f'{lbl}_dx_{unit}', f'{lbl}_dy_{unit}']
        if track_interdot_disp and n >= 2:
            header.append(f'displacement_{unit}')
        if track_interdot_dist and n >= 2:
            header.append(f'distance_{unit}')

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for idx, (t, d) in enumerate(results):
                pts = positions[idx] if idx < len(positions) else [None] * n
                # Pad in case positions list is shorter
                if len(pts) < n:
                    pts = list(pts) + [None] * (n - len(pts))
                row = [f'{t:.4f}']

                # Pixel position
                if track_pixel_pos:
                    for i in range(n):
                        p = pts[i]
                        if p is not None:
                            row += [f'{p[0]:.2f}', f'{h - p[1]:.2f}']
                        else:
                            row += ['', '']

                # mm position
                if track_mm_pos:
                    for i in range(n):
                        p = pts[i]
                        if p is not None and ppm:
                            row += [f'{p[0]/ppm:.4f}', f'{(h - p[1])/ppm:.4f}']
                        else:
                            row += ['', '']

                # Per-dot displacement
                if track_dot_disp:
                    for i in range(n):
                        p = pts[i]
                        p0 = ref_positions[i] if i < len(ref_positions) else None
                        if p is not None and p0 is not None:
                            dx = p[0] - p0[0]
                            dy = (h - p[1]) - (h - p0[1])
                            if ppm:
                                row += [f'{dx/ppm:.4f}', f'{dy/ppm:.4f}']
                            else:
                                row += [f'{dx:.2f}', f'{dy:.2f}']
                        else:
                            row += ['', '']

                # Inter-dot displacement / distance
                if track_interdot_disp and n >= 2:
                    if d is not None and d0 is not None:
                        row.append(f'{d - d0:.4f}')
                    else:
                        row.append('')
                if track_interdot_dist and n >= 2:
                    row.append(f'{d:.4f}' if d is not None else '')

                writer.writerow(row)


# ── Frame annotation ────────────────────────────────────────────────────────
def annotate_frame(frame, dots, dist_val=None, unit="px"):
    """Draw crosshairs on all dots. Line+distance label only between 2 dots."""
    vis = frame.copy()
    if not dots:
        return vis

    valid_dots = [d for d in dots if d is not None]
    if not valid_dots:
        return vis

    overlay = vis.copy()
    sz = 18
    color = (0, 255, 0)
    for d in valid_dots:
        pt = (int(d[0]), int(d[1]))
        cv2.line(overlay, (pt[0] - sz, pt[1]), (pt[0] + sz, pt[1]), color, 2)
        cv2.line(overlay, (pt[0], pt[1] - sz), (pt[0], pt[1] + sz), color, 2)
        cv2.circle(overlay, pt, 12, color, 2)

    # Connect dot1 and dot2 if both present
    if len(dots) >= 2 and dots[0] is not None and dots[1] is not None:
        pt1 = (int(dots[0][0]), int(dots[0][1]))
        pt2 = (int(dots[1][0]), int(dots[1][1]))
        cv2.line(overlay, pt1, pt2, (0, 200, 255), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)

    # Distance label between the two dots
    if dist_val is not None and len(dots) >= 2 and dots[0] is not None and dots[1] is not None:
        pt1 = (int(dots[0][0]), int(dots[0][1]))
        pt2 = (int(dots[1][0]), int(dots[1][1]))
        label = f"{dist_val:.2f} {unit}"
        mid = ((pt1[0] + pt2[0]) // 2 + 15, (pt1[1] + pt2[1]) // 2)
        cv2.putText(vis, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2, cv2.LINE_AA)

    return vis
