"""
Core dot-tracking logic for Instron tensile, roller, and hinge test videos.

Supports several test types, auto-detected from the filename:
  - "tensile":      two dark Sharpie dots on a bright specimen between jaws
  - "roller":       one or two bright magenta paint-pen dots on a grey mechanism
  - "hinge":        two sets of small black marker dots (up to 4 each) on the
                    two coupler bars of a hinge mechanism
  - "hinge_colored": 2 green + 2 yellow paint dots on a hinge mechanism
                    (the original color-based hinge detector)

The tracker stores a variable number of dots (1 to 8) per video and
computes inter-dot distance only when there are exactly 2 dots.
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


VALID_TEST_TYPES = ('tensile', 'roller', 'hinge', 'hinge_colored')


def detect_test_type(filename):
    """Infer test type from filename.

    Returns one of VALID_TEST_TYPES if the filename contains that keyword
    (case-insensitive), or None if no recognized keyword is present. There is
    no silent default: callers should refuse to process a None-typed video so
    the user can correct the filename rather than get the wrong detector.

    'hinge colored' / 'hinge-colored' / 'hinge_colored' selects the legacy
    green/yellow paint-dot detector; plain 'hinge' selects the black-dot
    detector (two sets of up to 4 small dark marker dots).
    """
    name = Path(filename).stem.lower()
    if 'roller' in name:
        return 'roller'
    if 'hinge colored' in name or 'hinge-colored' in name or 'hinge_colored' in name:
        return 'hinge_colored'
    if 'hinge' in name:
        return 'hinge'
    if 'tensile' in name:
        return 'tensile'
    return None


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


def find_initial_dots_tensile(gray, max_dots=4):
    """
    Find up to `max_dots` dark Sharpie dots on a light specimen in the
    first frame. Returns a list of (x, y) sorted bottom-to-top, or None.
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

    # Greedily select up to `max_dots` highest-scoring clusters that are
    # each vertically separated (>= 50px) from every already-selected dot,
    # then return them sorted bottom-to-top (dot1 = bottom).
    clusters.sort(key=lambda c: c[4], reverse=True)
    selected = []
    for cl in clusters:
        if len(selected) >= max_dots:
            break
        if all(abs(cl[1] - s[1]) >= 50 for s in selected):
            selected.append(cl)
    if len(selected) < 2:
        return None
    selected.sort(key=lambda c: c[1], reverse=True)  # bottom (larger y) first
    return [(c[0], c[1]) for c in selected]


# ── Roller: bright magenta paint-pen dot detection ──────────────────────────
def find_initial_dots_roller(bgr, max_dots=4):
    """
    Find up to `max_dots` bright magenta paint-pen dots on a grey mechanism.

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

    # Accept further dots only if their score is comparable and they are
    # well-separated (>100px) from every dot already accepted.
    for cand in candidates[1:]:
        if len(dots) >= max_dots:
            break
        if cand[4] <= 0.6 * top[4]:
            continue
        if all(np.hypot(cand[0] - dx, cand[1] - dy) > 100 for dx, dy in dots):
            dots.append((cand[0], cand[1]))

    # Sort bottom-to-top (larger y = bottom on screen)
    dots.sort(key=lambda p: p[1], reverse=True)
    return dots


# ── Hinge: green/yellow paint-dot detection ──────────────────────────────────
# Green dots are bright and well-saturated; yellow dots are dimmer and sit at a
# slightly lower hue. Thresholds give margin above the dim yellow dots while
# excluding metal hardware (screw heads etc.) with a similar but darker, less
# saturated yellow-brown tint (H~20, S~88, V~92).
GREEN_H = (45, 85)
GREEN_S = 70
GREEN_V = 70
YELLOW_H = (33, 44)
YELLOW_S = 100
YELLOW_V = 110


def _maybe_split_blobs(blobs, lab, max_dots):
    """
    If fewer than `max_dots` blobs were found and the largest one is
    oversized (likely two dots merged together), erode its mask until it
    splits into two components and use their centroids.
    """
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


def _find_color_dots(hsv, h_lo, h_hi, s_min, v_min, max_dots):
    """Find up to `max_dots` blobs within the given HSV range, sorted
    top-to-bottom (smaller y first)."""
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = ((h_ch >= h_lo) & (h_ch <= h_hi) &
            (s_ch >= s_min) & (v_ch >= v_min)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    n_labels, labels, stats, cents = cv2.connectedComponentsWithStats(mask)

    blobs = []
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 12 or area > 1500:
            continue
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        aspect = bw / max(bh, 1)
        if aspect < 0.3 or aspect > 3.3:
            continue
        cx, cy = cents[i]
        blobs.append([float(cx), float(cy), int(area), i])

    blobs = _maybe_split_blobs(blobs, labels, max_dots)

    blobs.sort(key=lambda b: b[2], reverse=True)
    blobs = blobs[:max_dots]
    blobs.sort(key=lambda b: b[1])  # top-to-bottom
    return [(b[0], b[1]) for b in blobs]


def find_initial_dots_color(bgr, max_per_color=2):
    """
    Find up to `max_per_color` green dots and `max_per_color` yellow dots.

    Returns a list of ((x, y), color) tuples — green dots first (top-to-
    bottom), then yellow dots (top-to-bottom) — or None if neither color
    was found.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    greens = _find_color_dots(hsv, *GREEN_H, GREEN_S, GREEN_V, max_per_color)
    yellows = _find_color_dots(hsv, *YELLOW_H, YELLOW_S, YELLOW_V, max_per_color)
    if not greens and not yellows:
        return None
    return [(p, 'green') for p in greens] + [(p, 'yellow') for p in yellows]


# ── Hinge: small black marker-dot detection (two sets of up to 4) ───────────
def _best_collinear_group(points, max_pts=4, min_pts=3,
                           perp_tol=8.0, min_span=15.0, max_span=180.0):
    """
    Find the best-scoring group of `min_pts`-`max_pts` roughly-collinear
    points — a row of marker dots on a hinge bar.

    For every pair of points, treat them as defining a line and count how
    many other points lie within `perp_tol` of it. The pair/line with the
    most inliers (ties broken by total contrast) wins. If more than
    `max_pts` points lie near the winning line, keep the `max_pts` closest
    to it. Returns the winning list of points, or None if no pair yields
    at least `min_pts` inliers.
    """
    best = None
    best_score = (0, 0.0)
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, _ = points[i]
            x2, y2, _ = points[j]
            span = np.hypot(x2 - x1, y2 - y1)
            if span < min_span or span > max_span:
                continue
            ux, uy = (x2 - x1) / span, (y2 - y1) / span
            inliers = []
            for p in points:
                px, py, _ = p
                vx, vy = px - x1, py - y1
                perp = abs(vx * uy - vy * ux)
                if perp <= perp_tol:
                    inliers.append(p)
            if len(inliers) < min_pts:
                continue
            if len(inliers) > max_pts:
                inliers.sort(key=lambda p: abs((p[0] - x1) * uy - (p[1] - y1) * ux))
                inliers = inliers[:max_pts]
            score = (len(inliers), sum(p[2] for p in inliers))
            if score > best_score:
                best_score = score
                best = inliers
    return best


def _merge_close_points(points, dist_thresh=10.0):
    """
    Single-linkage merge of (x, y, score) points within `dist_thresh` of
    each other, repeated until stable. Each merged group becomes one point
    at the centroid of its members, with the max score.
    """
    current = list(points)
    while True:
        merged = []
        used = set()
        changed = False
        for i, c1 in enumerate(current):
            if i in used:
                continue
            group = [c1]
            used.add(i)
            for j, c2 in enumerate(current):
                if j in used:
                    continue
                if np.hypot(c1[0] - c2[0], c1[1] - c2[1]) < dist_thresh:
                    group.append(c2)
                    used.add(j)
            if len(group) > 1:
                changed = True
            avg_x = np.mean([g[0] for g in group])
            avg_y = np.mean([g[1] for g in group])
            max_score = max(g[2] for g in group)
            merged.append((avg_x, avg_y, max_score))
        current = merged
        if not changed:
            return current


def find_initial_dots_hinge_black(bgr, max_per_set=4):
    """
    Find up to two sets of `max_per_set` small black marker dots, one set
    per coupler bar of the hinge mechanism.

    Returns a list of ((x, y), group) tuples — 'set1' dots first (the
    higher/upper bar, ordered left-to-right), then 'set2' dots — or None
    if two collinear groups of at least 3 dots each could not be found.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    all_candidates = []
    for thresh_val in range(90, 175, 5):
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area < 6 or area > 150:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            aspect = bw / max(bh, 1)
            if aspect < 0.2 or aspect > 5.0:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            ix, iy = int(cx), int(cy)
            inner_r, outer_r = 8, 22
            y1, y2 = max(0, iy - outer_r), min(h, iy + outer_r)
            x1, x2 = max(0, ix - outer_r), min(w, ix + outer_r)
            patch = gray[y1:y2, x1:x2]

            yy, xx = np.ogrid[-(iy - y1):(y2 - iy), -(ix - x1):(x2 - ix)]
            dist = np.sqrt(xx.astype(float)**2 + yy.astype(float)**2)
            ann = (dist > inner_r) & (dist <= outer_r)
            if ann.sum() == 0:
                continue

            surround = patch[ann].mean()
            center = gray[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3].mean()
            contrast = surround - center
            if surround > 130 and contrast > 30:
                all_candidates.append((cx, cy, contrast))

    if len(all_candidates) < 3:
        return None

    # Cluster nearby detections (same physical dot found at multiple
    # thresholds) and drop large/stable blobs (screws, hardware) that get
    # detected at almost every threshold level.
    clusters = []
    used = set()
    for i, c1 in enumerate(all_candidates):
        if i in used:
            continue
        group = [c1]
        used.add(i)
        for j, c2 in enumerate(all_candidates):
            if j in used:
                continue
            if np.hypot(c1[0] - c2[0], c1[1] - c2[1]) < 10:
                group.append(c2)
                used.add(j)
        avg_x = np.mean([g[0] for g in group])
        avg_y = np.mean([g[1] for g in group])
        max_contrast = max(g[2] for g in group)
        if len(group) <= 10:
            clusters.append((avg_x, avg_y, max_contrast))

    if len(clusters) < 3:
        return None

    # A second merge pass collapses near-duplicate cluster centroids that
    # single-linkage chaining can leave behind (e.g. two sub-clusters of
    # the same dot whose centroids end up a few px apart).
    clusters = _merge_close_points(clusters, dist_thresh=10.0)
    if len(clusters) < 3:
        return None

    remaining = list(clusters)
    groups = []
    for _ in range(2):
        group = _best_collinear_group(remaining, max_per_set)
        if group is None:
            break
        groups.append(group)
        remaining = [p for p in remaining if p not in group]

    if len(groups) < 2:
        return None

    # Order groups top-to-bottom (set1 = upper bar, set2 = lower bar), and
    # dots within each group left-to-right (the faint 4th dot, if found,
    # ends up last).
    groups.sort(key=lambda g: np.mean([p[1] for p in g]))

    labeled = []
    for set_idx, group in enumerate(groups, start=1):
        ordered = sorted(group, key=lambda p: p[0])
        for p in ordered:
            labeled.append(((p[0], p[1]), f'set{set_idx}'))
    return labeled


# ── Unified initial detection ───────────────────────────────────────────────
def find_initial_dots(frame_bgr, test_type='tensile'):
    """Dispatch to the right detection for the given test type."""
    if test_type == 'roller':
        return find_initial_dots_roller(frame_bgr)
    if test_type == 'hinge_colored':
        labeled = find_initial_dots_color(frame_bgr)
        return [p for p, _c in labeled] if labeled else None
    if test_type == 'hinge':
        labeled = find_initial_dots_hinge_black(frame_bgr)
        return [p for p, _g in labeled] if labeled else None
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


def refine_centroid_color(bgr, pos, color, patch_size=30):
    """
    Snap to the centroid of a green or yellow paint dot near `pos`.
    Uses saturation weighting within the dot's own hue range.
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

    h_lo, h_hi = GREEN_H if color == 'green' else YELLOW_H
    hue_ok = ((h_ch >= h_lo) & (h_ch <= h_hi)).astype(float)
    weights = s_ch * hue_ok
    total = weights.sum()
    if total < 30:
        return pos  # not enough of this color found

    yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    cx = (xx * weights).sum() / total + x1
    cy = (yy * weights).sum() / total + y1
    return (cx, cy)


def refine_centroid(frame_bgr, gray, pos, test_type='tensile', patch_size=30, color=None):
    """Dispatch to the right refinement for the test type."""
    if test_type == 'roller':
        return refine_centroid_bright(frame_bgr, pos, patch_size)
    if test_type == 'hinge_colored':
        # Hinge dots sit close together (~30px); cap the patch so the
        # centroid doesn't bleed into a neighboring dot of the same color.
        return refine_centroid_color(frame_bgr, pos, color or 'green', min(patch_size, 10))
    if test_type == 'hinge':
        # Black marker dots sit close together (~30px) within a set; cap
        # the patch so the centroid doesn't bleed into a neighboring dot.
        return refine_centroid_dark(gray, pos, min(patch_size, 12))
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
    variable dot count (1 to 4) for tensile, roller, or hinge test videos.
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
        self.colors = []             # [None, ...] or ['green', 'green', 'yellow', 'yellow'] for hinge
        self.templates = []
        self.ref_templates = []
        self.n_dots = 0

        # History (parallel arrays)
        self.results = []            # [(time, inter_dot_distance_or_None), ...]
        self.positions = []          # [[(x,y), (x,y)], ...] length == n_dots per frame
        self.frame_indices = []
        self.current_frame_idx = 0
        self.consecutive_failures = 0
        self.redetections = 0        # number of successful re-locks via full detector
        self.finished = False
        self.error = None
        self._template_update_counter = 0

    # ---- public API ----
    def open(self):
        """Open video and detect dots in the first frame."""
        if self.test_type is None:
            self.error = (
                f"Could not determine test type for '{self.video_path.name}'. "
                f"Include one of these keywords in the filename: "
                f"{', '.join(VALID_TEST_TYPES)}."
            )
            self.finished = True
            return None

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

        if self.test_type == 'hinge_colored':
            labeled = find_initial_dots_color(frame)
            if not labeled:
                self.error = f"Could not detect any dots ({self.test_type})"
                self.finished = True
                return None
            detected = [p for p, _c in labeled]
            self.colors = [c for _p, c in labeled]
        elif self.test_type == 'hinge':
            labeled = find_initial_dots_hinge_black(frame)
            if not labeled:
                self.error = f"Could not detect any dots ({self.test_type})"
                self.finished = True
                return None
            detected = [p for p, _g in labeled]
            self.colors = [g for _p, g in labeled]
        else:
            detected = find_initial_dots(frame, self.test_type)
            if detected is None or len(detected) < 1:
                self.error = f"Could not detect any dots ({self.test_type})"
                self.finished = True
                return None
            self.colors = [None] * len(detected)

        # Refine each dot
        self.dots = [refine_centroid(frame, gray, d, self.test_type, color=c)
                     for d, c in zip(detected, self.colors)]
        self.n_dots = len(self.dots)

        # Templates (grayscale template matching works for both dot types)
        self.templates = [extract_template(gray, d, 40) for d in self.dots]
        self.ref_templates = [t.copy() for t in self.templates]

        # Calibration + inter-dot distance only make sense for exactly 2 dots.
        # With 3+ dots we track positions only (no inter-dot output).
        dist = None
        if self.n_dots == 2:
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

        # Track each dot independently via template matching
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

        track_ok = not any_fail
        if track_ok:
            # Refine centroids, then reject if any dot made an excessive jump
            refined = [refine_centroid(frame, gray, p, self.test_type, 24, color=c)
                       for p, c in zip(new_positions, self.colors)]
            max_jump = 0.0
            for i in range(self.n_dots):
                dx = refined[i][0] - self.dots[i][0]
                dy = refined[i][1] - self.dots[i][1]
                max_jump = max(max_jump, float(np.sqrt(dx * dx + dy * dy)))
            if max_jump > search_radius * 0.8:
                track_ok = False
            else:
                new_positions = refined

        # Fallback: if template tracking has failed for a few frames, run the
        # full initial detector to re-acquire the dot(s) after a large jump.
        if not track_ok and self.consecutive_failures >= 2:
            redetected = self._try_redetect(frame, gray)
            if redetected is not None:
                refined = [refine_centroid(frame, gray, p, self.test_type, 24, color=c)
                           for p, c in zip(redetected, self.colors)]
                self.dots = refined
                # Refresh templates from the re-acquired positions
                self.templates = [extract_template(gray, d, 40)
                                  for d in self.dots]
                self.consecutive_failures = 0
                self._template_update_counter = 0
                self.redetections += 1

                if self.n_dots == 2:
                    px_dist = self._dot_distance_px()
                    dist = (px_dist / self.px_per_mm
                            if self.px_per_mm else px_dist)
                else:
                    dist = None
                self.results.append((t, dist))
                self.positions.append(list(self.dots))
                self.frame_indices.append(self.current_frame_idx)
                return self._annotate(frame)

        if not track_ok:
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

        # Tracking succeeded normally
        self.consecutive_failures = 0
        self.dots = new_positions

        # Update rolling templates every 15 frames
        self._template_update_counter += 1
        if self._template_update_counter >= 15:
            self._template_update_counter = 0
            self.templates = [extract_template(gray, d, 40) for d in self.dots]

        # Inter-dot distance (only if exactly 2 dots)
        if self.n_dots == 2:
            px_dist = self._dot_distance_px()
            dist = px_dist / self.px_per_mm if self.px_per_mm else px_dist
        else:
            dist = None

        self.results.append((t, dist))
        self.positions.append(list(self.dots))
        self.frame_indices.append(self.current_frame_idx)
        return self._annotate(frame)

    def _try_redetect(self, frame, gray):
        """
        Run the full initial-dot detector on the current frame to re-acquire
        dots after template tracking fails. Matches detected candidates to
        existing dot identities by proximity to the last known positions,
        rejecting matches that are absurdly far (>200 + 60 * failures px).

        Returns a list of positions (one per existing dot), or None on failure.
        """
        # Max acceptable displacement grows with how long we've been lost.
        max_d = 200.0 + self.consecutive_failures * 60.0
        max_d2 = max_d * max_d

        if self.test_type in ('hinge', 'hinge_colored'):
            if self.test_type == 'hinge_colored':
                labeled = find_initial_dots_color(frame)
            else:
                labeled = find_initial_dots_hinge_black(frame)
            if not labeled:
                return None
            # Group candidates by their identity (color, or set1/set2) so
            # identities can't swap between groups.
            remaining = {}
            for p, g in labeled:
                remaining.setdefault(g, []).append(p)

            new_positions = [None] * self.n_dots
            for i in range(self.n_dots):
                last = self.dots[i]
                cands = remaining.get(self.colors[i], [])
                best_k, best_d2 = None, float('inf')
                for k, cand in enumerate(cands):
                    dx = cand[0] - last[0]
                    dy = cand[1] - last[1]
                    d2 = dx * dx + dy * dy
                    if d2 < best_d2:
                        best_d2 = d2
                        best_k = k
                if best_k is None or best_d2 > max_d2:
                    return None
                new_positions[i] = cands.pop(best_k)
            return new_positions

        detected = find_initial_dots(frame, self.test_type)
        if not detected:
            return None

        remaining = list(detected)
        new_positions = [None] * self.n_dots

        for i in range(self.n_dots):
            last = self.dots[i]
            best_k, best_d2 = None, float('inf')
            for k, cand in enumerate(remaining):
                dx = cand[0] - last[0]
                dy = cand[1] - last[1]
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    best_k = k
            if best_k is None or best_d2 > max_d2:
                return None
            new_positions[i] = remaining.pop(best_k)

        return new_positions

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

    def _dot_label(self, i):
        """Label for dot index i: 'green1'/'yellow2' for hinge_colored mode,
        'set1_1'/'set2_4' for hinge mode (1-indexed within each group), or
        'dot{i+1}' otherwise."""
        if self.colors and i < len(self.colors) and self.colors[i]:
            group = self.colors[i]
            nth = sum(1 for j in range(i + 1) if self.colors[j] == group)
            sep = '_' if group.startswith('set') else ''
            return f'{group}{sep}{nth}'
        return f'dot{i+1}'

    def _annotate(self, frame):
        dist_val = self.results[-1][1] if (self.results and self.n_dots == 2) else None
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
                 f"{self.test_type.replace('_', ' ')}"]
        if self.initial_distance_mm:
            parts.append(f"cal: {self.initial_distance_mm} mm")
        if self.redetections:
            parts.append(f"relocks: {self.redetections}")
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
            lbl = self._dot_label(i)
            if track_pixel_pos:
                header += [f'{lbl}_x_px', f'{lbl}_y_px']
        for i in range(n):
            lbl = self._dot_label(i)
            if track_mm_pos:
                header += [f'{lbl}_x_mm', f'{lbl}_y_mm']
        for i in range(n):
            lbl = self._dot_label(i)
            if track_dot_disp:
                header += [f'{lbl}_dx_{unit}', f'{lbl}_dy_{unit}']
        if track_interdot_disp and n == 2:
            header.append(f'displacement_{unit}')
        if track_interdot_dist and n == 2:
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
                if track_interdot_disp and n == 2:
                    if d is not None and d0 is not None:
                        row.append(f'{d - d0:.4f}')
                    else:
                        row.append('')
                if track_interdot_dist and n == 2:
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
    color = (255, 0, 0)  # bright blue (BGR)
    for d in valid_dots:
        pt = (int(d[0]), int(d[1]))
        cv2.line(overlay, (pt[0] - sz, pt[1]), (pt[0] + sz, pt[1]), color, 2)
        cv2.line(overlay, (pt[0], pt[1] - sz), (pt[0], pt[1] + sz), color, 2)
        cv2.circle(overlay, pt, 12, color, 2)

    # Connect dot1 and dot2 only when there are exactly two dots
    if len(dots) == 2 and dots[0] is not None and dots[1] is not None:
        pt1 = (int(dots[0][0]), int(dots[0][1]))
        pt2 = (int(dots[1][0]), int(dots[1][1]))
        cv2.line(overlay, pt1, pt2, (0, 200, 255), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)

    # Distance label between the two dots
    if dist_val is not None and len(dots) == 2 and dots[0] is not None and dots[1] is not None:
        pt1 = (int(dots[0][0]), int(dots[0][1]))
        pt2 = (int(dots[1][0]), int(dots[1][1]))
        label = f"{dist_val:.2f} {unit}"
        mid = ((pt1[0] + pt2[0]) // 2 + 15, (pt1[1] + pt2[1]) // 2)
        cv2.putText(vis, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2, cv2.LINE_AA)

    return vis
