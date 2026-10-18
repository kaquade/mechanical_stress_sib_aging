"""Operando CT: inward and outward radial expansion of the jelly roll against cell voltage.

    python operando_ct.py --input <dataset>/operando_ct --output results

Writes expansion_per_scan.csv (one row per scan) and expansion_vs_voltage.csv
(mean and standard deviation over the complete cycles 2 to 5, per sweep direction).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy.signal import find_peaks


PIXEL_SIZE_UM = 13.792
CENTER_X = CENTER_Y = 668  # cell axis in the cross-sections


# =============================================================================
# 1. Cell voltage and current per scan
#    The potentiostat log and the scans are matched by their timestamps; a scan
#    lasts until the next one starts. The sweep direction, and with it the cycle
#    number, follows from the charge passed between successive scans.
# =============================================================================

LOG_NAME = re.compile(r"(\d{8})T(\d{6})")
MIN_VALID_VOLTAGE_V = 0.1


def read_log(path: Path, log_date: pd.Timestamp) -> pd.DataFrame:
    """Potentiostat log with an absolute time; a day is added where the clock wraps."""
    log = pd.read_csv(path, comment="#", names=["Timestamp", "U_V", "I_A", "P_W"], dtype=str)
    log = log[log["Timestamp"] != "Timestamp"].copy()
    for column in ("U_V", "I_A"):
        log[column] = pd.to_numeric(log[column], errors="coerce")
    log = log.dropna(subset=["U_V", "I_A"]).reset_index(drop=True)
    log["datetime"] = pd.to_datetime(log_date.strftime("%Y-%m-%d ") + log["Timestamp"].astype(str),
                                     format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
    rollover = (log["datetime"].diff() < pd.Timedelta(seconds=-1)).cumsum()
    log["datetime"] += rollover * pd.Timedelta(days=1)
    return log


def electrical_data(folder: Path) -> pd.DataFrame:
    """Voltage at start and end of every scan, mean voltage, mean current and cycle."""
    log_path = next(folder.glob("potentiostat_log_*.csv"))
    log_start = pd.to_datetime("".join(LOG_NAME.search(log_path.name).groups()),
                               format="%Y%m%d%H%M%S")
    table = pd.read_csv(folder / "scan_start_times.csv", parse_dates=["scan_start"])
    table = table.sort_values("scan")
    scan_starts = pd.Series(table["scan_start"].to_numpy(), index=table["scan"].to_numpy())

    log = read_log(log_path, log_start.normalize())
    dt_s = log["datetime"].diff().dt.total_seconds().fillna(0.1)
    log["charge_ah"] = (log["I_A"].fillna(0) * dt_s / 3600).cumsum()

    start_min = ((scan_starts - log_start).dt.total_seconds() / 60).to_numpy()
    end_min = np.append(start_min[1:], start_min[-1] + float(np.mean(np.diff(start_min))))
    log = log.assign(t_min=(log["datetime"] - log_start).dt.total_seconds() / 60)

    rows = []
    for scan, t0, t1 in zip(scan_starts.index, start_min, end_min):
        window = log[(log["t_min"] >= t0) & (log["t_min"] < t1)]
        window = window[window["U_V"] > MIN_VALID_VOLTAGE_V]
        row = {"scan": scan, "time_min": (t0 + t1) / 2}
        if window.empty:
            row.update(dict.fromkeys(("voltage_start_V", "voltage_end_V", "voltage_mean_V",
                                      "current_A", "charge_ah"), np.nan))
        else:
            row.update(voltage_start_V=float(window["U_V"].iloc[0]),
                       voltage_end_V=float(window["U_V"].iloc[-1]),
                       voltage_mean_V=float(window["U_V"].mean()),
                       current_A=float(window["I_A"].mean()),
                       charge_ah=float(window["charge_ah"].mean()))
        rows.append(row)
    scans = pd.DataFrame(rows)
    # A new cycle starts at every switch from discharging to charging.
    d_charge = scans["charge_ah"].diff().rolling(3, center=True, min_periods=1).mean()
    scans["cycle"] = ((d_charge >= 0).astype(int).diff() == 1).cumsum() + 1
    columns = ["time_min", "voltage_start_V", "voltage_end_V", "voltage_mean_V", "current_A"]
    # Values are used at four decimals, as in the reported analysis.
    scans[columns] = scans[columns].map(lambda x: float(f"{x:.4f}") if np.isfinite(x) else np.nan)
    return scans.drop(columns="charge_ah")


# =============================================================================
# 2. Inner boundary: core of the jelly roll
#    A 400 x 400 px crop around the cell axis is median filtered, normalized
#    between its 5th and 95th intensity percentile and thresholded at 0.61. The
#    core is the region around the axis bounded by the first cathode winding.
#    The unused inner anode can intermittently join the cathode and cut into the
#    core; the boundary is therefore held above the 90th percentile of all scans
#    minus 1 px, and single-scan outward excursions are capped by a three-scan median.
# =============================================================================

CROP_SIZE = 400
CATHODE_THRESHOLD = 0.61
MEDIAN_FILTER_SIZE = 3
CLOSING_RADIUS = 2
CORE_ANGLE_SAMPLES = 720
CORE_TEMPORAL_ENVELOPE_PERCENTILE = 90.0
CORE_INWARD_MOTION_ALLOWANCE = 1.0
CORE_ANGULAR_SMOOTHING_SIGMA = 1.5
MAX_CORE_AREA_DEVIATION = 0.20


def disk(radius: int) -> np.ndarray:
    coordinates = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    return xx * xx + yy * yy <= radius * radius


def touches_border(mask: np.ndarray) -> bool:
    return bool(np.any(mask[0, :]) or np.any(mask[-1, :]) or np.any(mask[:, 0])
                or np.any(mask[:, -1]))


def normalize_crops(crops: np.ndarray) -> np.ndarray:
    normalized = np.empty(crops.shape, dtype=np.float32)
    for index, crop in enumerate(crops):
        filtered = ndi.median_filter(crop, size=MEDIAN_FILTER_SIZE, mode="nearest")
        filtered = filtered.astype(np.float32)
        low = float(np.percentile(filtered, 5.0))
        high = float(np.percentile(filtered, 95.0))
        normalized[index] = np.clip((filtered - low) / (high - low), 0.0, 1.0)
    return normalized


def build_core_mask(electrode_mask: np.ndarray, seed_x: int, seed_y: int):
    """Core bounded by the first electrode wrap; a second closing removes the
    thin, open-ended unused anode inside it."""
    issues = []
    closed = ndi.binary_closing(electrode_mask, structure=disk(CLOSING_RADIUS), border_value=0)
    labels, _ = ndi.label(~closed, structure=np.ones((3, 3), dtype=bool))
    center_label = int(labels[seed_y, seed_x])
    if center_label == 0:
        return np.zeros_like(electrode_mask, dtype=bool), ["center_classified_as_electrode"]
    core = ndi.binary_fill_holes(labels == center_label).astype(bool)
    core = ndi.binary_closing(core, structure=disk(CLOSING_RADIUS), border_value=0)
    core = ndi.binary_fill_holes(core).astype(bool)
    if not np.any(core):
        issues.append("empty_core_mask")
    elif touches_border(core):
        issues.append("touches_crop_boundary")
    return core, issues


def stabilize_core_masks(core_masks, seed_x: int, seed_y: int) -> list[np.ndarray]:
    angle_samples = CORE_ANGLE_SAMPLES
    shape = core_masks[0].shape
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    distance = np.sqrt((xx - seed_x) ** 2 + (yy - seed_y) ** 2)
    theta = np.mod(np.arctan2(yy - seed_y, xx - seed_x), 2.0 * np.pi)
    angle_index = np.floor(theta / (2.0 * np.pi) * angle_samples).astype(np.int32) % angle_samples
    flat_distance, flat_angle_index = distance.ravel(), angle_index.ravel()

    boundary_radii = np.zeros((len(core_masks), angle_samples), dtype=np.float32)
    usable = np.ones(len(core_masks), dtype=bool)
    for index, mask in enumerate(core_masks):
        if not np.any(mask) or touches_border(mask):
            usable[index] = False
            continue
        np.maximum.at(boundary_radii[index], flat_angle_index[mask.ravel()],
                      flat_distance[mask.ravel()])
        detected = boundary_radii[index] > 0.0
        if np.count_nonzero(detected) < 3:
            usable[index] = False
        elif not np.all(detected):
            samples = np.arange(angle_samples)
            boundary_radii[index] = np.interp(samples, samples[detected],
                                              boundary_radii[index, detected], period=angle_samples)
    if np.count_nonzero(usable) < 2:
        return [mask.copy() for mask in core_masks]

    envelope = np.percentile(boundary_radii[usable], CORE_TEMPORAL_ENVELOPE_PERCENTILE, axis=0)
    envelope = ndi.gaussian_filter1d(envelope, sigma=CORE_ANGULAR_SMOOTHING_SIGMA, mode="wrap")
    stable = np.maximum(boundary_radii, (envelope - CORE_INWARD_MOTION_ALLOWANCE)[np.newaxis, :])
    despiked = stable.copy()
    for index in range(1, len(core_masks) - 1):
        if np.all(usable[index - 1: index + 2]):
            despiked[index] = np.minimum(stable[index],
                                         np.median(stable[index - 1: index + 2], axis=0))
    return [distance <= radii[angle_index] if is_usable else mask.copy()
            for mask, radii, is_usable in zip(core_masks, despiked, usable)]


def core_areas(images: list[np.ndarray]) -> tuple[np.ndarray, list[list[str]]]:
    """Core area in px of every scan and the issues found."""
    left, top = CENTER_X - CROP_SIZE // 2, CENTER_Y - CROP_SIZE // 2
    crops = np.stack([image[top: top + CROP_SIZE, left: left + CROP_SIZE] for image in images])
    normalized = normalize_crops(crops)
    seed = CROP_SIZE // 2
    masks, issues = zip(*(build_core_mask(frame > CATHODE_THRESHOLD, seed, seed)
                          for frame in normalized))
    issues = [list(i) for i in issues]
    areas = np.asarray([np.count_nonzero(m) for m in stabilize_core_masks(list(masks), seed, seed)])
    usable = [a for a, i in zip(areas, issues) if a > 0 and "touches_crop_boundary" not in i]
    median_area = float(np.median(usable)) if usable else 0.0
    for area, frame_issues in zip(areas, issues):
        deviation = abs(area - median_area) / median_area if median_area > 0 else 0.0
        if area > 0 and deviation > MAX_CORE_AREA_DEVIATION:
            frame_issues.append("core_area_outlier")
    return areas, issues


# =============================================================================
# 3. Outer boundary: edge of the jelly roll
#    Radial intensity profiles at 1,440 angles between 560 and 670 px from the
#    axis. The inner wall of the can is where the profile first exceeds the Otsu
#    threshold of the can annulus; electrode layers are curvature maxima of the
#    profile that stand out from the valley inside them. The edge follows the
#    outermost layer, 2 px outward, except over the current-collector tab, where
#    it follows the tab's outer face, and across the low-density void at the end
#    of the cathode winding, where the loose outer anode is skipped and the edge
#    follows the next electrode inward. The void is detected in every scan; its
#    stack median is used for all scans.
# =============================================================================

OUTER_MIN_RADIUS, OUTER_MAX_RADIUS = 560, 670
CAN_ANNULUS = (620, 665)
CAN_MINIMUM_RUN_LENGTH = 4
OUTER_BOUNDARY_INSET = 2.0
OUTER_ANGLE_SAMPLES = 1440
OUTER_ANGULAR_MEDIAN_SIZE = 21
OUTER_ANGULAR_OUTLIER_LIMIT = 6.0
OUTER_ANGULAR_SMOOTHING_SIGMA = 2.0
MAX_OUTER_AREA_DEVIATION = 0.10
JELLY_ROLL_CURVATURE_SIGMA = 1.5
JELLY_ROLL_CURVATURE_PROMINENCE = 0.5
JELLY_ROLL_MIN_LAYER_CONTRAST = 10.0
JELLY_ROLL_LAYER_FLOOR_WINDOW = 14
JELLY_ROLL_PEAK_MIN_DISTANCE = 5
JELLY_ROLL_PEAK_SEARCH_DEPTH = 55
JELLY_ROLL_ANGULAR_MEDIAN_SIZE = 41
JELLY_ROLL_ANGULAR_OUTLIER_LIMIT = 6.0
WINDING_PITCH = 23.0  # px, cathode to cathode
OUTER_WRAP_SKIP_FRACTION = (0.30, 0.75)
TAB_INTENSITY_FACTOR = 1.6
TAB_MINIMUM_RUN = 5
TAB_GAP_CLOSING_SAMPLES = 41
TAB_OUTER_SEARCH_DEPTH = 20
LOOSE_ARC_INTENSITY_MEDIAN_SIZE = 21
LOOSE_ARC_REFINE_WINDOW_SAMPLES = 40
LOOSE_ARC_REFINE_MIN_STEP_PX = 4.0
LAYER_END_FADE_SAMPLES = 12
LOOSE_ARC_START_OFFSET_SAMPLES = 5
MASK_STEP_LIMIT_PX = 3.0
LOOSE_ARC_MIN_RUN_SAMPLES = 9
LOOSE_ARC_START_WINDOW_DEG = (0.0, 30.0)  # 0 deg points right, clockwise
LOOSE_ARC_END_WINDOW_DEG = (60.0, 120.0)
LOOSE_ARC_SPAN_DEG = (45.0, 135.0)
LOOSE_ARC_MIN_INNER_LAYER_COVERAGE = 0.98
LOOSE_ARC_BOUNDARY_STACK_OUTLIER_LIMIT = 2.0
LOOSE_ARC_BOUNDARY_TEMPORAL_MEDIAN_SIZE = 5
LOOSE_ARC_STABILIZATION_WINDOW_DEG = (0.0, 140.0)
LOOSE_ARC_PATH_ANCHOR_DEG = 45.0
LOOSE_ARC_PATH_MAX_STEP_PX = 3.0
LOOSE_ARC_CROSSING_MIN_RUN_SAMPLES = 4


def otsu_threshold(image: np.ndarray) -> float:
    """Otsu threshold of an image scaled to 0..1."""
    quantized = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    histogram = np.bincount(quantized.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram / histogram.sum()
    weight = np.cumsum(probabilities)
    mean = np.cumsum(probabilities * np.arange(256))
    denominator = weight * (1.0 - weight)
    variance = np.zeros(256, dtype=np.float64)
    valid = denominator > 0
    variance[valid] = (mean[-1] * weight[valid] - mean[valid]) ** 2 / denominator[valid]
    return (int(np.argmax(variance[:-1])) + 0.5) / 255.0


def geometry(shape: tuple[int, int]) -> dict:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    distance = np.sqrt((xx - CENTER_X) ** 2 + (yy - CENTER_Y) ** 2)
    theta = np.mod(np.arctan2(yy - CENTER_Y, xx - CENTER_X), 2.0 * np.pi)
    position = theta / (2.0 * np.pi) * OUTER_ANGLE_SAMPLES
    floor = np.floor(position)
    return {"distance": distance,
            "angle_index": floor.astype(np.int32) % OUTER_ANGLE_SAMPLES,
            "angle_fraction": (position - floor).astype(np.float32),
            "can_annulus": (distance >= CAN_ANNULUS[0]) & (distance <= CAN_ANNULUS[1])}


def circular_interpolate(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    if np.count_nonzero(valid) < 3:
        raise RuntimeError("too few valid angular samples for outer boundary")
    indices = np.arange(len(values))
    known, known_values = indices[valid], values[valid]
    return np.interp(indices, np.concatenate((known - len(values), known, known + len(values))),
                     np.concatenate((known_values, known_values, known_values)))


def mask_from_boundary(radii: np.ndarray, geo: dict) -> np.ndarray:
    """Rasterize a radius per angle; a step larger than a layer offset stays sharp."""
    inner = radii[geo["angle_index"]]
    outer = radii[(geo["angle_index"] + 1) % len(radii)]
    fraction = geo["angle_fraction"]
    local = inner * (1.0 - fraction) + outer * fraction
    local = np.where(np.abs(outer - inner) > MASK_STEP_LIMIT_PX,
                     np.where(fraction < 0.5, inner, outer), local)
    return geo["distance"] <= local


def true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    changes = np.diff(np.concatenate(([False], values, [False])).astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def clean_circular_boundary(raw, outlier_limit, median_size=OUTER_ANGULAR_MEDIAN_SIZE,
                            breakpoints=None) -> np.ndarray:
    """Interpolate, despike and smooth a boundary; arcs between breakpoints
    (layer changes) are filtered separately so each step stays a step."""
    if not breakpoints:
        boundary = circular_interpolate(raw)
        local_median = ndi.median_filter(boundary, size=median_size, mode="wrap")
        outliers = np.abs(boundary - local_median) > outlier_limit
        boundary[outliers] = local_median[outliers]
        return ndi.gaussian_filter1d(boundary, sigma=OUTER_ANGULAR_SMOOTHING_SIGMA, mode="wrap")
    angle_samples = len(raw)
    cuts = sorted({int(point) % angle_samples for point in breakpoints})
    cleaned = np.empty_like(raw, dtype=np.float64)
    for start, end in zip(cuts, cuts[1:] + [cuts[0] + angle_samples]):
        indices = np.arange(start, end) % angle_samples
        segment = np.asarray(raw[indices], dtype=np.float64)
        finite = np.flatnonzero(np.isfinite(segment))
        if not len(finite):
            cleaned[indices] = np.nan
            continue
        positions = np.arange(len(segment))
        segment = np.interp(positions, positions[finite], segment[finite])
        if len(segment) < 3:
            cleaned[indices] = segment
            continue
        size = min(median_size, len(segment) if len(segment) % 2 else len(segment) - 1)
        local_median = ndi.median_filter(segment, size=max(size, 3), mode="nearest")
        segment = np.where(np.abs(segment - local_median) > outlier_limit, local_median, segment)
        cleaned[indices] = ndi.gaussian_filter1d(segment, sigma=OUTER_ANGULAR_SMOOTHING_SIGMA,
                                                 mode="nearest")
    if not np.all(np.isfinite(cleaned)):
        cleaned = circular_interpolate(cleaned)
    return cleaned


def layer_indices(profile: np.ndarray, can_start_index: int) -> list[int]:
    """Electrode layers, innermost first: curvature maxima at least 10 gray
    levels above the valley just inside them."""
    if len(profile) < 3:
        return []
    smoothed = ndi.gaussian_filter1d(profile, sigma=1.0, mode="nearest")
    curvature = -ndi.gaussian_filter1d(profile, sigma=JELLY_ROLL_CURVATURE_SIGMA, order=2,
                                       mode="nearest")
    candidates, _ = find_peaks(curvature, prominence=JELLY_ROLL_CURVATURE_PROMINENCE,
                               distance=JELLY_ROLL_PEAK_MIN_DISTANCE)
    innermost = max(0, can_start_index - JELLY_ROLL_PEAK_SEARCH_DEPTH)
    accepted = []
    for index in candidates[candidates >= innermost]:
        index = int(index)
        floor = float(np.min(smoothed[max(0, index - JELLY_ROLL_LAYER_FLOOR_WINDOW): index + 1]))
        if float(smoothed[index]) - floor >= JELLY_ROLL_MIN_LAYER_CONTRAST:
            accepted.append(index)
    return accepted


def next_inward_layer_index(layers) -> int | None:
    """Layer 0.30 to 0.75 of a winding pitch inside the outermost one."""
    if len(layers) < 2:
        return None
    lower, upper = (fraction * WINDING_PITCH for fraction in OUTER_WRAP_SKIP_FRACTION)
    for candidate in reversed(layers[:-1]):
        separation = layers[-1] - candidate
        if lower <= separation <= upper:
            return candidate
        if separation > upper:
            break
    return None


def sample_outer_profiles(image: np.ndarray, geo: dict) -> dict:
    """Can, outermost-layer and next-layer radius at every angle."""
    filtered = ndi.median_filter(image, size=3, mode="nearest")
    can_threshold = otsu_threshold(filtered[geo["can_annulus"]].astype(np.float32) / 255.0) * 255.0
    angles = np.linspace(0.0, 2.0 * np.pi, OUTER_ANGLE_SAMPLES, endpoint=False)
    radii = np.arange(OUTER_MIN_RADIUS, OUTER_MAX_RADIUS + 1)
    samples = ndi.map_coordinates(
        filtered.astype(np.float32),
        [CENTER_Y + np.sin(angles)[:, np.newaxis] * radii,
         CENTER_X + np.cos(angles)[:, np.newaxis] * radii], order=1, mode="nearest")
    can_like = samples > can_threshold
    raw_can, raw_outer, raw_inner, interlayer = (
        np.full(OUTER_ANGLE_SAMPLES, np.nan) for _ in range(4))

    layers_by_angle = {}
    for index, radial_classes in enumerate(can_like):
        changes = np.diff(np.concatenate(([False], radial_classes, [False])).astype(np.int8))
        starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1) - 1
        qualifying = (ends - starts + 1) >= CAN_MINIMUM_RUN_LENGTH
        if not np.any(qualifying):
            continue
        can_start_index = int(starts[qualifying][int(np.argmax(ends[qualifying]))])
        raw_can[index] = radii[can_start_index] - 1.0
        layers = layer_indices(samples[index, : max(can_start_index - 1, 1)], can_start_index)
        if layers:
            layers_by_angle[index] = layers

    # The current-collector tab is far brighter than any electrode layer; runs
    # of such angles are the tab, single gaps in it are bridged.
    outermost = np.array([float(samples[i, layers[-1]]) for i, layers in layers_by_angle.items()])
    tab_level = float(np.median(outermost)) * TAB_INTENSITY_FACTOR if len(outermost) else math.inf
    tab = np.zeros(OUTER_ANGLE_SAMPLES, dtype=bool)
    for index, layers in layers_by_angle.items():
        tab[index] = float(samples[index, layers[-1]]) > tab_level
    tiled = ndi.binary_opening(np.tile(tab, 3), structure=np.ones(TAB_MINIMUM_RUN, dtype=bool))
    tiled = ndi.binary_closing(tiled, structure=np.ones(TAB_GAP_CLOSING_SAMPLES, dtype=bool))
    tab = tiled[OUTER_ANGLE_SAMPLES: 2 * OUTER_ANGLE_SAMPLES]

    for index, layers in layers_by_angle.items():
        outer_index = layers[-1]
        if tab[index]:
            # Over the tab the edge follows its outer face, where the profile
            # has fallen halfway back to the level outside.
            profile = samples[index]
            limit = min(outer_index + TAB_OUTER_SEARCH_DEPTH, len(profile) - 1)
            tail = profile[outer_index: limit + 1]
            if len(tail) > 2:
                beyond = np.flatnonzero(tail <= 0.5 * (float(tail[0]) + float(np.min(tail))))
                if len(beyond):
                    outer_index = outer_index + int(beyond[0])
        raw_outer[index] = radii[outer_index]
        inner_index = next_inward_layer_index(layers)
        if inner_index is None:
            continue
        raw_inner[index] = radii[inner_index]
        between = samples[index, inner_index + 1: outer_index]
        if len(between):
            interlayer[index] = float(np.mean(between))
    coverage = float(np.count_nonzero(np.isfinite(raw_can)) / OUTER_ANGLE_SAMPLES)
    return {"can": raw_can, "outer": raw_outer, "inner": raw_inner, "interlayer": interlayer,
            "tab": tab, "can_coverage": coverage}


def refine_sector_endpoint(outer_layer: np.ndarray, index: int, half_width: int = 10) -> int:
    """Snap a sector end onto the clearest radial step of the outer layer nearby."""
    angle_samples = len(outer_layer)
    smoothed = ndi.median_filter(circular_interpolate(outer_layer.copy()), size=9, mode="wrap")
    candidates = np.arange(index - LOOSE_ARC_REFINE_WINDOW_SAMPLES,
                           index + LOOSE_ARC_REFINE_WINDOW_SAMPLES + 1) % angle_samples
    steps = np.array([np.median(smoothed[np.arange(k, k + half_width) % angle_samples])
                      - np.median(smoothed[np.arange(k - half_width, k) % angle_samples])
                      for k in candidates])
    best = int(np.argmax(np.abs(steps)))
    return index if abs(steps[best]) < LOOSE_ARC_REFINE_MIN_STEP_PX else int(candidates[best])


def detect_loose_arc_sector(profiles: dict) -> tuple[int, int] | None:
    """Start and end angle index of the void crossed by the loose outer anode."""
    interlayer = profiles["interlayer"]
    valid = np.isfinite(interlayer) & np.isfinite(profiles["inner"])
    if np.count_nonzero(valid) < 3:
        return None
    smoothed = ndi.median_filter(circular_interpolate(interlayer),
                                 size=LOOSE_ARC_INTENSITY_MEDIAN_SIZE, mode="wrap")
    supported = ndi.binary_closing(
        np.tile(valid, 3), structure=np.ones(LOOSE_ARC_INTENSITY_MEDIAN_SIZE, dtype=bool)
    )[len(valid): 2 * len(valid)]
    low_density = supported & (smoothed < float(np.median(smoothed[valid])))
    n = len(interlayer)

    def window_runs(window):
        start = int(math.ceil(window[0] / 360.0 * n))
        end = int(math.floor(window[1] / 360.0 * n)) + 1
        return [(start + a, start + b) for a, b in true_runs(low_density[start:end])
                if b - a >= LOOSE_ARC_MIN_RUN_SAMPLES]

    start_runs = window_runs(LOOSE_ARC_START_WINDOW_DEG)
    end_runs = window_runs(LOOSE_ARC_END_WINDOW_DEG)
    if not start_runs or not end_runs:
        return None
    start = (refine_sector_endpoint(profiles["outer"], start_runs[0][0])
             + LOOSE_ARC_START_OFFSET_SAMPLES) % n
    end = refine_sector_endpoint(profiles["outer"], end_runs[-1][1])
    if end <= start:
        start, end = start_runs[0][0], end_runs[-1][1]
    if not LOOSE_ARC_SPAN_DEG[0] <= (end - start) * 360.0 / n <= LOOSE_ARC_SPAN_DEG[1]:
        return None
    coverage = np.count_nonzero(supported[start:end]) / max(end - start, 1)
    if coverage < LOOSE_ARC_MIN_INNER_LAYER_COVERAGE:
        return None
    return start, end


def trace_loose_arc_layer(profiles: dict, start: int, end: int) -> np.ndarray:
    """Follow the next electrode inward through the sector; where the loose anode
    crosses it, that electrode becomes the outermost candidate."""
    outer, inner = profiles["outer"], profiles["inner"]
    path = np.full(len(outer), np.nan)
    valid = np.flatnonzero(np.isfinite(inner[start:end])) + start
    if not len(valid):
        return path
    preferred = int(round(LOOSE_ARC_PATH_ANCHOR_DEG / 360.0 * len(outer)))
    anchor = int(valid[np.argmin(np.abs(valid - preferred))])
    path[anchor] = previous = inner[anchor]
    for index in range(anchor - 1, start - 1, -1):
        if np.isfinite(inner[index]) and abs(inner[index] - previous) <= LOOSE_ARC_PATH_MAX_STEP_PX:
            path[index] = previous = inner[index]

    previous = path[anchor]
    crossing, crossed = [], False
    for index in range(anchor + 1, end):
        if crossed:
            path[index] = outer[index]
            continue
        if np.isfinite(inner[index]) and abs(inner[index] - previous) <= LOOSE_ARC_PATH_MAX_STEP_PX:
            path[index] = previous = inner[index]
            crossing.clear()
            continue
        if np.isfinite(outer[index]) and abs(outer[index] - previous) <= LOOSE_ARC_PATH_MAX_STEP_PX:
            crossing.append(index)
            if len(crossing) >= LOOSE_ARC_CROSSING_MIN_RUN_SAMPLES:
                path[crossing] = outer[crossing]
                crossed, previous = True, outer[index]
            continue
        crossing.clear()
    return path


def trace_outer_boundary(profiles: dict, sector: tuple[int, int], geo: dict):
    """Cleaned jelly-roll and can boundary of one scan, the can mask and issues."""
    start, end = sector
    boundary = profiles["outer"].copy()
    tab = profiles["tab"]
    segment = trace_loose_arc_layer(profiles, start, end)[start:end]
    tracked = np.flatnonzero(np.isfinite(segment))
    if len(tracked):
        segment[: tracked[0]] = segment[tracked[0]]
        segment[tracked[-1] + 1:] = segment[tracked[-1]]
    boundary[start:end] = segment
    # Angles where the layer fades just before and after the sector are dropped.
    n = len(boundary)
    boundary[np.arange(start - LAYER_END_FADE_SAMPLES, start) % n] = np.nan
    boundary[np.arange(end, end + LAYER_END_FADE_SAMPLES) % n] = np.nan
    boundary = np.minimum(boundary + OUTER_BOUNDARY_INSET, profiles["can"] - 1.0)
    valid = ndi.binary_closing(
        np.tile(np.isfinite(boundary) | tab, 3),
        structure=np.ones(LOOSE_ARC_INTENSITY_MEDIAN_SIZE, dtype=bool))[n: 2 * n]
    breakpoints = [start, end]
    for run_start, run_end in true_runs(tab):
        breakpoints.extend((run_start, run_end + 1))

    can = clean_circular_boundary(profiles["can"], OUTER_ANGULAR_OUTLIER_LIMIT)
    boundary = clean_circular_boundary(boundary, JELLY_ROLL_ANGULAR_OUTLIER_LIMIT,
                                       median_size=JELLY_ROLL_ANGULAR_MEDIAN_SIZE,
                                       breakpoints=breakpoints)
    boundary = np.minimum(boundary, can - 1.0)
    issues = []
    if profiles["can_coverage"] < 0.98:
        issues.append("incomplete_can_detection")
    if np.count_nonzero(valid) / n < 0.98:
        issues.append("incomplete_jelly_roll_detection")
    if float(np.min(boundary)) <= OUTER_MIN_RADIUS or float(np.max(can)) >= OUTER_MAX_RADIUS:
        issues.append("boundary_outside_search_range")
    if float(np.mean(can - boundary)) <= 0.0 or touches_border(mask_from_boundary(boundary, geo)):
        issues.append("invalid_jelly_roll_boundary")
    return boundary, can, issues


def stabilize_loose_arc(boundaries: np.ndarray, cans: np.ndarray) -> np.ndarray:
    """Within 0 to 140 deg, replace deviations of more than 2 px from the stack
    median (each scan keeps its own radial offset) and apply a five-scan median."""
    n = boundaries.shape[1]
    active = np.zeros(n, dtype=bool)
    active[int(round(LOOSE_ARC_STABILIZATION_WINDOW_DEG[0] / 360.0 * n)):
           int(round(LOOSE_ARC_STABILIZATION_WINDOW_DEG[1] / 360.0 * n))] = True
    reference = np.median(boundaries, axis=0)
    offsets = np.median(boundaries[:, ~active] - reference[~active], axis=1)
    expected = (np.median(boundaries - offsets[:, np.newaxis], axis=0)[np.newaxis, :]
                + offsets[:, np.newaxis])
    corrected = boundaries.copy()
    outliers = active[np.newaxis, :] & (
        np.abs(corrected - expected) > LOOSE_ARC_BOUNDARY_STACK_OUTLIER_LIMIT)
    corrected[outliers] = np.broadcast_to(expected, corrected.shape)[outliers]
    filtered = ndi.median_filter(corrected, size=(LOOSE_ARC_BOUNDARY_TEMPORAL_MEDIAN_SIZE, 1),
                                 mode="nearest")
    corrected[:, active] = filtered[:, active]
    return np.minimum(corrected, cans - 1.0)


def outer_radii(images: list[np.ndarray]) -> tuple[np.ndarray, list[list[str]]]:
    """Area-equivalent radius in px of the jelly roll in every scan and the issues found."""
    geo = geometry(images[0].shape)
    profiles = [sample_outer_profiles(image, geo) for image in images]
    detected = [s for s in map(detect_loose_arc_sector, profiles) if s is not None]
    sector = (int(round(float(np.median([s[0] for s in detected])))),
              int(round(float(np.median([s[1] for s in detected])))))
    traced = [trace_outer_boundary(p, sector, geo) for p in profiles]
    cans = np.asarray([t[1] for t in traced])
    boundaries = stabilize_loose_arc(np.asarray([t[0] for t in traced]), cans)
    masks = [mask_from_boundary(b, geo) for b in boundaries]
    areas = np.asarray([np.count_nonzero(m) for m in masks])
    issues = [t[2] for t in traced]
    median_area = float(np.median(areas))
    for mask, can, area, frame_issues in zip(masks, cans, areas, issues):
        if abs(area - median_area) / median_area > MAX_OUTER_AREA_DEVIATION:
            frame_issues.append("outer_area_outlier")
        if not np.count_nonzero(mask_from_boundary(can, geo) & ~mask):
            frame_issues.append("empty_jelly_roll_can_gap")
    return np.asarray([math.sqrt(area / math.pi) for area in areas]), issues


# =============================================================================
# 4. Expansion and its voltage dependence
#    Inward expansion is the decrease of the core radius, outward expansion the
#    increase of the jelly-roll radius, both relative to the first scan. Per
#    sweep direction, every complete cycle is referenced to its own low-voltage
#    end and the cycles are averaged at their mean scan voltages.
# =============================================================================

CYCLES = (2, 3, 4, 5)
COMPLETE_CYCLE_SPAN_FRACTION = 0.8


def complete_cycles(scans: pd.DataFrame) -> np.ndarray:
    """Scans of cycles whose voltage span reaches 80 % of the widest one; the
    first and last cycle of the series are half sweeps."""
    spans = {}
    for row in scans.itertuples():
        if not math.isfinite(row.voltage_mean_V):
            continue
        values = (row.voltage_start_V, row.voltage_end_V, row.voltage_mean_V)
        low, high = spans.get(row.cycle, (math.inf, -math.inf))
        spans[row.cycle] = (min(low, min(values)), max(high, max(values)))
    widths = {c: hi - lo for c, (lo, hi) in spans.items() if math.isfinite(hi - lo)}
    widest = max(widths.values())
    complete = {c for c, w in widths.items() if w >= COMPLETE_CYCLE_SPAN_FRACTION * widest}
    return scans["cycle"].isin(complete).to_numpy()


def cycle_mean(voltage, values, cycle, selected) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and sample standard deviation over the cycles at every scan voltage."""
    curves = []
    for number in CYCLES:
        in_cycle = selected & (cycle == number)
        if np.count_nonzero(in_cycle) < 2:
            continue
        order = np.argsort(voltage[in_cycle])
        x, y = voltage[in_cycle][order], values[in_cycle][order]
        y = y - y[0]
        unique = np.unique(x)
        curves.append((unique, np.asarray([float(np.mean(y[x == u])) for u in unique])))
    grid = np.unique(np.concatenate([x for x, _ in curves]))
    mean, sd = [], []
    for v in grid:
        contributors = [float(np.interp(v, x, y)) for x, y in curves
                        if float(np.min(x)) <= v <= float(np.max(x))]
        mean.append(float(np.mean(contributors)))
        sd.append(float(np.std(contributors, ddof=1)) if len(contributors) >= 2 else math.nan)
    return grid, np.asarray(mean), np.asarray(sd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True,
                        help="folder operando_ct of the dataset")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    scans = electrical_data(args.input)
    paths = {int(re.match(r"v(\d+)", p.stem).group(1)): p
             for p in (args.input / "cross_sections").glob("v*.png")}
    images = []
    for scan in scans["scan"]:
        with Image.open(paths[scan]) as image:
            images.append(np.asarray(image.convert("L"), dtype=np.uint8))
    areas, core_issues = core_areas(images)
    outer, outer_issues = outer_radii(images)
    failed = {int(s): i + o for s, i, o in zip(scans["scan"], core_issues, outer_issues) if i + o}
    if failed:
        raise RuntimeError(f"segmentation failed validation: {failed}")

    core = np.sqrt(areas.astype(np.float64) / math.pi)
    inward, outward = core[0] - core, outer - outer[0]
    scans = scans.assign(core_radius_um=core * PIXEL_SIZE_UM, outer_radius_um=outer * PIXEL_SIZE_UM,
                         inward_um=inward * PIXEL_SIZE_UM, outward_um=outward * PIXEL_SIZE_UM,
                         total_um=(inward + outward) * PIXEL_SIZE_UM)
    args.output.mkdir(parents=True, exist_ok=True)
    scans.to_csv(args.output / "expansion_per_scan.csv", index=False)

    voltage = scans["voltage_mean_V"].to_numpy(float)
    charging = (np.isfinite(scans["voltage_start_V"]) & np.isfinite(scans["voltage_end_V"])
                & (scans["voltage_end_V"] > scans["voltage_start_V"])).to_numpy()
    usable = np.isfinite(voltage) & complete_cycles(scans)
    tables = []
    for direction, selected in (("discharge", usable & ~charging), ("charge", usable & charging)):
        table = None
        for name, values in (("inward", inward), ("outward", outward), ("total", inward + outward)):
            grid, mean, sd = cycle_mean(voltage, values, scans["cycle"].to_numpy(), selected)
            if table is None:
                table = pd.DataFrame({"direction": direction, "voltage_V": grid})
            table[f"{name}_mean_um"] = mean * PIXEL_SIZE_UM
            table[f"{name}_sd_um"] = sd * PIXEL_SIZE_UM
        tables.append(table)
    pd.concat(tables).to_csv(args.output / "expansion_vs_voltage.csv", index=False)


if __name__ == "__main__":
    main()
