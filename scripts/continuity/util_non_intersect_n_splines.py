# Created by jing at 01.03.25

import random
import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import make_interp_spline, interp1d

from scripts import config
from scripts.utils import encode_utils, data_utils, pos_utils
from scripts.utils.shape_utils import overlaps, overflow

def get_spline_keypoints(line_obj_num, x_min, x_max):
    # Distance factor increases with object count
    base_dist = 0.15
    dist_factor = base_dist + 0.02 * line_obj_num
    # Random endpoints within band
    start = [random.uniform(x_min, x_max), random.uniform(0.1, 0.9)]
    end = [random.uniform(x_min, x_max), random.uniform(0.1, 0.9)]
    # Center point is offset from midpoint by dist_factor
    midpoint = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2]
    direction = np.array(end) - np.array(start)
    norm_dir = direction / (np.linalg.norm(direction) + 1e-6)
    # Offset perpendicular to the line
    perp = np.array([-norm_dir[1], norm_dir[0]])
    center_point = midpoint + perp * dist_factor
    return np.array([start, center_point.tolist(), end])


def get_spline_points(points, n):
    # Separate the points into x and y coordinates
    x = points[:, 0]
    y = points[:, 1]
    # Generate a smooth spline curve (use k=3 for cubic spline interpolation)
    # Spline interpolation
    spl_x = make_interp_spline(np.linspace(0, 1, len(x)), x, k=2)
    spl_y = make_interp_spline(np.linspace(0, 1, len(y)), y, k=2)

    # Dense sampling to approximate arc-length
    dense_t = np.linspace(0, 1, 1000)
    dense_x, dense_y = spl_x(dense_t), spl_y(dense_t)

    # Calculate cumulative arc length
    arc_lengths = np.sqrt(np.diff(dense_x) ** 2 + np.diff(dense_y) ** 2)
    cum_arc_length = np.insert(np.cumsum(arc_lengths), 0, 0)

    # Interpolate to find points equally spaced by arc-length
    equal_distances = np.linspace(0, cum_arc_length[-1], n)
    interp_t = interp1d(cum_arc_length, dense_t)(equal_distances)

    # Get equally spaced points
    equal_x, equal_y = spl_x(interp_t), spl_y(interp_t)

    positions = np.stack([equal_x, equal_y], axis=-1)
    return positions


def get_band_mask(points, band_idx, band_width, slope_angle, clu_num):
    # Project points onto the slope direction
    theta = np.deg2rad(slope_angle)
    direction = np.array([np.cos(theta), np.sin(theta)])
    projections = np.dot(points, direction)
    band_start = band_idx * band_width
    band_end = (band_idx + 1) * band_width
    return (projections >= band_start) & (projections < band_end)

def get_spline_keypoints_in_band(line_obj_num, band_idx, band_width, slope_angle, clu_num, th=1, curve_prob=0.5):
    theta = np.deg2rad(slope_angle)
    direction = np.array([np.cos(theta), np.sin(theta)])
    max_tries = 50000
    margin = 0.02
    min_end_dist = 0.2
    tries = 0
    while True:
        pts = np.random.uniform(0.0, 1.0, (3, 2))
        projections = np.dot(pts, direction)
        band_start = band_idx * band_width - margin
        band_end = (band_idx + 1) * band_width + margin
        start, end = pts[0], pts[2]
        diff_x = abs(start[0] - end[0])
        diff_y = abs(start[1] - end[1])
        if (
            np.all((projections >= band_start) & (projections < band_end))
            and np.linalg.norm(start - end) >= min_end_dist
            and (diff_x + diff_y) < th
        ):
            break
        tries += 1
        if tries > max_tries:
            raise RuntimeError("Failed to sample keypoints in band")
    midpoint = (start + end) / 2
    line_dir = end - start
    norm_dir = line_dir / (np.linalg.norm(line_dir) + 1e-6)
    perp = np.array([-norm_dir[1], norm_dir[0]])
    base_dist = 0.15
    dist_factor = base_dist + 0.015 * line_obj_num
    # Randomly decide if the spline is straight/slightly curved or U-shaped
    if random.random() < curve_prob:
        center_point = midpoint + perp * dist_factor * random.uniform(0.0, 0.5)  # Slight curve
    else:
        center_point = midpoint  # Straight line
    return np.array([start, center_point, end])

def position_continuity_non_intersected_n_splines(obj_size, is_positive, clu_num, params, irrel_params, cf_params, obj_quantity, pin):
    line_obj_num = {"s": 5, "m": 7, "l": 9, "xl": 11, "xxl": 14, "xxxl": 18}.get(obj_quantity, 2)
    logic = {"shape": ["square", "circle"], "color": ["green", "yellow"], "size": [obj_size], "count": True}
    invariant_shape = random.choice(config.all_shapes)
    invariant_color = random.choice(config.color_large_exclude_gray)
    invariant_size = obj_size

    all_spline_points = []
    group_ids = []
    min_spline_dist = 0.1

    # Random slope angle between 0 and 90 degrees
    slope_angle = random.uniform(0, 90)
    band_width = 1.0 / clu_num

    for i in range(clu_num):
        valid = False
        tries = 0
        while not valid and tries < 1000:
            key_points = get_spline_keypoints_in_band(line_obj_num, i, band_width, slope_angle, clu_num)
            spline_points = get_spline_points(key_points, line_obj_num)
            if not np.all((spline_points >= 0) & (spline_points <= 1)):
                tries += 1
                continue
            if all(
                np.min(cdist(spline_points, prev_spline)) > min_spline_dist
                for prev_spline in all_spline_points
            ):
                valid = True
            tries += 1
        all_spline_points.append(spline_points)
        group_ids.extend([i] * line_obj_num)

    logic_obj_num = line_obj_num * clu_num
    shapes = data_utils.assign_property(is_positive, "shape", params, cf_params, irrel_params, invariant_shape, logic["shape"], config.all_shapes, logic_obj_num)
    colors = data_utils.assign_property(is_positive, "color", params, cf_params, irrel_params, invariant_color, logic["color"], config.color_large_exclude_gray, logic_obj_num)
    sizes = data_utils.assign_property(is_positive, "size", params, cf_params, irrel_params, invariant_size, logic["size"], data_utils.get_random_sizes(logic_obj_num, obj_size), logic_obj_num)

    has_position = is_positive or "continuity" in cf_params
    if has_position:
        positions = np.concatenate(all_spline_points)
    else:
        positions = pos_utils.get_random_positions(logic_obj_num, obj_size)
        group_ids = [-1] * len(positions)
    objs = encode_utils.encode_scene(positions, sizes, colors, shapes, group_ids, is_positive)
    return objs


def get_logic_rules(is_positive, params, cf_params, irrel_params):
    head = "group_target(X)"
    body = "in(O,X),in(G,X),"
    if "color" in params:
        body += "has_color(blue,O),has_color(red,O),"
    if "size" in params:
        body += "same_obj_size(G),"
    if "shape" in params:
        body += ("has_shape(O1,square),has_shape(O2,circle),no_shape(O3,triangle),"
                 "in(O1,G),in(O2,G),in(O3,G),")
    rule = f"{head}:-{body}principle(continuity,G)."
    logic = {
        "rule": rule,
        "is_positive": is_positive,
        "fixed_props": params,
        "cf_params": cf_params,
        "irrel_params": irrel_params,
        "principle": "continuity"
    }
    return logic


def non_intersected_n_splines(params, irrel_params, is_positive, clu_num, obj_quantity, pin=True):
    obj_size = 0.05
    cf_params = data_utils.get_proper_sublist(params + ["continuity"])
    objs = position_continuity_non_intersected_n_splines(obj_size, is_positive, clu_num, params, irrel_params, cf_params, obj_quantity, pin=pin)
    t = 0
    tt = 0
    max_try = 1000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = position_continuity_non_intersected_n_splines(obj_size, is_positive, clu_num, params, irrel_params, cf_params, obj_quantity, pin=pin)
        if tt > 10:
            tt = 0
            obj_size = obj_size * 0.90
        tt = tt + 1
        t = t + 1
    logic_rules = get_logic_rules(is_positive, params, cf_params, irrel_params)

    return objs, logic_rules
