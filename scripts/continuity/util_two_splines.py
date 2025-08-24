# Created by jing at 01.03.25

import random
import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import make_interp_spline, interp1d

from scripts import config
from scripts.utils import encode_utils, data_utils, pos_utils
from scripts.utils.shape_utils import overlaps, overflow


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


def position_continuity_n_splines(obj_size, is_positive, clu_num, params, irrel_params, cf_params, obj_quantity, pin):
    # Number of objects per spline
    line_obj_num = {"s": 5, "m": 7, "l": 12, "xl": 17, "xxl": 21, "xxxl": 25}.get(obj_quantity, 2)
    logic = {"shape": ["square", "circle"], "color": ["green", "yellow"], "size": [obj_size], "count": True}
    invariant_shape = random.choice(config.all_shapes)
    invariant_color = random.choice(config.color_large_exclude_gray)
    invariant_size = obj_size

    center_point = [random.uniform(0.4, 0.6), random.uniform(0.4, 0.6)]
    all_spline_points = []
    group_ids = []

    for i in range(clu_num):
        valid = False
        while not valid:
            start = [random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)]
            end = [random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)]
            key_points = np.array([start, center_point, end])
            spline_points = get_spline_points(key_points, line_obj_num)
            # Check if all points are inside [0,1]
            if np.all((spline_points >= 0) & (spline_points <= 1)):
                valid = True
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


def with_intersected_n_splines(params, irrel_params, is_positive, clu_num, obj_quantity, pin=True):
    obj_size = 0.05
    cf_params = data_utils.get_proper_sublist(params + ["continuity"])
    objs = position_continuity_n_splines(obj_size, is_positive, clu_num, params, irrel_params, cf_params, obj_quantity, pin=pin)
    t = 0
    tt = 0
    max_try = 1000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = position_continuity_n_splines(obj_size, is_positive, clu_num, params, irrel_params, cf_params, obj_quantity, pin=pin)
        if tt > 10:
            tt = 0
            obj_size = obj_size * 0.90
        tt = tt + 1
        t = t + 1
    logic_rules = get_logic_rules(is_positive, params, cf_params, irrel_params)

    return objs, logic_rules
