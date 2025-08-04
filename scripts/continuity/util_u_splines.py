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


def position_continuity_u_splines(obj_size, is_positive, clu_num, params, irrel_params, obj_quantity, pin):
    # draw the main road
    center_point = [random.uniform(0.4, 0.6), random.uniform(0.4, 0.6)]

    line1_key_points = np.array([
        [random.uniform(0.1, 0.2), random.uniform(0.4, 0.6)],  # start point
        center_point,
        [random.uniform(0.8, 0.9), random.uniform(0.4, 0.6)]
    ])

    line2_key_points = np.array([
        [random.uniform(0.1, 0.2), random.uniform(0.1, 0.2)],  # start point
        [random.uniform(0.4, 0.6), random.uniform(0.8, 0.9)],  # start point,
        [random.uniform(0.8, 0.9), random.uniform(0.1, 0.2)]
    ])

    line_obj_num_1 = {"s": 5, "m": 7, "l": 12, "xl": 15, "xxl": 20, "xxxl": 25}.get(obj_quantity, 2)
    line_obj_num_2 = {"s": 7, "m": 10, "l": 15, "xl": 20, "xxl": 25, "xxxl": 30}.get(obj_quantity, 2)

    line1_points = get_spline_points(line1_key_points, line_obj_num_1)
    line2_points = get_spline_points(line2_key_points, line_obj_num_2)

    line1_num = len(line1_points)
    line2_num = len(line2_points)
    total_num = len(line1_points) + len(line2_points)
    group_ids = [0] * line1_num + [1] * line2_num

    logic = {"shape": ["square", "circle"], "color": ["green", "yellow"], "size": [obj_size], "count": True}
    invariant_shape = random.choice(config.all_shapes)
    invariant_color = random.choice(config.color_large_exclude_gray)
    invariant_size = obj_size
    cf_params = data_utils.get_proper_sublist(params + ["position"])
    shapes = data_utils.assign_property(is_positive, "shape", params, cf_params, irrel_params, invariant_shape, logic["shape"], config.all_shapes, total_num)
    colors = data_utils.assign_property(is_positive, "color", params, cf_params, irrel_params, invariant_color, logic["color"], config.color_large_exclude_gray, total_num)
    sizes = data_utils.assign_property(is_positive, "size", params, cf_params, irrel_params, invariant_size, logic["size"], data_utils.get_random_sizes(total_num, obj_size),
                                       total_num)

    has_position = is_positive or "position" in cf_params
    if has_position:
        positions = np.concatenate((line1_points, line2_points))
    else:
        positions = pos_utils.get_random_positions(len(line1_points) + len(line2_points), obj_size)
        group_ids = [-1] * len(positions)
    objs = encode_utils.encode_scene(positions, sizes, colors, shapes, group_ids, is_positive)
    return objs


def get_logic_rules(params):
    head = "group_target(X)"
    body = "in(O,X),in(G,X),"
    if "color" in params:
        body += "has_color(green,O),has_color(yellow,O),"
    if "size" in params:
        body += "same_obj_size(G),"
    if "shape" in params:
        body += ("has_shape(O1,square),has_shape(O2,circle),no_shape(O3,triangle),"
                 "in(O1,G),in(O2,G),in(O3,G),")
    rule = f"{head}:-{body}principle(continuity,G)."
    return rule


def non_overlap_u_splines(params, irrel_params, is_positive, clu_num, obj_quantity, pin):
    obj_size = 0.05
    objs = position_continuity_u_splines(obj_size, is_positive, clu_num, params, irrel_params, obj_quantity, pin)
    t = 0
    tt = 0
    max_try = 2000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = position_continuity_u_splines(obj_size, is_positive, clu_num, params, irrel_params, obj_quantity, pin)
        if tt > 10:
            tt = 0
            obj_size = obj_size * 0.90
        tt = tt + 1
        t = t + 1
    logic_rules = get_logic_rules(params)
    return objs, logic_rules
