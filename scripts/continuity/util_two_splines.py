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


def position_continuity_two_splines(obj_size, is_positive, clu_num, params, obj_quantity, pin):
    objs = []

    # draw the main road
    center_point = [random.uniform(0.4, 0.6), random.uniform(0.4, 0.6)]

    line1_key_points = np.array([
        [random.random() * 0.1 + 0.1, random.random() * 0.1 + 0.7],  # start point
        center_point,
        [random.uniform(0.7, 0.8), random.uniform(0.1, 0.3)]
    ])

    line2_key_points = np.array([
        [random.random() * 0.1 + 0.1, random.uniform(0.1, 0.3)],  # start point
        center_point,
        [random.uniform(0.7, 0.8), random.uniform(0.7, 0.8)]
    ])

    line_obj_num = {"s": 5, "m": 7, "l": 12}.get(obj_quantity, 2)

    line1_points = get_spline_points(line1_key_points, line_obj_num)
    line2_points = get_spline_points(line2_key_points, line_obj_num)
    group_ids = [0] * len(line1_points) + [1] * len(line2_points)
    is_random = False
    if is_positive:
        if "shape" in params or random.random() < 0.5:
            shapes = [random.choice(config.bk_shapes[1:])] * line_obj_num * 2
        else:
            shapes = data_utils.random_select_unique_mix(config.bk_shapes[1:], line_obj_num * 2)

        if "color" in params or random.random() < 0.5:
            colors = [random.choice(config.color_large_exclude_gray)] * line_obj_num * 2
        else:
            colors = data_utils.random_select_unique_mix(config.color_large_exclude_gray, line_obj_num * 2)

        if "size" in params or random.random() < 0.5:
            sizes = [obj_size] * line_obj_num * 2
        else:
            sizes = [random.uniform(obj_size * 0.6, obj_size * 1.5) for _ in
                     range(line_obj_num * 2)]

        positions = np.concatenate((line1_points, line2_points))
    else:
        if "shape" in params or random.random() < 0.5:
            shapes = data_utils.random_select_unique_mix(config.bk_shapes[1:], line_obj_num * 2)
        else:
            shapes = [random.choice(config.bk_shapes[1:])] * line_obj_num * 2

        if "color" in params or random.random() < 0.5:
            colors = data_utils.random_select_unique_mix(config.color_large_exclude_gray, line_obj_num * 2)
        else:
            colors = [random.choice(config.color_large_exclude_gray)] * line_obj_num * 2

        if "size" in params or random.random() < 0.5:
            sizes = [random.uniform(obj_size * 0.6, obj_size * 1.5) for _ in range(line_obj_num * 2)]
        else:
            sizes = [obj_size] * line_obj_num * 2
        if pin:
            positions = np.concatenate((line1_points, line2_points))
        else:
            is_random = True
            positions = pos_utils.get_random_positions(len(line1_points) + len(line2_points), obj_size)
    try:
        for i in range(len(positions)):
            if is_random:
                group_id = -1
            else:
                group_id = group_ids[i]
            objs.append(encode_utils.encode_objs(
                x=positions[i][0],
                y=positions[i][1],
                size=sizes[i],
                color=colors[i],
                shape=shapes[i],
                line_width=-1,
                solid=True,
                group_id=group_id,
            ))
    except Exception as e:
        raise e
    return objs


def non_overlap_two_splines(params, is_positive, clu_num, obj_quantity, pin=True):
    obj_size = 0.05
    objs = position_continuity_two_splines(obj_size, is_positive, clu_num, params, obj_quantity, pin=pin)
    t = 0
    tt = 0
    max_try = 1000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = position_continuity_two_splines(obj_size, is_positive, clu_num, params, obj_quantity, pin=pin)
        if tt > 10:
            tt = 0
            obj_size = obj_size * 0.90
        tt = tt + 1
        t = t + 1
    return objs
