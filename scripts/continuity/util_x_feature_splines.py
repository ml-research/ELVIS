# Created by jing at 01.03.25

import random
import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import make_interp_spline, interp1d

from scripts import config
from scripts.utils import encode_utils, data_utils, pos_utils
from scripts.utils.pos_utils import get_spline_points


def get_shaded_points(given_points, dx, dy):
    shaded_points = []
    for point in given_points:
        shaded_points.append([point[0] + dx, point[1] + dy])
    return shaded_points


def feature_continuity_x_splines(params, is_positive, clu_num, obj_quantity, pin):
    objs = []
    obj_size = 0.05

    # draw the main road
    center_point = [random.uniform(0.4, 0.6), random.uniform(0.4, 0.6)]

    line1_key_points = np.array([
        [random.uniform(0.1, 0.2), random.uniform(0.1, 0.2)],  # start point
        center_point,
        [random.uniform(0.8, 0.9), random.uniform(0.8, 0.9)]
    ])

    line2_key_points = np.array([
        [random.uniform(0.1, 0.2), random.uniform(0.8, 0.9)],  # start point
        center_point,
        [random.uniform(0.8, 0.9), random.uniform(0.1, 0.2)]
    ])
    dx = random.uniform(0.005, 0.03)
    dy = random.uniform(-0.03, -0.005)

    line1_num = {"s": 5, "m": 7, "l": 12}.get(obj_quantity, 2)
    line2_num = {"s": 7, "m": 10, "l": 15}.get(obj_quantity, 2)

    line1_points = get_spline_points(line1_key_points, line1_num)
    line1_points_shade = get_shaded_points(line1_points, dx, dy)
    line2_points = get_spline_points(line2_key_points, line2_num)

    if is_positive:
        if "shape" in params or random.random() < 0.5:
            shapes = [random.choice(config.bk_shapes[1:])] * line1_num * 2
            shapes += [random.choice(config.bk_shapes[1:])] * line2_num
        else:
            shapes = data_utils.random_select_unique_mix(config.bk_shapes[1:], line1_num * 2)
            shapes += data_utils.random_select_unique_mix(config.bk_shapes[1:], line2_num)

        if "color" in params or random.random() < 0.5:
            colors = [random.choice(config.color_large_exclude_gray)] * line1_num * 2
            colors += [random.choice(config.color_large_exclude_gray)] * line2_num

        else:
            colors = data_utils.random_select_unique_mix(config.color_large_exclude_gray, line1_num * 2)
            colors += data_utils.random_select_unique_mix(config.color_large_exclude_gray, line2_num)
        if "size" in params or random.random() < 0.5:
            sizes = [obj_size] * line1_num * 2
            sizes += [obj_size] * line2_num
        else:
            sizes = [random.uniform(obj_size * 0.6, obj_size * 1.5) for _ in range(line1_num * 2)]
            sizes += [random.uniform(obj_size * 0.6, obj_size * 1.5) for _ in range(line2_num)]

        positions = np.concatenate((line1_points, line1_points_shade, line2_points))
    else:
        if "shape" in params or random.random() < 0.5:
            shapes = data_utils.random_select_unique_mix(config.bk_shapes[1:], line1_num * 2)
            shapes += data_utils.random_select_unique_mix(config.bk_shapes[1:], line2_num)
        else:
            shapes = [random.choice(config.bk_shapes[1:])] * line1_num * 2
            shapes += [random.choice(config.bk_shapes[1:])] * line2_num
        if "color" in params or random.random() < 0.5:
            colors = data_utils.random_select_unique_mix(config.color_large_exclude_gray, line1_num * 2)
            colors += data_utils.random_select_unique_mix(config.color_large_exclude_gray, line2_num)
        else:
            colors = [random.choice(config.color_large_exclude_gray)] * line1_num * 2
            colors += [random.choice(config.color_large_exclude_gray)] * line2_num
        if "size" in params or random.random() < 0.5:
            sizes = [random.uniform(obj_size * 0.6, obj_size * 1.5) for _ in range(line1_num * 2)]
            sizes += [random.uniform(obj_size * 0.6, obj_size * 1.5) for _ in range(line2_num)]

        else:
            sizes = [obj_size] * line1_num * 2
            sizes += [obj_size] * line2_num
        if pin:
            positions = np.concatenate((line1_points, line1_points_shade, line2_points))
        else:
            positions = pos_utils.get_random_positions(len(line1_points) + len(line1_points_shade) + line2_points,
                                                       obj_size)
    try:
        for i in range(len(positions)):
            objs.append(encode_utils.encode_objs(
                x=positions[i][0],
                y=positions[i][1],
                size=sizes[i],
                color=colors[i],
                shape=shapes[i],
                line_width=-1,
                solid=True
            ))
    except Exception as e:
        raise e
    return objs
