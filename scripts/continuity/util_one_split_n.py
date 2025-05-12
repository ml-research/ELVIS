# Created by jing at 01.03.25
import random
import numpy as np
import math
from scripts.utils.shape_utils import overlaps, overflow

from scripts import config
from scripts.utils import pos_utils, encode_utils, data_utils


def get_main_positions(main_road_length, minx, dx, main_y):
    positions = []
    for i in range(main_road_length):
        positions.append([minx + i * dx, main_y])
    return positions


def get_split_positions(split_road_length, minx, dx, main_y, dy):
    positions = []
    for i in range(split_road_length):
        positions.append([minx + i * dx, main_y + (i + 1) * dy])
    return positions


def get_main_split_positions(split_road_length, minx, dx, main_y, dy):
    positions = []
    for i in range(split_road_length):
        positions.append([minx + i * dx, main_y - (i + 1) * dy])
    return positions


def continuity_one_splits_n(obj_size, is_positive, params, obj_quantity, prin_in_neg):
    objs = []
    main_road_length = {"s": 2, "m": 3, "l": 5}.get(obj_quantity, 2)
    split_road_length = {"s": 2, "m": 3, "l": 5}.get(obj_quantity, 2)

    dx = 0.08
    dy = 0.08
    minx = 0.1
    main_y = 0.5 + random.uniform(0, 0.1)

    if is_positive:
        if "shape" in params:
            shapes_main = [random.choice(config.bk_shapes[1:])] * (main_road_length + split_road_length)
            shapes_split = [random.choice(config.bk_shapes[1:])] * split_road_length
            shapes = shapes_main + shapes_split
        else:
            shapes_main = data_utils.random_select_unique_mix(config.bk_shapes[1:],
                                                              (main_road_length + split_road_length))
            shapes_split = data_utils.random_select_unique_mix(config.bk_shapes[1:], split_road_length)
            shapes = shapes_main + shapes_split

        if "color" in params:
            colors_main = [random.choice(config.color_large_exclude_gray)] * (main_road_length + split_road_length)
            colors_split = [random.choice(config.color_large_exclude_gray)] * split_road_length
            colors = colors_main + colors_split

        else:
            colors_main = data_utils.random_select_unique_mix(config.color_large_exclude_gray,
                                                              (main_road_length + split_road_length))
            colors_split = data_utils.random_select_unique_mix(config.color_large_exclude_gray, split_road_length)
            colors = colors_main + colors_split

        if "size" in params:
            sizes_main = [obj_size] * (main_road_length + split_road_length)
            sizes_split = [obj_size] * split_road_length
            sizes = sizes_main + sizes_split
        else:
            sizes_main = [random.uniform(obj_size * 0.6, obj_size * 1.5) for _ in
                          range((main_road_length + split_road_length))]
            sizes_split = [random.uniform(obj_size * 0.6, obj_size * 1.5) for _ in range(split_road_length)]
            sizes = sizes_main + sizes_split

        positions_main = get_main_positions(main_road_length, minx, dx, main_y)
        positions_main += get_main_split_positions(split_road_length, minx + main_road_length * dx, dx, main_y, dy)

        positions_split = get_split_positions(split_road_length, minx + main_road_length * dx, dx, main_y, dy)
        positions = positions_main + positions_split
    else:
        if "shape" in params or random.random() < 0.5:
            shapes_main = data_utils.random_select_unique_mix(config.bk_shapes[1:],
                                                              (main_road_length + split_road_length))
            shapes_split = data_utils.random_select_unique_mix(config.bk_shapes[1:], split_road_length)
            shapes = shapes_main + shapes_split
        else:
            shapes_main = [random.choice(config.bk_shapes[1:])] * (main_road_length + split_road_length)
            shapes_split = [random.choice(config.bk_shapes[1:])] * split_road_length
            shapes = shapes_main + shapes_split

        if "color" in params or random.random() < 0.5:

            colors_main = data_utils.random_select_unique_mix(config.color_large_exclude_gray,
                                                              (main_road_length + split_road_length))
            colors_split = data_utils.random_select_unique_mix(config.color_large_exclude_gray, split_road_length)
            colors = colors_main + colors_split
        else:
            colors_main = [random.choice(config.color_large_exclude_gray)] * (main_road_length + split_road_length)
            colors_split = [random.choice(config.color_large_exclude_gray)] * split_road_length
            colors = colors_main + colors_split
        if "size" in params or random.random() < 0.5:
            sizes_main = [random.uniform(obj_size * 0.6, obj_size * 1.5) for _ in
                          range((main_road_length + split_road_length))]
            sizes_split = [random.uniform(obj_size * 0.6, obj_size * 1.5) for _ in range(split_road_length)]
            sizes = sizes_main + sizes_split
        else:
            sizes_main = [obj_size] * (main_road_length + split_road_length)
            sizes_split = [obj_size] * split_road_length
            sizes = sizes_main + sizes_split

        if prin_in_neg:
            positions_main = get_main_positions(main_road_length, minx, dx, main_y)
            positions_main += get_main_split_positions(split_road_length, minx + main_road_length * dx, dx, main_y, dy)

            positions_split = get_split_positions(split_road_length, minx + main_road_length * dx, dx, main_y, dy)
            positions = positions_main + positions_split
        else:
            positions = pos_utils.get_random_positions(main_road_length + split_road_length, obj_size)

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
    return objs


def non_overlap_one_split_n(params, is_positive, clu_num, obj_quantity, prin_in_neg=True):
    obj_size = 0.05

    objs = continuity_one_splits_n(obj_size, is_positive, params, obj_quantity, prin_in_neg=prin_in_neg)
    t = 0
    tt = 0
    max_try = 1000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = continuity_one_splits_n(obj_size, is_positive, params, obj_quantity, prin_in_neg=prin_in_neg)
        if tt > 10:
            tt = 0
            obj_size = obj_size * 0.90
        tt = tt + 1
        t = t + 1
    return objs
