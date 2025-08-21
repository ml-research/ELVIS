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


def continuity_one_splits_n(clu_num, obj_size, is_positive, params, irrel_params, obj_quantity, prin_in_neg):
    main_road_length = int(config.standard_quantity_dict[obj_quantity] * 0.4)
    split_road_length = int(config.standard_quantity_dict[obj_quantity] * 0.4)
    logic = {
        "shape": ["square", "circle"],
        "color": ["green", "yellow"],
        "size": [obj_size],
        "count": True
    }
    dx = 0.1
    dy = 0.1
    center_x = 0.5
    center_y = 0.5
    invariant_shape = random.choice(config.all_shapes)
    invariant_color = random.choice(config.color_large_exclude_gray)
    invariant_size = obj_size

    total_main = main_road_length + split_road_length * clu_num
    total_split = split_road_length * clu_num

    cf_params = data_utils.get_proper_sublist(params + ["continuity"])

    # shapes_main = data_utils.assign_property(is_positive, "shape", params, cf_params, irrel_params, invariant_shape, logic["shape"], config.all_shapes, total_main)
    # shapes_split = data_utils.assign_property(is_positive, "shape", params, cf_params, irrel_params, invariant_shape, logic["shape"], config.all_shapes, total_split)
    # shapes = shapes_main + shapes_split

    shapes_main = data_utils.assign_property(is_positive, "shape", params, cf_params, irrel_params, invariant_shape, logic["shape"], config.all_shapes, main_road_length)
    shapes_split = data_utils.assign_property(is_positive, "shape", params, cf_params, irrel_params, invariant_shape, logic["shape"], config.all_shapes, total_split)
    shapes = shapes_main + shapes_split

    colors_main = data_utils.assign_property(is_positive, "color", params, cf_params, irrel_params, invariant_color, logic["color"], config.color_large_exclude_gray,
                                             main_road_length)
    colors_split = data_utils.assign_property(is_positive, "color", params, cf_params, irrel_params, invariant_color, logic["color"], config.color_large_exclude_gray, total_split)
    colors = colors_main + colors_split

    # colors_main = data_utils.assign_property(is_positive, "color", params, cf_params, irrel_params, invariant_color, logic["color"], config.color_large_exclude_gray, total_main)
    # colors_split = data_utils.assign_property(is_positive, "color", params, cf_params, irrel_params, invariant_color, logic["color"], config.color_large_exclude_gray, total_split)
    # colors = colors_main + colors_split

    sizes_main = data_utils.assign_property(is_positive, "size", params, cf_params, irrel_params, invariant_size, logic["size"],
                                            data_utils.get_random_sizes(main_road_length, obj_size), main_road_length)
    sizes_split = data_utils.assign_property(is_positive, "size", params, cf_params, irrel_params, invariant_size, logic["size"],
                                             data_utils.get_random_sizes(total_split, obj_size), total_split)
    sizes = sizes_main + sizes_split

    #
    # sizes_main = data_utils.assign_property(is_positive, "size", params, cf_params, irrel_params, invariant_size, logic["size"], data_utils.get_random_sizes(total_main, obj_size),
    #                                         total_main)
    # sizes_split = data_utils.assign_property(is_positive, "size", params, cf_params, irrel_params, invariant_size, logic["size"],
    #                                          data_utils.get_random_sizes(total_split, obj_size),
    #                                          total_split)
    # sizes = sizes_main + sizes_split

    # has_position = is_positive or "continuity" in cf_params
    # if has_position:
    #     positions_main = get_main_positions(main_road_length, minx, dx, main_y)
    #     position_split_1 = get_main_split_positions(split_road_length, minx + main_road_length * dx, dx, main_y, dy)
    #     position_split_2 = get_split_positions(split_road_length, minx + main_road_length * dx, dx, main_y, dy)
    #     positions = positions_main + position_split_1 + position_split_2
    #     group_ids = [0] * len(positions_main) + [1] * len(position_split_1) + [2] * len(position_split_2)
    # else:
    #     positions = pos_utils.get_random_positions(main_road_length + split_road_length, obj_size)
    #     group_ids = [-1] * len(positions)
    # objs = encode_utils.encode_scene(positions, sizes, colors, shapes, group_ids, is_positive)
    # return objs

    has_position = is_positive or "continuity" in cf_params
    if has_position:
        main_angle = random.uniform(math.pi - math.pi / 8, math.pi + math.pi / 8)  # 180°, left

        # main_angle = random.uniform(-math.pi/4, math.pi/4)
        dx_main = dx * math.cos(main_angle)
        dy_main = dx * math.sin(main_angle)
        positions_main = []
        for i in range(main_road_length):
            x = center_x + i * dx_main
            y = center_y + i * dy_main
            positions_main.append([x, y])
        split_start_x = center_x
        split_start_y = center_y
        split_positions = []
        group_ids = [0] * len(positions_main)
        # Evenly spread angles from -45° to +45° (rightward)
        if clu_num == 1:
            angles = [math.pi]
        else:
            angles = [(-math.pi * 0.125) + i * (math.pi * 0.75) / (clu_num - 1) for i in range(clu_num)]
        for idx, angle in enumerate(angles):
            dx_split = dx * math.cos(angle)
            dy_split = dy * math.sin(angle)
            for j in range(1, split_road_length):
                x = split_start_x + j * dx_split
                y = split_start_y + j * dy_split
                split_positions.append([x, y])
                group_ids.append(idx + 1)
        positions = positions_main + split_positions
    else:
        positions = pos_utils.get_random_positions(main_road_length + split_road_length * clu_num, obj_size)
        group_ids = [-1] * len(positions)
    objs = encode_utils.encode_scene(positions, sizes, colors, shapes, group_ids, is_positive)
    return objs


def get_logic_rules(is_positive, params, cf_params, irrel_params):
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

    logic = {
        "rule": rule,
        "is_positive": is_positive,
        "fixed_props": params,
        "cf_params": cf_params,
        "irrel_params": irrel_params,
        "principle": "continuity"
    }
    return logic


def non_overlap_one_split_n(params, irrel_params, is_positive, clu_num, obj_quantity, prin_in_neg=True):
    obj_size = 0.05
    cf_params = data_utils.get_proper_sublist(params + ["continuity"])
    objs = continuity_one_splits_n(clu_num, obj_size, is_positive, params, irrel_params, obj_quantity, prin_in_neg=prin_in_neg)
    t = 0
    tt = 0
    max_try = 1000
    # while (overlaps(objs) or overflow(objs)) and (t < max_try):
    #     objs = continuity_one_splits_n(obj_size, is_positive, params, irrel_params, obj_quantity, prin_in_neg=prin_in_neg)
    #     if tt > 10:
    #         tt = 0
    #         obj_size = obj_size * 0.90
    #     tt = tt + 1
    #     t = t + 1
    logic_rules = get_logic_rules(is_positive, params, cf_params, irrel_params)
    return objs, logic_rules
