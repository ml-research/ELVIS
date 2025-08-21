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


def feature_continuity_overlap_splines(params, irrel_params, is_positive, clu_num, obj_quantity, pin):
    # Set spline length and object size based on object quantity
    obj_num_map = {"s": 5, "m": 7, "l": 12, "xl": 17, "xxl": 21, "xxxl": 25}
    line_obj_num = obj_num_map.get(obj_quantity, 2)
    # Spline length: shorter for fewer objects, longer for more
    min_length_map = {5: 0.3, 7: 0.3, 12: 0.6, 17: 0.7, 21: 0.9}
    min_length = min_length_map.get(line_obj_num, 0.4)
    # Object size: larger for fewer objects, smaller for more
    obj_size_map = {5: 0.04, 7: 0.05, 12: 0.05, 17: 0.03, 21: 0.04, 25: 0.035}
    obj_size = obj_size_map.get(line_obj_num, 0.05)

    cf_params = data_utils.get_proper_sublist(params + ["continuity"])
    dx = random.uniform(0.005, 0.02)
    dy = random.uniform(-0.02, -0.005)
    logic = {"shape": ["square", "circle"], "color": ["green", "yellow"], "size": [obj_size], "count": True}
    invariant_shape = random.choice(config.all_shapes)
    invariant_color = random.choice(config.color_large_exclude_gray)
    invariant_size = obj_size

    all_spline_points = []
    group_ids = []
    total_num = 0
    endpoints = []
    min_dist = 0.25

    for i in range(clu_num):
        valid = False
        while not valid:
            start = [random.uniform(0, 1), random.uniform(0, 1)]
            end = [random.uniform(0, 1), random.uniform(0, 1)]
            center_point = [random.uniform(0.3, 0.7), random.uniform(0.3, 0.7)]
            # Use dynamic min_length
            if np.linalg.norm(np.array(start) - np.array(end)) < min_length:
                continue
            valid = True
            for prev_start, prev_end in endpoints:
                if (np.linalg.norm(np.array(start) - np.array(prev_start)) < min_dist or
                        np.linalg.norm(np.array(start) - np.array(prev_end)) < min_dist or
                        np.linalg.norm(np.array(end) - np.array(prev_start)) < min_dist or
                        np.linalg.norm(np.array(end) - np.array(prev_end)) < min_dist):
                    valid = False
                    break
        key_points = np.array([start, center_point, end])
        spline_points = get_spline_points(key_points, line_obj_num)
        all_spline_points.append(spline_points)
        group_ids.extend([i] * line_obj_num)
        total_num += line_obj_num
        endpoints.append((start, end))
        if i == 0:
            shaded_points = get_shaded_points(spline_points, dx, dy)
            all_spline_points.append(np.array(shaded_points))
            group_ids.extend([i] * line_obj_num)
            total_num += line_obj_num

    shapes = data_utils.assign_property(is_positive, "shape", params, cf_params, irrel_params, invariant_shape, logic["shape"], config.all_shapes, total_num)
    colors = data_utils.assign_property(is_positive, "color", params, cf_params, irrel_params, invariant_color, logic["color"], config.color_large_exclude_gray, total_num)
    sizes = data_utils.assign_property(is_positive, "size", params, cf_params, irrel_params, invariant_size, logic["size"], data_utils.get_random_sizes(total_num, obj_size), total_num)

    has_position = is_positive or "continuity" in cf_params
    if has_position:
        positions = np.concatenate(all_spline_points)
    else:
        positions = pos_utils.get_random_positions(total_num, obj_size)
        group_ids = [-1] * len(positions)
    objs = encode_utils.encode_scene(positions, sizes, colors, shapes, group_ids, is_positive)
    logic_rules = get_logic_rules(is_positive, params, cf_params, irrel_params)
    return objs, logic_rules