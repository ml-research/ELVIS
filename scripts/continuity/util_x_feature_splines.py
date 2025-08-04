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


def feature_continuity_x_splines(params, irrel_params, is_positive, clu_num, obj_quantity, pin):
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

    line1_num = {"s": 5, "m": 7, "l": 12, "xl": 17, "xxl": 21, "xxxl": 25}.get(obj_quantity, 2)
    line2_num = {"s": 5, "m": 7, "l": 12, "xl": 17, "xxl": 21, "xxxl": 25}.get(obj_quantity, 2) + 2
    total_num = line1_num * 2 + line2_num

    line1_points = get_spline_points(line1_key_points, line1_num)
    line1_points_shade = get_shaded_points(line1_points, dx, dy)
    line2_points = get_spline_points(line2_key_points, line2_num)
    group_ids = [0] * line1_num * 2 + [1] * line2_num
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
        positions = np.concatenate((line1_points, line1_points_shade, line2_points))
    else:
        positions = pos_utils.get_random_positions(len(line1_points) + len(line2_points) + len(line1_points_shade), obj_size)
        group_ids = [-1] * len(positions)
    objs = encode_utils.encode_scene(positions, sizes, colors, shapes, group_ids, is_positive)
    logic_rules = get_logic_rules(params)
    return objs, logic_rules
