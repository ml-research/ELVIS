# Created by jing at 01.03.25

import random
import math
import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import make_interp_spline, interp1d

from scripts import config
from scripts.utils import encode_utils, data_utils, pos_utils
from scripts.utils.shape_utils import overlaps, overflow
from scripts.utils.pos_utils import get_feature_triangle_positions


def feature_closure_triangle(is_positive, clu_num, params, irrel_params, pin):
    obj_num = 3
    # Generate random anchors for clusters ensuring proper distance
    group_anchors = []
    for _ in range(clu_num):
        group_anchors.append(pos_utils.generate_random_anchor(group_anchors, 0.25, 0.25, 0.75, 0.45, 0.95))
    clu_size = ({1: 0.4 + random.random() * 0.1,
                 2: 0.3 + random.random() * 0.1,
                 3: 0.3 + random.random() * 0.1,
                 4: 0.3 + random.random() * 0.1
                 }.get(clu_num, 0.3))
    obj_size = clu_size * (0.3 + random.random() * 0.1)

    color_val = random.choice(config.color_large_exclude_gray)
    size_val = obj_size
    objs = []
    for i in range(clu_num):
        positions = get_feature_triangle_positions(group_anchors[i], clu_size)
        shapes = ["pac_man"] * obj_num
        # 50% of the negative images, random object positions but other properties as same as positive
        is_random = False
        if not is_positive and pin and random.random() > 0.5:
            start_angles = random.sample(range(0, 360), 3)
            end_angles = [angle + 300 for angle in start_angles]
            is_positive = True
            is_random = True
        else:
            start_angles = [120, 0, 240]
            end_angles = [angle + 300 for angle in start_angles]

        if is_positive:
            if "color" in params or random.random() < 0.5:
                colors = random.choices(["blue", "red"], k=obj_num)
            else:
                colors = data_utils.random_select_unique_mix(config.color_large_exclude_gray, obj_num)
            if "color" in irrel_params:
                colors = [color_val] * obj_num

            if "size" in params:
                sizes = [obj_size] * obj_num
            else:
                sizes = data_utils.get_random_sizes(obj_num, obj_size)

            if "size" in irrel_params:
                sizes = [size_val] * obj_num
        else:
            cf_params = data_utils.get_proper_sublist(params)
            # if "position" not in cf_params and random.random() < 0.5:
            #     positions = pos_utils.get_random_positions(obj_num, obj_size)
            if "color" in cf_params:
                colors = random.choices(["blue", "red"], k=obj_num)
            else:
                colors = data_utils.random_select_unique_mix(config.color_large_exclude_gray, obj_num)
            if "color" in irrel_params:
                colors = [color_val] * obj_num

            if "size" not in cf_params:
                sizes = data_utils.get_random_sizes(obj_num, obj_size)
            else:
                sizes = [obj_size] * obj_num
            if "size" in irrel_params:
                sizes = [size_val] * obj_num

        group_id = -1 if is_random else i
        objs += encode_utils.encode_scene(positions, sizes, colors, shapes,
                                          [group_id] * len(positions), is_positive, start_angles, end_angles)

    return objs


def get_logic_rules(params):
    head = "group_target(X)"
    body = "in(O,X),in(G,X),"
    if "color" in params:
        body += "has_color(blue,O),has_color(red,O),"
    if "size" in params:
        body += "same_obj_size(G),"
    rule = f"{head}:-{body}group_shape(triangle,G),principle(closure,G)."
    return rule

def non_overlap_feature_triangle(params, irrel_params, is_positive, clu_num, pin):
    objs = feature_closure_triangle(is_positive, clu_num, params, irrel_params, pin)
    t = 0
    tt = 0
    max_try = 2000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = feature_closure_triangle(is_positive, clu_num, params, irrel_params, pin)
        if tt > 10:
            tt = 0
        tt = tt + 1
        t = t + 1
    logic_rules = get_logic_rules(params)

    return objs, logic_rules
