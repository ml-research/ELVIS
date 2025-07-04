# Created by jing at 01.03.25
import math

import random
import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import make_interp_spline, interp1d

from scripts import config
from scripts.utils import encode_utils, data_utils, pos_utils
from scripts.utils.shape_utils import overlaps, overflow


def generate_random_anchor(existing_anchors):
    cluster_dist = 0.1  # Increased to ensure clear separation
    while True:
        anchor = [random.uniform(0.4, 0.7), random.uniform(0.4, 0.7)]
        if all(pos_utils.euclidean_distance(anchor, existing) > cluster_dist for existing in existing_anchors):
            return anchor


def closure_big_triangle(obj_size, is_positive, clu_num, params, obj_quantity, pin):
    objs = []
    positions = []
    # Generate random anchors for clusters ensuring proper distance
    group_anchors = []
    for _ in range(clu_num):
        group_anchors.append(generate_random_anchor(group_anchors))
    group_ids = []
    for i in range(clu_num):
        x = group_anchors[i][0]
        y = group_anchors[i][1]
        positions += pos_utils.get_triangle_positions(obj_quantity, x, y)
        group_ids += [i]*len(positions)
    obj_num = len(positions)

    # 30% of the negative images, random object positions but other properties as same as positive
    is_random = False
    if not is_positive and pin and random.random() < 0.3:
        positions = pos_utils.get_random_positions(obj_num, obj_size)
        is_positive = True
        is_random = True
    if is_positive:
        if "shape" in params or random.random() < 0.5:
            shapes = random.choices(["square", "circle"], k=obj_num)
        else:
            if random.random() < 0.5:
                shapes = random.choices(["triangle", "circle"], k=obj_num)
            else:
                shapes = random.choices(["triangle", "square"], k=obj_num)

        if "color" in params or random.random() < 0.5:
            colors = random.choices(["green", "yellow"], k=obj_num)
        else:
            colors = data_utils.random_select_unique_mix(config.color_large_exclude_gray, obj_num)

        if "size" in params or random.random() < 0.5:
            # shapes = [random.choice(["square", "circle"])] * obj_num
            sizes = [obj_size] * obj_num
        else:
            sizes = data_utils.get_random_sizes(obj_num,obj_size)

    else:

        cf_params = data_utils.get_proper_sublist(params)
        if "shape" in cf_params:
            shapes = random.choices(["square", "circle"], k=obj_num)
        else:
            if random.random() < 0.5:
                shapes = data_utils.random_select_unique_mix(["triangle", "circle"], obj_num)
            else:
                shapes = data_utils.random_select_unique_mix(["triangle", "square"], obj_num)
        if "color" in cf_params:
            colors = random.choices(["green", "yellow"], k=obj_num)
        else:
            colors = data_utils.random_select_unique_mix(config.color_large_exclude_gray, obj_num)

        if "size" in cf_params:
            sizes = [obj_size] * obj_num
        else:
            sizes = data_utils.get_random_sizes(obj_num,obj_size)


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


def non_overlap_big_triangle(params, is_positive, clu_num, obj_quantity, pin):
    obj_size = 0.05
    objs = closure_big_triangle(obj_size, is_positive, clu_num, params, obj_quantity, pin)
    t = 0
    tt = 0
    max_try = 2000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = closure_big_triangle(obj_size, is_positive, clu_num, params, obj_quantity, pin)
        if tt > 10:
            tt = 0
            obj_size = obj_size * 0.90
        tt = tt + 1
        t = t + 1
    return objs
