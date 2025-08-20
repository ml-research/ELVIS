# Created by jing at 01.03.25
import math

import random
import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import make_interp_spline, interp1d

from scripts import config
from scripts.utils import encode_utils, data_utils, pos_utils
from scripts.utils.shape_utils import overlaps, overflow
from scripts.utils.data_utils import get_shapes, get_colors


def generate_random_anchor(existing_anchors):
    cluster_dist = 0.1  # Increased to ensure clear separation
    while True:
        anchor = [random.uniform(0.4, 0.7), random.uniform(0.4, 0.7)]
        if all(pos_utils.euclidean_distance(anchor, existing) > cluster_dist for existing in existing_anchors):
            return anchor


def generate_positions(clu_num, obj_quantity, obj_size, cluster=True):
    # obj_quantity = config.standard_quantity_dict[obj_quantity]
    positions = []
    group_anchors = []
    for _ in range(clu_num):
        group_anchors.append(generate_random_anchor(group_anchors))
    group_ids = []
    for i in range(clu_num):
        x, y = group_anchors[i]
        if cluster:
            pos = pos_utils.get_triangle_positions(obj_quantity, x, y)
        else:
            pos = pos_utils.get_random_positions(config.standard_quantity_dict[obj_quantity], obj_size)
        positions += pos
        group_ids += [i] * len(pos)
    return positions, group_ids


def closure_big_triangle(obj_size, is_positive, clu_num, params, irrel_params, cf_params, obj_quantity, pin):
    logic = {
        "shape": ["square", "circle"],
        "color": ["green", "yellow"],
        "size": obj_size,
        "count": True
    }
    all_shapes = config.all_shapes
    all_colors = config.color_large_exclude_gray

    # --- Count logic ---
    if is_positive:
        counts = [obj_quantity] * clu_num
    else:
        rules = params + ["closure"]
        to_break = set([random.choice(rules)])
        for rule in rules:
            if rule not in to_break and random.random() < 0.1:
                to_break.add(rule)
        # Count: break by using a different count per cluster
        if "count" in to_break:
            available_counts = [k for k in config.standard_quantity_dict.keys() if k != obj_quantity]
            counts = [random.choice(available_counts) for _ in range(clu_num)]
        else:
            counts = [obj_quantity] * clu_num
    # --- Position logic ---
    cluster = True
    if not is_positive and "closure" in to_break:
        cluster = False

    # --- Position and group id generation ---
    positions = []
    group_ids = []
    group_anchors = []
    for i in range(clu_num):
        count = config.standard_quantity_dict[counts[i]] // 1.5
        new_anchor = pos_utils.generate_random_anchor(group_anchors, cluster_dist=0.1, x_min=0.35, x_max=0.80, y_min=0.34, y_max=0.75)
        group_anchors.append(new_anchor)
        if cluster:
            pos = pos_utils.get_triangle_positions(counts[i], group_anchors[i][0], group_anchors[i][1])
            if None in pos:
                raise ValueError("Position generation returned None, check parameters.")
        else:
            pos = pos_utils.get_random_positions(count, obj_size)
            if None in pos:
                raise ValueError("Position generation returned None, check parameters.")
        positions += pos
        group_ids += [i] * len(pos)

    obj_num = len(positions)

    irrel_shape = random.choice(all_shapes)
    irrel_color = random.choice(all_colors)
    irrel_size = obj_size

    def assign_property(prop, relevant, irrel_value, logic_values, all_values):
        if prop in irrel_params:
            return [irrel_value] * obj_num
        if is_positive:
            if prop in params:
                return random.choices(logic_values, k=obj_num)
            else:
                return random.choices(all_values, k=obj_num)
        else:
            if prop in cf_params and prop not in to_break:
                return random.choices(logic_values, k=obj_num)
            else:
                return random.choices(all_values, k=obj_num)

    shapes = assign_property("shape", params, irrel_shape, logic["shape"], all_shapes)
    colors = assign_property("color", params, irrel_color, logic["color"], all_colors)
    sizes = assign_property("size", params, irrel_size, [logic["size"]], data_utils.get_random_sizes(obj_num, obj_size))

    try:
        objs = encode_utils.encode_scene(positions, sizes, colors, shapes, group_ids, is_positive)
    except TypeError:
        raise TypeError("Error in encoding scene with positions, sizes, colors, shapes, and group_ids.")

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
    rule = f"{head}:-{body}group_shape(triangle,G),principle(closure,G)."
    logic = {
        "rule": rule,
        "is_positive": is_positive,
        "fixed_props": params,
        "cf_params": cf_params,
        "irrel_params": irrel_params,
        "principle": "closure",
    }
    return logic


def separate_big_triangle(params, irrel_params, is_positive, clu_num, obj_quantity, pin):
    obj_size = 0.05
    cf_params = data_utils.get_proper_sublist(params + ["closure"])

    objs = closure_big_triangle(obj_size, is_positive, clu_num, params, irrel_params, cf_params, obj_quantity, pin)
    t = 0
    tt = 0
    max_try = 2000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = closure_big_triangle(obj_size, is_positive, clu_num, params, irrel_params, cf_params, obj_quantity, pin)
        if tt > 10:
            tt = 0
            obj_size = obj_size * 0.90
        tt = tt + 1
        t = t + 1
    logic_rules = get_logic_rules(is_positive, params, cf_params, irrel_params)

    return objs, logic_rules
