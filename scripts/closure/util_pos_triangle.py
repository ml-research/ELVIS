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


def closure_big_triangle(obj_size, is_positive, clu_num, params, irrel_params, obj_quantity, pin):
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
        rules = params + ["position"]
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
    if not is_positive and "position" in to_break:
        cluster = False

    # --- Position and group id generation ---
    positions = []
    group_ids = []
    group_anchors = []
    for i in range(clu_num):
        count = config.standard_quantity_dict[counts[i]]
        group_anchors.append(pos_utils.generate_random_anchor(group_anchors))
        if cluster:
            pos = pos_utils.get_triangle_positions(counts[i], group_anchors[i][0], group_anchors[i][1])
        else:
            pos = pos_utils.get_random_positions(count, obj_size)
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
            cf_params = data_utils.get_proper_sublist(params + ["position"])
            if prop in cf_params and prop not in to_break:
                return random.choices(logic_values, k=obj_num)
            else:
                return random.choices(all_values, k=obj_num)

    shapes = assign_property("shape", params, irrel_shape, logic["shape"], all_shapes)
    colors = assign_property("color", params, irrel_color, logic["color"], all_colors)
    sizes = assign_property("size", params, irrel_size, [logic["size"]], data_utils.get_random_sizes(obj_num, obj_size))

    objs = encode_utils.encode_scene(positions, sizes, colors, shapes, group_ids, is_positive)
    return objs

    #
    # if is_positive:
    #     # Use fixed count if "count" in params, else use default
    #     count = logic["count"] if "count" in params else random.choice(list(config.standard_quantity_dict.keys()))
    #     positions, group_ids = generate_positions(clu_num, count, obj_size, cluster=True)
    #     obj_num = len(positions)
    #     shapes = get_shapes(obj_num, logic["shape"]) if "shape" in params else get_shapes(obj_num, all_shapes)
    #     colors = get_colors(obj_num, logic["color"]) if "color" in params else get_colors(obj_num, all_colors)
    #     sizes = [logic["size"]] * obj_num if "size" in params else data_utils.get_random_sizes(obj_num, obj_size)
    # else:
    #     # Always break at least one rule
    #     rules = params
    #     to_break = set([random.choice(rules)])
    #     for rule in rules:
    #         if rule not in to_break and random.random() < 0.5:
    #             to_break.add(rule)
    #
    #     # Break "count" by varying object number per cluster
    #     if "count" in to_break:
    #         # Randomize count per cluster
    #         available_counts = [k for k in config.standard_quantity_dict.keys() if k != obj_quantity]
    #         counts = [random.choice(available_counts) for _ in range(clu_num)]
    #         # counts = [random.choice(list(config.standard_quantity_dict.keys())) for _ in range(clu_num)]
    #         positions = []
    #         group_ids = []
    #         for i, c in enumerate(counts):
    #             pos, _ = generate_positions(1, c, obj_size, cluster="position" not in to_break)
    #             positions += pos
    #             group_ids += [i] * len(pos)
    #         obj_num = len(positions)
    #     else:
    #         count = logic["count"]
    #         positions, group_ids = generate_positions(clu_num, count, obj_size, cluster="position" not in to_break)
    #         obj_num = len(positions)
    #
    #     shapes = get_shapes(obj_num, logic["shape"]) if "shape" in params and "shape" not in to_break else get_shapes(obj_num, all_shapes)
    #     colors = get_colors(obj_num, logic["color"]) if "color" in params and "color" not in to_break else get_colors(obj_num, all_colors)
    #     sizes = [logic["size"]] * obj_num if "size" in params and "size" not in to_break else data_utils.get_random_sizes(obj_num, obj_size)
    #
    # objs = encode_utils.encode_scene(positions, sizes, colors, shapes, group_ids, is_positive)
    # return objs


def get_logic_rules(fixed_props):
    head = "image_target(X)"

    body = ""
    if "shape" in fixed_props:
        body += "has_shape(X,square),has_shape(X,circle),no_shape(X,triangle),"
    if "color" in fixed_props:
        body += "has_color(X,green),has_color(X,yellow)"
    rule = f"{head}:-{body}principle(closure)."
    return rule


def separate_big_triangle(params, irrel_params, is_positive, clu_num, obj_quantity, pin):
    obj_size = 0.05
    objs = closure_big_triangle(obj_size, is_positive, clu_num, params, irrel_params, obj_quantity, pin)
    t = 0
    tt = 0
    max_try = 2000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = closure_big_triangle(obj_size, is_positive, clu_num, params, irrel_params, obj_quantity, pin)
        if tt > 10:
            tt = 0
            obj_size = obj_size * 0.90
        tt = tt + 1
        t = t + 1
    logic_rules = get_logic_rules(params)

    return objs, logic_rules
