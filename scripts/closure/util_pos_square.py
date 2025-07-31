# Created by jing at 01.03.25


import random
import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import make_interp_spline, interp1d

from scripts import config
from scripts.utils import encode_utils, data_utils, pos_utils
from scripts.utils.shape_utils import overlaps, overflow


def closure_big_square(obj_size, is_positive, clu_num, params, irrel_params, obj_quantity, pin):
    logic = {
        "shape": ["triangle", "circle"],
        "color": ["blue", "red"],
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
            pos = pos_utils.get_square_positions(counts[i], group_anchors[i][0], group_anchors[i][1])
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


def get_logic_rules(params):
    head = "group_target(X)"
    body = "in(O,X),in(G,X),"
    if "color" in params:
        body += "has_color(blue,O),has_color(red,O),"
    if "size" in params:
        body += "same_obj_size(G),"
    if "shape" in params:
        body += ("has_shape(O1,trianlge),has_shape(O2,circle),no_shape(O3,square),"
                 "in(O1,G),in(O2,G),in(O3,G),")
    rule = f"{head}:-{body}group_shape(square,G),principle(closure,G)."
    return rule


def separate_big_square(rel_params, irrel_params, is_positive, clu_num, obj_quantity, pin):
    obj_size = 0.05
    objs = closure_big_square(obj_size, is_positive, clu_num, rel_params, irrel_params, obj_quantity, pin)
    t = 0
    tt = 0
    max_try = 2000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = closure_big_square(obj_size, is_positive, clu_num, rel_params, irrel_params, obj_quantity, pin)
        if tt > 10:
            tt = 0
            obj_size = obj_size * 0.90
        tt = tt + 1
        t = t + 1
    logic_rules = get_logic_rules(rel_params)
    return objs, logic_rules
