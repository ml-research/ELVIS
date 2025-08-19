# Created by jing at 26.02.25


import random

from scripts import config
from scripts.utils.shape_utils import overlaps, overflow
from scripts.utils import pos_utils, encode_utils, data_utils


def proximity_fixed_props(fixed_props, irrel_params, cf_params, clu_num, is_positive, obj_size, obj_quantities):
    cluster_dist = 0.25  # Increased to ensure clear separation
    neighbour_dist = 0.05
    group_size = config.standard_quantity_dict[obj_quantities]
    group_radius = config.get_grp_r(0.1, obj_quantities)
    # group_radius = {"s": 0.05, "m": 0.08, "l": 0.1}.get(obj_quantities, 0.05)
    logic = {
        "shape": random.sample(config.all_shapes, 2),
        "color": random.sample(config.color_large_exclude_gray, 2),
    }

    def generate_random_anchor(existing_anchors):
        while True:
            anchor = [random.uniform(0.10, 0.9), random.uniform(0.1, 0.9)]
            if all(pos_utils.euclidean_distance(anchor, existing) > cluster_dist for existing in existing_anchors):
                return anchor

    # Generate random anchors for clusters ensuring proper distance
    group_anchors = []
    for _ in range(clu_num):
        group_anchors.append(generate_random_anchor(group_anchors))

    # group_anchors = [generate_random_anchor([]) for _ in range(cluster_num)]
    objs = []
    # Determine how many clusters will contain fixed shapes
    fixed_clusters = random.randint(0, clu_num - 1)
    fix_indices = random.sample(range(clu_num), fixed_clusters)

    for a_i in range(clu_num):
        group_ids = [a_i] * group_size
        grp_all_same = True if is_positive or (not is_positive and a_i in fix_indices) else False

        if is_positive or ("proximity" in cf_params and not is_positive):
            neighbour_points = pos_utils.generate_points(group_anchors[a_i], group_radius, group_size, neighbour_dist)
        else:
            neighbour_points = pos_utils.get_random_positions(group_size, obj_size)

        if grp_all_same and "shape" in fixed_props or ("shape" in cf_params and not is_positive):
            shapes = [random.choice(logic["shape"]) for _ in range(group_size)]
        else:
            shapes = [random.choice(config.all_shapes) for _ in range(group_size)]

        if grp_all_same and "color" in fixed_props or ("color" in cf_params and not is_positive):
            colors = [random.choice(logic["color"]) for _ in range(group_size)]
        else:
            colors = [random.choice(config.color_large_exclude_gray) for _ in range(group_size)]

        if grp_all_same and "size" in fixed_props or ("size" in cf_params and not is_positive):
            sizes = [obj_size] * group_size
        else:
            sizes = [random.uniform(obj_size * 0.4, obj_size * 1.5) for _ in range(group_size)]

        if "shape" in irrel_params:
            shapes = [random.choice(config.all_shapes)] * group_size
        if "color" in irrel_params:
            colors = [random.choice(config.color_large_exclude_gray)] * group_size
        if "size" in irrel_params:
            sizes = [obj_size] * group_size

        objs += encode_utils.encode_scene(neighbour_points, sizes, colors, shapes, group_ids, is_positive)
    return objs


def get_logics(is_positive, fixed_props, cf_params, irrel_params):
    head = "group_target(X)"
    body = ""
    if "shape" in fixed_props:
        body += "has_shape(X,triangle),"
    if "color" in fixed_props:
        body += "has_color(X,red),"
    rule = f"{head}:-{body}principle(proximity)."
    logic = {
        "rule": rule,
        "is_positive": is_positive,
        "fixed_props": fixed_props,
        "cf_params": cf_params,
        "irrel_params": irrel_params,
        "principle": "proximity",

    }
    return logic


def non_overlap_fixed_props(params, irrel_params, is_positive, clu_num, obj_quantities, pin):
    obj_size = 0.05
    cf_params = data_utils.get_proper_sublist(params + ["proximity"])

    objs = proximity_fixed_props(params, irrel_params, cf_params, clu_num, is_positive, obj_size, obj_quantities)
    t = 0
    tt = 0
    max_try = 1000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = proximity_fixed_props(params, irrel_params, cf_params, clu_num, is_positive, obj_size, obj_quantities)
        if tt > 10:
            tt = 0
            obj_size = obj_size * 0.90
        tt = tt + 1
        t = t + 1
    logics = get_logics(is_positive, params, cf_params, irrel_params)
    return objs, logics
