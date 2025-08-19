# Created by jing at 25.02.25


import random

from scripts import config
from scripts.utils.shape_utils import overlaps, overflow
from scripts.utils import pos_utils, encode_utils, data_utils


def get_invariant_values(irrel_params, obj_size):
    invariant = {
        "shape": random.choice(config.all_shapes) if "shape" in irrel_params else None,
        "color": random.choice(config.color_large_exclude_gray) if "color" in irrel_params else None,
        "size": obj_size,
        "count": True if "count" in irrel_params else False,
    }
    return invariant


def assign_properties(grp_obj_num, has_red_triangle, params, invariant, cf_params, logic):
    if "shape" in cf_params or ("shape" in params and has_red_triangle):
        shapes = [logic["shape"][0]] + [random.choice(logic["shape"]) for _ in range(grp_obj_num - 1)]
    else:
        shapes = [random.choice(config.all_shapes) for _ in range(grp_obj_num)]
    if "color" in cf_params or ("color" in params and has_red_triangle):
        colors = [logic["color"][0]] + [random.choice(config.color_large_exclude_gray) for _ in range(grp_obj_num - 1)]
    else:
        selected_color = config.color_large_exclude_gray.copy()
        selected_color.remove("red")
        colors = [random.choice(selected_color) for _ in range(grp_obj_num)]
    if "size" in cf_params or ("size" in params and has_red_triangle):
        sizes = [logic["size"]] * grp_obj_num
    else:
        sizes = [random.uniform(logic["size"] * 0.5, logic["size"] * 1.7) for _ in range(grp_obj_num)]

    if invariant["shape"] is not None:
        shapes = [invariant["shape"]] * grp_obj_num
    if invariant["color"] is not None:
        colors = [invariant["color"]] * grp_obj_num
    if invariant["size"] is not None:
        sizes = [invariant["size"]] * grp_obj_num

    return shapes, colors, sizes


def proximity_red_triangle(is_positive, obj_size, clu_num, params, irrel_params, cf_params, obj_quantities, qualifiers):
    # settings
    cluster_dist = 0.3
    neighbour_dist = 0.05
    logic = {
        "shape": ["triangle"],
        "color": ["red"],
        "size": obj_size,
        "count": True,
    }
    group_sizes = {"s": range(2, 4), "m": range(3, 5), "l": range(5, 7), "xl": range(10, 15)}.get(obj_quantities, range(2, 4))
    group_radius = {"s": 0.06, "m": 0.08, "l": 0.11, "xl": 0.13}.get(obj_quantities, 0.05)
    invariant = get_invariant_values(irrel_params, obj_size)

    if invariant["count"]:
        grp_obj_nums = [random.choice(group_sizes)] * clu_num
    else:
        grp_obj_nums = [random.choice(group_sizes) for _ in range(clu_num)]

    group_anchors = []
    for _ in range(clu_num):
        group_anchors.append(pos_utils.generate_random_anchor(group_anchors, cluster_dist, 0.1, 0.9, 0.1, 0.9))
    if "all" in qualifiers:
        red_triangle_neg_indices = data_utils.not_all_true(clu_num)
        red_triangle_pos_indices = [True] * clu_num
    elif "exist" in qualifiers:
        red_triangle_neg_indices = [False] * clu_num
        red_triangle_pos_indices = data_utils.at_least_one_true(clu_num)
    else:
        raise ValueError("qualifiers must either be 'exist' or 'all'")
    objs = []
    for a_i in range(clu_num):
        grp_obj_num = grp_obj_nums[a_i]
        if not is_positive and "proximity" not in cf_params:
            neighbour_points = pos_utils.get_random_positions(grp_obj_num, obj_size)
        else:
            neighbour_points = pos_utils.generate_points(group_anchors[a_i], group_radius, grp_obj_num, neighbour_dist)
        has_red_triangle = red_triangle_pos_indices[a_i] if is_positive else red_triangle_neg_indices[a_i]
        shapes, colors, sizes = assign_properties(grp_obj_num, has_red_triangle, params, invariant, cf_params, logic)
        group_ids = [a_i] * grp_obj_num
        objs += encode_utils.encode_scene(neighbour_points, sizes, colors, shapes, group_ids, is_positive)
    return objs


def get_logics(is_positive, qualifiers, fixed_props, cf_params, irrel_params):
    if qualifiers == "all":
        head = "group_target(X)"
    else:
        head = "image_target(X)"
    body = ""
    if "shape" in fixed_props:
        body += "has_shape(X,triangle),"
    if "color" in fixed_props:
        body += "has_color(X,red),"
    rule = f"{head}:-{body}principle(proximity)."
    logic = {
        "rule": rule,
        "is_positive": is_positive,
        "qualifiers": qualifiers,
        "fixed_props": fixed_props,
        "cf_params": cf_params,
        "irrel_params": irrel_params,
        "principle": "proximity",

    }
    return logic


def non_overlap_red_triangle(params, irrel_params, is_positive, cluster_num, obj_quantities, qualifiers, pin):
    obj_size = 0.05
    # random determined properties
    cf_params = data_utils.get_proper_sublist(params + ["proximity"])
    objs = proximity_red_triangle(is_positive, obj_size, cluster_num, params, irrel_params, cf_params, obj_quantities, qualifiers)
    t = 0
    tt = 0
    max_try = 1000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = proximity_red_triangle(is_positive, obj_size, cluster_num, params, irrel_params, cf_params, obj_quantities, qualifiers)
        if tt > 10:
            tt = 0
            obj_size = obj_size * 0.90
        tt = tt + 1
        t = t + 1

    logics = get_logics(is_positive, qualifiers, params, cf_params, irrel_params)

    return objs, logics
