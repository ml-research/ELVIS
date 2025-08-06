# Created by jing at 26.02.25


import random

from scripts import config
from scripts.utils.shape_utils import overlaps, overflow
from scripts.utils import pos_utils, encode_utils, data_utils


def proximity_big_small(is_positive, obj_size, clu_num, params, irrel_params, cf_params, obj_quantities):
    cluster_dist = 0.3  # Increased to ensure clear separation
    neighbour_dist = 0.05
    group_size = config.standard_quantity_dict[obj_quantities]
    group_radius = config.get_grp_r(0.1, obj_quantities)
    logic = {"shape": random.sample(config.all_shapes, group_size), "color": random.sample(config.color_large_exclude_gray, group_size)}
    objs = []

    if not is_positive and random.random() < 0.3:
        big_size = obj_size
        is_positive = True
    else:
        big_size = random.random() * 0.05 + 0.1

    if not is_positive and "count" in params:
        new_cluster_num = random.randint(1, clu_num + 1)
        while new_cluster_num == clu_num:
            new_cluster_num = random.randint(1, clu_num + 1)
        clu_num = new_cluster_num

    # Generate random anchors for clusters ensuring proper distance
    group_anchors = []
    for _ in range(clu_num):
        group_anchors.append(pos_utils.generate_random_anchor(group_anchors, cluster_dist, 0.1, 0.9, 0.1, 0.9))

    big_shape = "triangle"
    for a_i in range(clu_num):
        is_random = False
        group_size = random.choice(group_size)
        # determine positions
        if is_positive or ("proximity" in cf_params and not is_positive):
            neighbour_points = pos_utils.generate_points(group_anchors[a_i], group_radius, group_size, neighbour_dist)
        else:
            neighbour_points = pos_utils.get_random_positions(group_size, obj_size)

        if "shape" in params and is_positive or ("shape" in cf_params and not is_positive):
            shapes = [big_shape] + random.choices(config.bk_shapes[1:], k=group_size - 1)
        else:
            shapes = [random.choice(["circle", "square"])] + random.choices(config.bk_shapes[1:], k=group_size - 1)

        if "color" in params and is_positive or ("color" in cf_params and not is_positive):
            colors = random.choices(["green", "yellow"], k=group_size)
        else:
            colors = data_utils.random_select_unique_mix(config.color_large_exclude_gray, group_size)

        if "size" in params and is_positive or ("size" in cf_params and not is_positive):
            sizes = [big_size] + [obj_size] * (group_size - 1)
        else:
            sizes = [random.uniform(obj_size * 0.4, obj_size * 0.7)] * group_size

        group_ids = [a_i] * group_size
        objs += encode_utils.encode_scene(neighbour_points, sizes, colors, shapes, group_ids, is_positive)
    return objs


def proximity_big_small_2(is_positive, given_size, cluster_num, fixed_props, obj_quantities):
    cluster_dist, neighbour_dist, group_radius = 0.3, 0.05, 0.05
    group_sizes = [2, 3]
    fixed_props = fixed_props.split("_")

    if not is_positive:
        fixed_props = pos_utils.random_pop_elements(fixed_props)

        if "count" in fixed_props:
            cluster_num = random.choice([n for n in range(1, cluster_num + 2) if n != cluster_num])

    def generate_random_anchor(existing_anchors):
        while True:
            anchor = [random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)]
            if all(pos_utils.euclidean_distance(anchor, existing) > cluster_dist for existing in existing_anchors):
                return anchor

    group_anchors = []
    for _ in range(cluster_num):
        group_anchors.append(generate_random_anchor(group_anchors))

    objs, big_size = [], random.uniform(0.1, 0.15)
    fixed_shape, fixed_color = random.choice(config.bk_shapes[1:]), random.choice(config.color_large_exclude_gray)

    for a_i in range(cluster_num):
        group_size = random.choice(group_sizes)
        neighbour_points = pos_utils.generate_points(group_anchors[a_i], group_radius, group_size, neighbour_dist)
        has_big = is_positive or a_i in range(cluster_num)

        for i, (x, y) in enumerate(neighbour_points):
            obj_size = big_size if i == 0 and has_big else given_size

            shape = fixed_shape if "shape" in fixed_props and is_positive else random.choice(config.bk_shapes[1:])
            color = fixed_color if "color" in fixed_props and is_positive else random.choice(
                config.color_large_exclude_gray)

            if "shape" in fixed_props and not is_positive:
                shape = fixed_shape if "count" in fixed_props else random.choice(
                    [s for s in config.bk_shapes[1:] if s != fixed_shape])
            if "color" in fixed_props and not is_positive:
                color = fixed_color if "count" in fixed_props else random.choice(
                    [c for c in config.color_large_exclude_gray if c != fixed_color])

            objs.append(
                encode_utils.encode_objs(x=x, y=y, size=obj_size, color=color,
                                         shape=shape, line_width=-1, solid=True,
                                         group_id=a_i))

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


def overlap_big_small(params, irrel_params, is_positive, cluster_num, obj_quantities, pin):
    obj_size = 0.05
    cf_params = data_utils.get_proper_sublist(params + ["proximity"])
    objs = proximity_big_small(is_positive, obj_size, cluster_num, params, irrel_params, cf_params, obj_quantities)
    return objs


def non_overlap_big_small_2(params, irrel_params, is_positive, cluster_num, obj_quantities):
    obj_size = 0.05
    cf_params = data_utils.get_proper_sublist(params + ["proximity"])

    objs = proximity_big_small_2(is_positive, obj_size, cluster_num, params, irrel_params, cf_params, obj_quantities)
    t = 0
    tt = 0
    max_try = 1000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = proximity_big_small_2(is_positive, obj_size, cluster_num, params, irrel_params, cf_params, obj_quantities)
        if tt > 10:
            tt = 0
            obj_size = obj_size * 0.90
        tt = tt + 1
        t = t + 1

    logics = get_logics(is_positive, params, cf_params, irrel_params)

    return objs, logics
