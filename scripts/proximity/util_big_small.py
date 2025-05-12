# Created by jing at 26.02.25


import random

from scripts import config
from scripts.utils.shape_utils import overlaps, overflow
from scripts.utils import pos_utils, encode_utils, data_utils


def proximity_big_small(is_positive, obj_size, clu_num, params, obj_quantities):
    cluster_dist = 0.3  # Increased to ensure clear separation
    neighbour_dist = 0.05

    group_sizes = {"s": range(2, 4), "m": range(3, 5), "l": range(2, 7)}.get(obj_quantities, range(2, 4))
    group_radius = {"s": 0.05, "m": 0.08, "l": 0.1}.get(obj_quantities, 0.05)
    x_min = 0.1
    x_max = 0.9
    y_min = 0.1
    y_max = 0.9
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
        group_anchors.append(pos_utils.generate_random_anchor(group_anchors, cluster_dist, x_min, x_max, y_min, y_max))

    big_shape = "triangle"
    for a_i in range(clu_num):
        group_size = random.choice(group_sizes)
        try:
            neighbour_points = pos_utils.generate_points(group_anchors[a_i], group_radius, group_size, neighbour_dist)
        except IndexError:
            raise IndexError
        # negative 30% no big, but else as same as positive
        sizes = [big_size] + [obj_size] * (group_size - 1)
        if is_positive:
            if "shape" in params or random.random() < 0.5:
                shapes = [big_shape] + random.choices(config.bk_shapes[1:], k=group_size - 1)
            else:
                shapes = [random.choice(["circle", "square"])] + random.choices(config.bk_shapes[1:], k=group_size - 1)

            if "color" in params or random.random() < 0.5:
                colors = random.choices(["green", "yellow"], k=group_size)
            else:
                colors = data_utils.random_select_unique_mix(config.color_large_exclude_gray, group_size)
        else:
            cf_params = data_utils.get_proper_sublist(params)
            if "shape" in cf_params:
                shapes = [big_shape] + random.choices(config.bk_shapes[1:], k=group_size - 1)
            else:
                shapes = [random.choice(["circle", "square"])] + random.choices(config.bk_shapes[1:], k=group_size - 1)
            if "color" in cf_params:
                colors = random.choices(["green", "yellow"], k=group_size)
            else:
                colors = data_utils.random_select_unique_mix(config.color_large_exclude_gray, group_size)

        try:
            for i in range(len(neighbour_points)):
                objs.append(encode_utils.encode_objs(
                    x=neighbour_points[i][0],
                    y=neighbour_points[i][1],
                    size=sizes[i],
                    color=colors[i],
                    shape=shapes[i],
                    line_width=-1,
                    solid=True
                ))
        except Exception as e:
            raise e
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
                encode_utils.encode_objs(x=x, y=y, size=obj_size, color=color, shape=shape, line_width=-1, solid=True))

    return objs


def overlap_big_small(fixed_props, is_positive, cluster_num, obj_quantities, pin):
    obj_size = 0.05
    objs = proximity_big_small(is_positive, obj_size, cluster_num, fixed_props, obj_quantities)
    return objs


def non_overlap_big_small_2(fixed_props, is_positive, cluster_num, obj_quantities):
    obj_size= 0.05
    objs = proximity_big_small_2(is_positive, obj_size, cluster_num, fixed_props, obj_quantities)
    t = 0
    tt = 0
    max_try = 1000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = proximity_big_small_2(is_positive, obj_size, cluster_num, fixed_props, obj_quantities)
        if tt > 10:
            tt = 0
            obj_size = obj_size * 0.90
        tt = tt + 1
        t = t + 1
    return objs
