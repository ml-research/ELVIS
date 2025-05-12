# Created by jing at 26.02.25


import random

from scripts import config
from scripts.utils.shape_utils import overlaps, overflow
from scripts.utils import pos_utils, encode_utils


def proximity_fixed_props(fixed_props, is_positive, obj_size, obj_quantities):
    cluster_dist = 0.3  # Increased to ensure clear separation
    neighbour_dist = 0.05

    cluster_num = random.randint(2, 4)

    group_size = {"s": 2, "m": 4, "l": 7}.get(obj_quantities, 2)
    group_radius = {"s": 0.05, "m": 0.08, "l": 0.1}.get(obj_quantities, 0.05)

    def generate_random_anchor(existing_anchors):
        while True:
            anchor = [random.uniform(0.15, 0.85), random.uniform(0.15, 0.85)]
            if all(pos_utils.euclidean_distance(anchor, existing) > cluster_dist for existing in existing_anchors):
                return anchor

    # Generate random anchors for clusters ensuring proper distance
    group_anchors = []
    for _ in range(cluster_num):
        group_anchors.append(generate_random_anchor(group_anchors))

    # group_anchors = [generate_random_anchor([]) for _ in range(cluster_num)]
    objs = []

    # Determine how many clusters will contain fixed shapes
    fixed_clusters = random.randint(0, cluster_num - 1)
    fix_indices = random.sample(range(cluster_num), fixed_clusters)
    # group_size = cluster_size

    shapes = [random.choice(config.bk_shapes[1:]) for i in range(group_size)]
    colors = [random.choice(config.color_large_exclude_gray) for i in range(group_size)]

    for a_i in range(cluster_num):
        if "shape" not in fixed_props:
            new_shapes = [random.choice(config.bk_shapes[1:]) for i in range(group_size)]
            # if new list is identical to the old list, regenerate the list
            while (new_shapes == shapes):
                new_shapes = [random.choice(config.bk_shapes[1:]) for i in range(group_size)]
            shapes = new_shapes

        if "color" not in fixed_props:
            new_colors = [random.choice(config.color_large_exclude_gray) for i in range(group_size)]
            while (new_colors == colors):
                new_colors = [random.choice(config.color_large_exclude_gray) for i in range(group_size)]
            colors = new_colors

        neighbour_points = pos_utils.generate_points(group_anchors[a_i], group_radius, group_size, neighbour_dist)
        has_fixed = a_i in fix_indices if not is_positive else True

        for i in range(group_size):

            if has_fixed:
                shape = shapes[i]
                color = colors[i]
            else:
                shape = random.choice(["triangle", "square", "circle"])
                color = random.choice(config.color_large_exclude_gray)
                if random.random() < 0.5:
                    while color == colors[i]:
                        color = random.choice(config.color_large_exclude_gray)
                else:
                    while shape == shapes[i]:
                        shape = random.choice(["triangle", "square", "circle"])
            try:
                x, y = neighbour_points[i]
            except IndexError:
                raise IndexError
            obj = encode_utils.encode_objs(x=x, y=y, size=obj_size, color=color, shape=shape, line_width=-1, solid=True)
            objs.append(obj)

    return objs


def non_overlap_fixed_props(fixed_props, is_positive, obj_quantities, pin):
    obj_size = 0.05
    objs = proximity_fixed_props(fixed_props, is_positive, obj_size, obj_quantities)
    t = 0
    tt = 0
    max_try = 1000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = proximity_fixed_props(fixed_props, is_positive, obj_size, obj_quantities)
        if tt > 10:
            tt = 0
            obj_size = obj_size * 0.90
        tt = tt + 1
        t = t + 1
    return objs
