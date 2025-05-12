# Created by jing at 25.02.25

import random

from scripts import config
from scripts.utils import pos_utils, encode_utils
from scripts.utils.shape_utils import overlaps, overflow


def proximity_grid(is_positive, obj_size, cluster_num=1, fixed_props="", obj_quantities="s"):
    objs = []
    grid_directions = "horizontal" if random.random() > 0.5 else "vertical"
    shape = random.choice(config.bk_shapes[1:])
    color = random.choice(config.color_large_exclude_gray)

    line_size = {"s": 2, "m": 5, "l": 8}.get(obj_quantities, 2)
    obj_size = {"s": obj_size * 1.2, "m": obj_size, "l": obj_size * 0.8}.get(obj_quantities, obj_size * 1.2)
    if cluster_num > 2:
        max_lines = 2
    else:
        max_lines = 3
    if not is_positive:
        if random.random() > 0.5 and cluster_num > 2:
            cluster_num -= 1
        else:
            cluster_num += 1
    for a_i in range(cluster_num):
        grid_lines = random.randint(1, max_lines)
        for i in range(grid_lines):
            if grid_directions == "vertical":
                x = 1 / cluster_num * a_i + 1 / cluster_num / (grid_lines + 1) * (i + 1) + 0.05
                for y_i in range(line_size):
                    y = (y_i + 1) / (line_size + 2)
                    if "shape" not in fixed_props:
                        shape = random.choice(config.bk_shapes[1:])
                    if "color" not in fixed_props:
                        color = random.choice(config.color_large_exclude_gray)

                    obj = encode_utils.encode_objs(x=x, y=y, size=obj_size, color=color, shape=shape, line_width=-1,
                                                   solid=True)
                    objs.append(obj)

            else:
                y = 1 / cluster_num * a_i + 1 / cluster_num / (grid_lines + 1) * (i + 1) + 0.05
                for x_i in range(line_size):
                    if "shape" not in fixed_props:
                        shape = random.choice(config.bk_shapes[1:])
                    if "color" not in fixed_props:
                        color = random.choice(config.color_large_exclude_gray)

                    x = (x_i + 1) / (line_size + 2)
                    obj = encode_utils.encode_objs(x=x, y=y, size=obj_size, color=color, shape=shape, line_width=-1,
                                                   solid=True)
                    objs.append(obj)

    return objs


def non_overlap_grid(fixed_props, is_positive, cluster_num, obj_quantities, pin):
    obj_size = 0.05
    objs = proximity_grid(is_positive, obj_size, cluster_num, fixed_props, obj_quantities)
    t = 0
    tt = 0
    max_try = 1000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = proximity_grid(is_positive, obj_size, cluster_num, fixed_props, obj_quantities)
        if tt > 10:
            tt = 0
            obj_size = obj_size * 0.90
        tt = tt + 1
        t = t + 1
    return objs
