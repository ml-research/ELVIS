# Created by jing at 02.03.25


import random

from scripts import config
from scripts.utils import encode_utils, data_utils, pos_utils
from scripts.utils.shape_utils import overlaps, overflow
from scripts.utils.pos_utils import get_feature_circle_positions


def feature_closure_circle(is_positive, clu_num, params, pin):
    cluster_dist = 0.3
    x_min = 0.2
    x_max = 0.8
    y_min = 0.4
    y_max = 0.95
    obj_num = 4

    objs = []
    # Generate random anchors for clusters ensuring proper distance
    group_anchors = []
    for _ in range(clu_num):
        group_anchors.append(pos_utils.generate_random_anchor(group_anchors, cluster_dist, x_min, x_max, y_min, y_max))

    for i in range(clu_num):
        clu_size = ({1: 0.2 + random.random() * 0.1,
                     2: 0.2 + random.random() * 0.1,
                     3: 0.2 + random.random() * 0.1,
                     }.get(clu_num, 0.3))
        obj_size = clu_size * (0.3 + random.random() * 0.1)

        positions = get_feature_circle_positions(group_anchors[i], clu_size)
        if not is_positive and pin and random.random() > 0.5:
            positions[-1] = pos_utils.random_shift_point(positions[-1], 0.05, clu_size / 2)
            is_positive = True
        if is_positive:
            if "shape" in params or random.random() > 0.5:
                shapes = [random.choice(["square", "triangle"])] * obj_num
                shapes += ["circle"]
            else:
                shapes = data_utils.random_select_unique_mix(config.bk_shapes[1:], obj_num)
                shapes += ["circle"]
            if "color" in params or random.random() < 0.5:
                colors = random.choices(["blue", "red"], k=obj_num)
                colors += ["lightgray"]
            else:
                colors = data_utils.random_select_unique_mix(config.color_large_exclude_gray, obj_num)
                colors += ["lightgray"]

            if "size" in params or random.random() < 0.5:
                shapes = [random.choice(["square", "triangle"])] * obj_num+ ["circle"]
                sizes = [obj_size] * obj_num
                sizes += [clu_size]
            else:
                sizes = [random.uniform(obj_size * 0.8, obj_size * 1) for _ in range(obj_num)]
                sizes += [clu_size]
        else:
            cf_params = data_utils.get_proper_sublist(params)
            if "shape" in cf_params:
                shapes = [random.choice(["square", "triangle"])] * obj_num
                shapes += ["circle"]
            else:
                shapes = data_utils.random_select_unique_mix(config.bk_shapes[1:], obj_num)
                shapes += ["circle"]
            if "color" in cf_params:
                colors = random.choices(["blue", "red"], k=obj_num)
                colors += ["lightgray"]
            else:
                colors = data_utils.random_select_unique_mix(config.color_large_exclude_gray, obj_num)
                colors += ["lightgray"]

            if "size" in cf_params:
                shapes = [random.choice(["square", "triangle"])] * obj_num + ["circle"]

                sizes = [obj_size] * obj_num
                sizes += [clu_size]
            else:
                sizes = [random.uniform(obj_size * 0.8, obj_size * 1) for _ in range(obj_num)]
                sizes += [clu_size]
        try:
            for i in range(len(positions)):
                objs.append(encode_utils.encode_objs(
                    x=positions[i][0],
                    y=positions[i][1],
                    size=sizes[i],
                    color=colors[i],
                    shape=shapes[i],
                    line_width=-1,
                    solid=True,
                ))
        except Exception as e:
            raise e

    return objs


def non_overlap_feature_circle(params, is_positive, clu_num, pin):
    objs = feature_closure_circle(is_positive, clu_num, params, pin)
    t = 0
    tt = 0
    max_try = 2000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = feature_closure_circle(is_positive, clu_num, params, pin)
        if tt > 10:
            tt = 0
        tt = tt + 1
        t = t + 1
    return objs
