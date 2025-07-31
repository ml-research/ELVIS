# Created by jing at 02.03.25


import random

from scripts import config
from scripts.utils import encode_utils, data_utils, pos_utils
from scripts.utils.shape_utils import overlaps, overflow
from scripts.utils.pos_utils import get_feature_circle_positions


def feature_closure_circle(is_positive, clu_num, params, irrel_params, pin):
    obj_num = 4

    clu_size = ({1: 0.2 + random.random() * 0.1,
                 2: 0.2 + random.random() * 0.1,
                 3: 0.2 + random.random() * 0.1,
                 }.get(clu_num, 0.3))
    obj_size = clu_size * (0.3 + random.random() * 0.1)
    # Generate random anchors for clusters ensuring proper distance
    group_anchors = []
    color_val = random.choice(config.color_large_exclude_gray)
    size_val = obj_size
    for _ in range(clu_num):
        group_anchors.append(pos_utils.generate_random_anchor(group_anchors, 0.3, 0.2, 0.8, 0.4, 0.95))

    objs = []
    for i in range(clu_num):
        positions = get_feature_circle_positions(group_anchors[i], clu_size)
        is_random = False
        if not is_positive and pin and random.random() > 0.5:
            positions[-1] = pos_utils.random_shift_point(positions[-1], 0.05, clu_size / 2)
            is_positive = True
            is_random = True
        shapes = ["square"] * obj_num + ["circle"]

        if is_positive:
            if "color" in params or random.random() < 0.5:
                colors = random.choices(["blue", "red"], k=obj_num)
                colors += ["lightgray"]
            else:
                colors = data_utils.random_select_unique_mix(config.color_large_exclude_gray, obj_num)
                colors += ["lightgray"]
            if "color" in irrel_params:
                colors = [color_val] * obj_num + ["lightgray"]

            if "size" in params or random.random() < 0.5:
                # shapes = [random.choice(["square", "triangle"])] * obj_num + ["circle"]
                sizes = [obj_size] * obj_num
                sizes += [clu_size]
            else:
                sizes = data_utils.get_random_sizes(obj_num, obj_size)
                sizes += [clu_size]
            if "size" in irrel_params:
                sizes = [size_val] * obj_num + [clu_size]
        else:
            cf_params = data_utils.get_proper_sublist(params)
            if "color" in cf_params:
                colors = random.choices(["blue", "red"], k=obj_num)
                colors += ["lightgray"]
            else:
                colors = data_utils.random_select_unique_mix(config.color_large_exclude_gray, obj_num)
                colors += ["lightgray"]
            if "color" in irrel_params:
                colors = [color_val] * obj_num + ["lightgray"]
            if "size" in cf_params:
                shapes = [random.choice(["square", "triangle"])] * obj_num + ["circle"]
                sizes = [obj_size] * obj_num
                sizes += [clu_size]
            else:
                sizes = data_utils.get_random_sizes(obj_num, obj_size)
                sizes += [clu_size]
            if "size" in irrel_params:
                sizes = [size_val] * obj_num + [clu_size]
        group_id = -1 if is_random else i
        objs += encode_utils.encode_scene(positions, sizes, colors, shapes,
                                          [group_id] * len(positions), is_positive)

    return objs


def get_logic_rules(params, clu_num):
    head = "group_target(X)"

    body = "in(O,X),in(G,X),"
    if "color" in params:
        body += "has_color(blue,O),has_color(red,O),"
    if "size" in params:
        body += "same_obj_size(G),"

    rule = f"{head}:-{body}group_shape(circle,G),principle(closure,G)."
    return rule


def non_overlap_feature_circle(params, irrel_params, is_positive, clu_num, pin):
    objs = feature_closure_circle(is_positive, clu_num, params, irrel_params, pin)
    t = 0
    tt = 0
    max_try = 2000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = feature_closure_circle(is_positive, clu_num, params, irrel_params, pin)
        if tt > 10:
            tt = 0
        tt = tt + 1
        t = t + 1
    logic_rules = get_logic_rules(params, clu_num)
    return objs, logic_rules
