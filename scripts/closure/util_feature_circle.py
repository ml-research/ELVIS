# Created by jing at 02.03.25


import random

from scripts import config
from scripts.utils import encode_utils, data_utils, pos_utils
from scripts.utils.shape_utils import overlaps, overflow
from scripts.utils.pos_utils import get_feature_circle_positions


def feature_closure_circle(is_positive, clu_num, params, irrel_params, cf_params, pin, try_count):
    obj_num = 4

    clu_size = ({1: 0.4 + random.random() * 0.1,
                 2: 0.3 + random.random() * 0.1,
                 3: 0.25 + random.random() * 0.1,
                 4: 0.2 + random.random() * 0.1,
                 5: 0.2 + random.random() * 0.1,
                 }.get(clu_num, 0.3))
    obj_size = clu_size * (0.3 + random.random() * 0.1) * 0.98 ** try_count
    # Generate random anchors for clusters ensuring proper distance
    group_anchors = []
    color_val = random.choice(config.color_large_exclude_gray)
    size_val = obj_size
    logic = {
        "shape": ["square", "pentagon", "hexagon"]
    }
    for _ in range(clu_num):
        group_anchors.append(pos_utils.generate_random_anchor(group_anchors, 0.3, 0.15, 0.85, 0.35, 0.95))

    objs = []
    tp_objs = []
    invariant_shape = random.choice(logic["shape"])
    for i in range(clu_num):
        positions = get_feature_circle_positions(group_anchors[i], clu_size)
        is_random = False
        if not is_positive and pin and random.random() > 0.5:
            positions[-1] = pos_utils.random_shift_point(positions[-1], 0.05, clu_size / 2)
            is_positive = True
            is_random = True

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
            if not is_positive and "closure" not in cf_params:
                positions = pos_utils.get_random_positions(obj_num + 1, obj_size)
        if "shape" in params and is_positive or (not is_positive and "shape" in cf_params):
            shapes = [random.choice(logic["shape"])] * obj_num + ["circle"]
        else:
            shapes = data_utils.random_select_unique_mix(logic["shape"], obj_num) + ["circle"]
        if "shape" in irrel_params:
            shapes = [invariant_shape] * obj_num + ["circle"]

        group_id = -1 if is_random else i
        objs += encode_utils.encode_scene(positions[:-1], sizes[:-1], colors[:-1], shapes[:-1], [group_id] * (len(positions) - 1), is_positive)
        tp_objs += encode_utils.encode_scene(positions[-1:], sizes[-1:], colors[-1:], shapes[-1:], [group_id], is_positive)

    return objs, tp_objs


def get_logic_rules(is_positive, params, cf_params, irrel_params):
    head = "group_target(X)"

    body = "in(O,X),in(G,X),"
    if "color" in params:
        body += "has_color(blue,O),has_color(red,O),"
    if "size" in params:
        body += "same_obj_size(G),"

    rule = f"{head}:-{body}group_shape(circle,G),principle(closure,G)."

    logic = {
        "rule": rule,
        "is_positive": is_positive,
        "fixed_props": params,
        "cf_params": cf_params,
        "irrel_params": irrel_params,
        "principle": "similarity",

    }
    return logic


def non_overlap_feature_circle(params, irrel_params, is_positive, clu_num, pin):
    cf_params = data_utils.get_proper_sublist(params + ["closure"])
    t = 0
    tt = 0
    objs, tp_objs = feature_closure_circle(is_positive, clu_num, params, irrel_params, cf_params, pin, t)

    max_try = 2000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs, tp_objs = feature_closure_circle(is_positive, clu_num, params, irrel_params, cf_params, pin, t)
        if tt > 10:
            tt = 0
        tt = tt + 1
        t = t + 1
        # if max_try>2000:
        #     raise ValueError("Max tries exceeded for non-overlapping feature circle generation.")
    logic_rules = get_logic_rules(is_positive, params, cf_params, irrel_params)
    all_objs = objs + tp_objs
    return all_objs, logic_rules
