# Created by jing at 02.03.25


import random

from scripts import config
from scripts.utils import encode_utils, data_utils, pos_utils
from scripts.utils.shape_utils import overlaps, overflow
from scripts.utils.pos_utils import get_feature_square_positions


def feature_closure_square(is_positive, clu_num, params, pin):
    cluster_dist = 0.2
    x_min = 0.25
    x_max = 0.75
    y_min = 0.5
    y_max = 0.95
    obj_num = 4

    objs = []
    # Generate random anchors for clusters ensuring proper distance
    group_anchors = []
    for _ in range(clu_num):
        group_anchors.append(pos_utils.generate_random_anchor(group_anchors, cluster_dist, x_min, x_max, y_min, y_max))

    for i in range(clu_num):
        clu_size = ({1: 0.3 + random.random() * 0.2,
                     2: 0.3 + random.random() * 0.1,
                     3: 0.3 + random.random() * 0.1,
                     4: 0.2 + random.random() * 0.1
                     }.get(clu_num, 0.3))
        obj_size = clu_size * (0.3 + random.random() * 0.1)

        positions = get_feature_square_positions(group_anchors[i], clu_size)
        shapes = ["pac_man"] * obj_num
        # 50% of the negative images, random object positions but other properties as same as positive
        if not is_positive and pin and random.random() > 0.5:
            start_angles = random.sample(range(0, 360), obj_num)
            end_angles = [angle + 270 for angle in start_angles]
            is_positive = True
        else:
            start_angles = [90, 270, 0, 180]
            end_angles = [angle + 270 for angle in start_angles]
        if is_positive:
            if "color" in params or random.random() < 0.5:
                colors = random.choices(["blue", "red"], k=obj_num)
            else:
                colors = data_utils.random_select_unique_mix(config.color_large_exclude_gray, obj_num)

            if "size" in params or random.random() < 0.5:
                sizes = [obj_size] * obj_num
            else:
                sizes = [random.uniform(obj_size * 0.8, obj_size * 1) for _ in range(obj_num)]
        else:
            cf_params = data_utils.get_proper_sublist(params)
            if "color" in cf_params:
                colors = random.choices(["blue", "red"], k=obj_num)
            else:
                colors = data_utils.random_select_unique_mix(config.color_large_exclude_gray, obj_num)

            if "size" in cf_params:
                sizes = [obj_size] * obj_num
            else:
                sizes = [random.uniform(obj_size * 0.6, obj_size * 1) for _ in range(obj_num)]
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
                    start_angle=start_angles[i],
                    end_angle=end_angles[i]
                ))
        except Exception as e:
            raise e

    return objs


def non_overlap_feature_square(params, is_positive, clu_num, pin):
    objs = feature_closure_square(is_positive, clu_num, params, pin)
    t = 0
    tt = 0
    max_try = 2000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = feature_closure_square(is_positive, clu_num, params, pin)
        if tt > 10:
            tt = 0
        tt = tt + 1
        t = t + 1
    return objs