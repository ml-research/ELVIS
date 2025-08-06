# Created by jing at 25.02.25

import random
import math

from scripts import config
from scripts.utils import encode_utils, data_utils, pos_utils


def get_circumference_angles(cluster_num):
    """
    Generate evenly spaced points on the circumference of a circle.
    """
    angles = []
    shift = random.random() * math.pi
    for i in range(cluster_num):
        angle = (2 * math.pi / cluster_num) * i + shift
        angles.append(angle)
    return angles


def get_circumference_positions(angle, radius, num_points=2, obj_dist_factor=1):
    """
    Generate multiple points near the given center, ensuring they remain on the circumference.
    """

    positions = []
    for p_i in range(num_points):
        angle_offset = 0.3 * p_i
        shifted_angle = angle + angle_offset
        x = 0.5 + radius * math.cos(shifted_angle) * obj_dist_factor
        y = 0.5 + radius * math.sin(shifted_angle) * obj_dist_factor
        positions.append((x, y))
    return positions


def overlap_circle_features(params, irrel_params, is_positive, cluster_num, obj_quantities, obj_dist_factor, pin):
    cf_params = data_utils.get_proper_sublist(params + ["proximity"])

    group_size = config.standard_quantity_dict[obj_quantities]
    objs = []

    setting = {
        "big_radius": 0.3 + random.random() * 0.5,

    }
    setting["obj_size"] = setting["big_radius"] * 0.07
    logic = {
        "shape": random.sample(config.all_shapes, 2),
        "color": random.sample(config.color_large_exclude_gray, 2),

    }

    if not is_positive:
        new_cluster_num = random.randint(1, cluster_num + 2)
        while new_cluster_num == cluster_num:
            new_cluster_num = random.randint(1, cluster_num + 2)
        cluster_num = new_cluster_num
    # Generate evenly distributed group centers on the circumference
    angles = get_circumference_angles(cluster_num)

    big_color = random.choice(config.color_large_exclude_gray)

    obj_size = setting["obj_size"]
    obj = encode_utils.encode_objs(x=0.5, y=0.5, size=setting["big_radius"], color=big_color,
                                   shape="circle", line_width=-1, solid=True,
                                   group_id=-1)
    objs.append(obj)

    for a_i in range(cluster_num):
        if "count" in params and is_positive or ("count" in cf_params and not is_positive):
            pass
        else:
            group_size = max(2, group_size + random.choice([-2, -1, 1, 2]))

        if "shape" in params or ("shape" in cf_params and not is_positive):
            shapes = random.choices(logic["shape"], k=group_size)
        else:
            try:
                shapes = random.choices(config.all_shapes, k=group_size)
            except ValueError:
                print("")

        if "color" in params and is_positive or ("color" in cf_params and not is_positive):
            colors = random.choices(logic["color"], k=group_size)
        else:
            colors = random.choices(config.color_large_exclude_gray, k= group_size)

        if "size" in params or ("size" in cf_params and not is_positive):
            sizes = [obj_size] * group_size
        else:
            sizes = [random.uniform(obj_size * 0.4, obj_size * 1.5) for _ in range(group_size)]

        if "proximity" in cf_params and not is_positive or (is_positive):
            cir_so = setting["big_radius"] * 0.8
            obj_dist_factor = 1
            positions = get_circumference_positions(angles[a_i], cir_so / 2, group_size, obj_dist_factor)
        else:
            positions = pos_utils.get_random_positions(group_size, obj_size)
        group_ids = [a_i] * group_size
        objs += encode_utils.encode_scene(positions, sizes, colors, shapes, group_ids, is_positive)
    logics = get_logics(is_positive, params, cf_params, irrel_params)
    return objs, logics


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
