# Created by jing at 25.02.25

import random
import math

from scripts import config
from scripts.utils import encode_utils, data_utils


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


def overlap_circle_features(fixed_props, is_positive, cluster_num, obj_quantities, obj_dist_factor, pin):
    objs = []
    group_sizes = {"s": range(2, 4), "m": range(3, 5), "l": range(2, 7)}.get(obj_quantities, range(2, 4))
    if not is_positive:
        new_cluster_num = random.randint(1, cluster_num + 2)
        while new_cluster_num == cluster_num:
            new_cluster_num = random.randint(1, cluster_num + 2)
        cluster_num = new_cluster_num
    # Generate evenly distributed group centers on the circumference
    angles = get_circumference_angles(cluster_num)
    if not is_positive and random.random() < 0.3:
        cir_so = (0.3 + random.random() * 0.5) * 0.7
        is_positive = True
    else:
        cir_so = 0.3 + random.random() * 0.5

    big_color = random.choice(config.color_large_exclude_gray)
    obj_size = cir_so * 0.07
    obj = encode_utils.encode_objs(x=0.5, y=0.5, size=cir_so, color=big_color,
                                   shape="circle", line_width=-1, solid=True)
    objs.append(obj)

    for a_i in range(cluster_num):
        group_size = random.choice(group_sizes)
        if is_positive:
            if "shape" in fixed_props:
                shapes = [random.choice(config.bk_shapes[1:])] * group_size
            else:
                shapes = data_utils.random_select_unique_mix(config.bk_shapes[1:], group_size)
            if "color" in fixed_props:
                selected_color = random.choice(config.color_large_exclude_gray)
                while selected_color == big_color:
                    selected_color = random.choice(config.color_large_exclude_gray)
                colors = [selected_color] * group_size
            else:
                colors = data_utils.random_select_unique_mix(config.color_large_exclude_gray, group_size)
        else:
            cf_params = data_utils.get_proper_sublist(fixed_props)
            if "shape" in cf_params:
                shapes = [random.choice(config.bk_shapes[1:])] * group_size
            else:
                shapes = data_utils.random_select_unique_mix(config.bk_shapes[1:], group_size)

            if "color" in cf_params:
                selected_color = random.choice(config.color_large_exclude_gray)
                while selected_color == big_color:
                    selected_color = random.choice(config.color_large_exclude_gray)
                colors = [selected_color] * group_size
            else:
                colors = data_utils.random_select_unique_mix(config.color_large_exclude_gray, group_size)

        # Get multiple nearby positions along the circumference
        positions = get_circumference_positions(angles[a_i], cir_so / 2, group_size, obj_dist_factor)
        for i in range(len(positions)):
            obj = encode_utils.encode_objs(x=positions[i][0], y=positions[i][1],
                                           size=obj_size,
                                           color=colors[i],
                                           shape=shapes[i],
                                           line_width=-1,
                                           solid=True)
            objs.append(obj)
    return objs
