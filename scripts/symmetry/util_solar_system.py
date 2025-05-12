# Created by jing at 27.02.25
import random
import numpy as np
import math
from scripts import config
from scripts.utils.shape_utils import overlaps, overflow
from scripts.utils import pos_utils, encode_utils, data_utils


def get_circumference_points(cluster_num, x, y, radius):
    """
    Generate evenly spaced points on the circumference of a circle.
    """
    points = []
    for i in range(cluster_num):
        angle = (2 * math.pi / cluster_num) * i
        cx = x + radius * math.cos(angle)
        cy = y + radius * math.sin(angle)
        points.append((cx, cy))
    return points


def get_surrounding_positions(center, radius, num_points):
    """
    Generate multiple points near the given center, ensuring they remain on the circumference.
    """
    all_positions = []
    for i in range(len(center)):
        positions = []
        for _ in range(num_points[i] * 2):
            angle_offset = random.uniform(-0.2, 0.2)  # Small random variation
            angle = math.atan2(center[i][1] - 0.5, center[i][0] - 0.5) + angle_offset
            if random.random() < 0.5:
                x = 0.5 + radius * math.cos(angle)
                y = 0.5 + radius * math.sin(angle)
            else:
                x = 0.5 - radius * math.cos(angle)
                y = 0.5 - radius * math.sin(angle)
            positions.append((x, y))
        all_positions.append(positions)
    return all_positions


def get_symmetry_on_cir_positions(center, radius, num_points):
    """
    Generate multiple points near the given center, ensuring they remain on the circumference.
    """
    all_positions = []
    for i in range(len(center)):
        positions = []
        angle = random.uniform(0, math.pi)

        for p_i in range(int(num_points[i])):
            angle_offset = 0.3 * p_i
            shifted_angle = angle + angle_offset
            x_right = 0.5 + radius * math.cos(shifted_angle)
            x_left = 0.5 - radius * math.cos(shifted_angle)

            y = 0.5 + radius * math.sin(shifted_angle)
            positions.append((x_right, y))
            positions.append((x_left, y))
        all_positions.append(positions)
    return all_positions


def symmetry_solar_sys(obj_size, is_positive, clu_num, params):
    objs = []
    if obj_size < 0.03:
        obj_size = 0.03

    shape = "circle"
    color = random.choice(config.color_large_exclude_gray)
    cir_so = 0.3 + random.random() * 0.1
    objs.append(encode_utils.encode_objs(
        x=0.5,
        y=0.5,
        size=cir_so,
        color=color,
        shape=shape,
        line_width=-1,
        solid=True
    ))
    dist = 1.2

    if "count" in params and not is_positive:
        clu_num = data_utils.neg_clu_num(clu_num, 1, clu_num + 2)

    # Generate evenly distributed group centers on the circumference
    group_centers = get_circumference_points(clu_num, 0.5, 0.5, cir_so)

    group_obj_num = [random.randint(2, 4) for i in range(clu_num)]

    if not is_positive and random.random() < 0.3:
        all_positions = pos_utils.get_almost_symmetry_positions(group_centers, cir_so * dist, group_obj_num)
        is_positive = True
    else:
        all_positions = get_symmetry_on_cir_positions(group_centers, cir_so * dist, group_obj_num)
    for a_i in range(clu_num):
        if is_positive:
            # group_obj_num = random.randint(2, 4)

            if "shape" in params:
                shapes = [random.choice(config.bk_shapes[1:])] * group_obj_num[a_i]
                shapes = data_utils.duplicate_maintain_order(shapes, 2)

            else:
                shapes = data_utils.random_select_unique_mix(config.bk_shapes[1:], group_obj_num[a_i])
                shapes = data_utils.duplicate_maintain_order(shapes, 2)
            if "color" in params:
                colors = [random.choice(config.color_large_exclude_gray)] * group_obj_num[a_i]
                colors = data_utils.duplicate_maintain_order(colors, 2)
            else:
                colors = data_utils.random_select_unique_mix(config.color_large_exclude_gray, group_obj_num[a_i])
                colors = data_utils.duplicate_maintain_order(colors, 2)

            if "size" in params:
                sizes = [obj_size] * group_obj_num[a_i]
                sizes = data_utils.duplicate_maintain_order(sizes, 2)
            else:
                sizes = [random.uniform(obj_size * 0.8, obj_size * 1) for _ in range(group_obj_num[a_i])]
                sizes = data_utils.duplicate_maintain_order(sizes, 2)

        else:
            # shape
            if "shape" in params:
                shapes = data_utils.random_select_unique_mix(config.bk_shapes[1:], group_obj_num[a_i])
                shapes = data_utils.duplicate_maintain_order(shapes, 2)
            else:
                shapes = [random.choice(config.bk_shapes[1:])] * group_obj_num[a_i]
                shapes = data_utils.duplicate_maintain_order(shapes, 2)
            if "color" in params:
                colors = data_utils.random_select_unique_mix(config.color_large_exclude_gray, group_obj_num[a_i])
                colors = data_utils.duplicate_maintain_order(colors, 2)
            else:
                colors = [random.choice(config.color_large_exclude_gray)] * group_obj_num[a_i]
                colors = data_utils.duplicate_maintain_order(colors, 2)
            if "size" in params:
                sizes = [random.uniform(obj_size * 0.9, obj_size * 1) for _ in range(group_obj_num[a_i])]
                sizes = data_utils.duplicate_maintain_order(sizes, 2)

            else:
                sizes = [obj_size] * group_obj_num[a_i]
                sizes = data_utils.duplicate_maintain_order(sizes, 2)
            if random.random() < 0.3:
                positions = get_surrounding_positions(group_centers, cir_so * dist, group_obj_num)
            else:
                positions = get_symmetry_on_cir_positions(group_centers, cir_so * dist, group_obj_num)

        for i in range(len(all_positions[a_i])):
            objs.append(encode_utils.encode_objs(
                x=all_positions[a_i][i][0],
                y=all_positions[a_i][i][1],
                size=sizes[i],
                color=colors[i],
                shape=shapes[i],
                line_width=-1,
                solid=True
            ))
    return objs


def non_overlap_soloar_sys(params, is_positive, clu_num):
    obj_size = 0.05

    objs = symmetry_solar_sys(obj_size, is_positive, clu_num, params)
    t = 0
    tt = 0
    max_try = 1000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = symmetry_solar_sys(obj_size, is_positive, clu_num, params)
        if tt > 10:
            tt = 0
            obj_size = obj_size * 0.90
        tt = tt + 1
        t = t + 1
    return objs
