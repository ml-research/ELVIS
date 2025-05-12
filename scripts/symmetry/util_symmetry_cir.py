# Created by jing at 01.03.25

import random
import math

from scripts import config
from scripts.utils import encode_utils, data_utils


def get_circumference_angles(cluster_num):
    """
    Generate evenly spaced points on the circumference of a circle.
    """
    angles = []
    for i in range(cluster_num):
        angle = (2 * math.pi / cluster_num) * i
        angles.append(angle)
    return angles


def get_symmetry_surrounding_positions(angle, radius, dtype, num_points=2):
    """
    Generate multiple points near the given center, ensuring they remain on the circumference.
    """
    positions = []
    for p_i in range(num_points):
        angle_offset = 0.3 * p_i
        shifted_angle = angle + angle_offset
        x = 0.5 + radius * math.cos(shifted_angle)
        y = 0.5 + radius * math.sin(shifted_angle)
        if dtype:
            x_symmetry = 0.5 - radius * math.cos(shifted_angle)

        else:
            x_symmetry = 0.5 - radius * math.sin(shifted_angle)
        positions.append((x, y))
        positions.append((x_symmetry, y))

    return positions


def get_surrounding_positions(center, radius, num_points=2):
    """
    Generate multiple points near the given center, ensuring they remain on the circumference.
    """
    positions = []
    for _ in range(num_points * 2):
        angle_offset = random.uniform(-0.2, 0.2)  # Small random variation
        angle = math.atan2(center[1] - 0.5, center[0] - 0.5) + angle_offset
        if random.random() < 0.5:
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
        else:
            x = 0.5 - radius * math.cos(angle)
            y = 0.5 - radius * math.sin(angle)
        positions.append((x, y))
    return positions


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


def feature_symmetry_circle(params, is_positive, clu_num=1):
    obj_size = 0.05
    objs = []
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
    if "count" in params and not is_positive:
        clu_num = data_utils.neg_clu_num(clu_num, 1, clu_num + 2)

    # Generate evenly distributed group centers on the circumference
    angles = get_circumference_angles(clu_num)
    group_centers = get_circumference_points(clu_num, 0.5, 0.5, cir_so)

    for a_i in range(clu_num):
        if is_positive:
            group_obj_num = random.randint(2, 4)
            positions = get_symmetry_surrounding_positions(angles[a_i], cir_so / 2, is_positive, group_obj_num)

            if "shape" in params:
                shapes = [random.choice(config.bk_shapes[1:])] * group_obj_num
                shapes = data_utils.duplicate_maintain_order(shapes, 2)

            else:
                shapes = data_utils.random_select_unique_mix(config.bk_shapes[1:], group_obj_num)
                shapes = data_utils.duplicate_maintain_order(shapes, 2)
            if "color" in params:
                colors = [random.choice(config.color_large_exclude_gray)] * group_obj_num
                colors = data_utils.duplicate_maintain_order(colors, 2)
            else:
                colors = data_utils.random_select_unique_mix(config.color_large_exclude_gray, group_obj_num)
                colors = data_utils.duplicate_maintain_order(colors, 2)

            if "size" in params:
                sizes = [obj_size] * group_obj_num
                sizes = data_utils.duplicate_maintain_order(sizes, 2)
            else:
                sizes = [random.uniform(obj_size * 0.6, obj_size * 1.5) for _ in range(group_obj_num)]
                sizes = data_utils.duplicate_maintain_order(sizes, 2)

        else:
            group_obj_num = random.randint(2, 4)
            # shape
            if "shape" in params:
                shapes = data_utils.random_select_unique_mix(config.bk_shapes[1:], group_obj_num)
                shapes = data_utils.duplicate_maintain_order(shapes, 2)
            else:
                shapes = [random.choice(config.bk_shapes[1:])] * group_obj_num
                shapes = data_utils.duplicate_maintain_order(shapes, 2)
            if "color" in params:
                colors = data_utils.random_select_unique_mix(config.color_large_exclude_gray, group_obj_num)
                colors = data_utils.duplicate_maintain_order(colors, 2)
            else:
                colors = [random.choice(config.color_large_exclude_gray)] * group_obj_num
                colors = data_utils.duplicate_maintain_order(colors, 2)
            if "size" in params:
                sizes = [random.uniform(obj_size * 0.9, obj_size * 2) for _ in range(group_obj_num)]
                sizes = data_utils.duplicate_maintain_order(sizes, 2)

            else:
                sizes = [obj_size] * group_obj_num
                sizes = data_utils.duplicate_maintain_order(sizes, 2)
            if random.random() < 0.3:
                positions = get_surrounding_positions(group_centers[a_i], cir_so * 1, group_obj_num)
            else:
                positions = get_symmetry_surrounding_positions(angles[a_i], cir_so / 2, is_positive, group_obj_num)
        try:
            for i in range(len(positions)):
                objs.append(encode_utils.encode_objs(
                    x=positions[i][0],
                    y=positions[i][1],
                    size=sizes[i],
                    color=colors[i],
                    shape=shapes[i],
                    line_width=-1,
                    solid=True
                ))
        except IndexError:
            raise IndexError
    return objs
