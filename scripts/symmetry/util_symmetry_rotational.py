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


def get_symmetry_surrounding_positions(num_points=2, symmetry_axis=0, min_dist_to_axis=0.2, min_dist_between_pairs=0.2, max_attempts=100):
    """
    Generate num_points pairs of symmetric positions along the given symmetry axis.
    symmetry_axis: axis in degrees (-45, 0, 45, 90)
    min_dist_to_axis: minimum distance from each object to the axis
    """
    positions = []
    axis_rad = math.radians(symmetry_axis)
    axis_dx = math.cos(axis_rad)
    axis_dy = math.sin(axis_rad)
    axis_point = (0.5, 0.5)
    for _ in range(num_points):
        for attempt in range(max_attempts):
            dist = random.uniform(min_dist_to_axis, 0.4)
            min_offset = math.pi / 12  # 15 degrees in radians
            max_offset = math.pi / 4  # 45 degrees in radians
            if random.random() < 0.5:
                angle_offset = random.uniform(-max_offset, -min_offset)
            else:
                angle_offset = random.uniform(min_offset, max_offset)

            perp_angle = axis_rad + math.pi / 2 + angle_offset
            x = axis_point[0] + dist * math.cos(perp_angle)
            y = axis_point[1] + dist * math.sin(perp_angle)
            px, py = x - axis_point[0], y - axis_point[1]
            proj = px * axis_dx + py * axis_dy
            sx = px - 2 * (proj * axis_dx)
            sy = py - 2 * (proj * axis_dy)
            x_sym = axis_point[0] + sx
            y_sym = axis_point[1] + sy
            # Check minimum distance to all existing positions
            too_close = False
            for (ox, oy) in positions:
                if math.hypot(x - ox, y - oy) < min_dist_between_pairs or math.hypot(x_sym - ox, y_sym - oy) < min_dist_between_pairs:
                    too_close = True
                    break
            if not too_close:
                positions.append((x, y))
                positions.append((x_sym, y_sym))
                break
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
        "principle": "symmetry",

    }
    return logic


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


def rotational_symmetry_pattern(params, irrel_params, is_positive, clu_num, obj_quantity, pin):
    obj_size = 0.05
    cf_params = data_utils.get_proper_sublist(params + ["symmetry"])
    cir_so = 0.3 + random.random() * 0.1
    center = (0.5, 0.5)
    if config.shape_quantity == "s":
        invariant_shape = random.choice(config.s_shapes)
    else:
        invariant_shape = random.choice(config.all_shapes)
    invariant_color = random.choice(config.color_large_exclude_gray)
    grp_obj_num = config.standard_quantity_dict[obj_quantity]
    logic = {
        "shape": ["triangle", "circle"],
        "color": ["red", "blue", "green", "yellow", "purple"]
    }
    # Define available pattern interfaces
    pattern_funcs = [
        symmetry_pattern_type1,
        symmetry_pattern_type2,
        # symmetry_pattern_type3,
        # Add more pattern functions as needed
    ]

    # Compute branch centers and angles
    angles = [(2 * math.pi / clu_num) * i for i in range(clu_num)]
    branch_centers = [
        (center[0] + cir_so * math.cos(angle), center[1] + cir_so * math.sin(angle))
        for angle in angles
    ]

    all_positions = []
    for i in range(clu_num):
        if not is_positive and "symmetry" not in cf_params:
            all_positions.append(get_almost_radius_symmetry_positions(center, angles[i], grp_obj_num))
        else:
            # pattern_func = random.choice(pattern_funcs)
            # positions = pattern_func(branch_centers[i], angles[i], grp_obj_num)
            positions = symmetry_pattern_type1(center, angles[i], grp_obj_num)
            all_positions.append(positions)

    objs = []
    for a_i in range(clu_num):
        if "shape" in params and is_positive or (not is_positive and "shape" in cf_params):
            shapes = [random.choice(logic["shape"]) for _ in range(grp_obj_num)]
        else:
            if config.shape_quantity == "s":
                shapes = [random.choice(config.s_shapes) for _ in range(grp_obj_num * 2)]
            else:
                shapes = [random.choice(config.all_shapes) for _ in range(grp_obj_num)]
        if "shape" in irrel_params:
            shapes = [invariant_shape] * grp_obj_num

        if "color" in params and is_positive or (not is_positive and "color" in cf_params):
            colors = [random.choice(logic["color"]) for _ in range(grp_obj_num)]
        else:
            colors = [random.choice(config.color_large_exclude_gray) for _ in range(grp_obj_num)]
        if "color" in irrel_params:
            colors = [invariant_color] * grp_obj_num

        if "size" in params and is_positive or (not is_positive and "size" in cf_params):
            sizes = [obj_size] * grp_obj_num
        else:
            sizes = [random.uniform(obj_size * 0.5, obj_size * 1.5) for _ in range(grp_obj_num)]
        if "size" in irrel_params:
            sizes = [obj_size] * grp_obj_num

        positions = all_positions[a_i]
        grp_ids = [a_i] * grp_obj_num
        objs += encode_utils.encode_scene(positions, sizes, colors, shapes, grp_ids, is_positive)

    # objs = encode_utils.encode_scene(all_positions, sizes, colors, shapes, grp_ids, is_positive)
    logics = get_logics(is_positive, params, cf_params, irrel_params)
    return objs, logics


def get_almost_radius_symmetry_positions(center,  angle, grp_obj_num):
    all_positions = []

    positions = symmetry_pattern_type2(center, angle, grp_obj_num)
    all_positions.extend(positions)
    return all_positions


# Example pattern interfaces
def symmetry_pattern_type1(center, angle, grp_obj_num):
    """
    Generate points along a straight line from the center outward in the direction of angle.
    """
    positions = []
    # Start from the center, move outward along the angle
    for i in range(1, grp_obj_num + 1):
        # Distance from center increases with i
        dist = 0.07 * i
        x = center[0] + dist * math.cos(angle)
        y = center[1] + dist * math.sin(angle)
        positions.append((x, y))
    return positions


def symmetry_pattern_type2(center, angle, grp_obj_num):
    """
    Generate points along a straight line from the center outward in the direction of angle with noise.
    """
    positions = []
    # Start from the center, move outward along the angle
    for i in range(1, grp_obj_num + 1):
        # Distance from center increases with i
        dist = 0.07 * i
        x = center[0] + dist * math.cos(angle) + random.uniform(-0.05, 0.05)
        y = center[1] + dist * math.sin(angle) + random.uniform(-0.05, 0.05)
        positions.append((x, y))
    return positions


def symmetry_pattern_type3(center, angle, grp_obj_num):
    # Implement pattern details here
    return [(center[0], center[1])] * grp_obj_num

# def rotational_symmetry_pattern(params, irrel_params, is_positive, clu_num, obj_quantity, pin):
#     obj_size = 0.05
#     axis_list = [-45, 0, 45, 90]
#     cf_params = data_utils.get_proper_sublist(params + ["symmetry"])
#     cir_so = 0.3 + random.random() * 0.1
#     objs = [encode_utils.encode_objs(
#         x=0.5,
#         y=0.5,
#         size=cir_so,
#         color=random.choice(config.color_large_exclude_gray),
#         shape="circle",
#         line_width=-1,
#         solid=True,
#         group_id=-1
#     )]
#
#     logic = {
#         "shape": ["hexagon", "star"],
#         "color": ["red", "blue", "green", "yellow", "purple"],
#     }
#     if "count" in params and not is_positive:
#         clu_num = data_utils.neg_clu_num(clu_num, 1, clu_num + 2)
#
#     # Generate evenly distributed group centers on the circumference
#     angles = get_circumference_angles(clu_num)
#     group_centers = get_circumference_points(clu_num, 0.5, 0.5, cir_so)
#
#     grp_obj_nums = [config.standard_quantity_dict[obj_quantity] - 3 for _ in range(clu_num)]
#
#     if "axis" in params and is_positive or (not is_positive and "axis" in cf_params):
#         sym_axis=axis_list[0]
#     else:
#         sym_axis = random.choice(axis_list)
#     if is_positive or ("symmetry" in cf_params and not is_positive):
#         if "axis" in params:
#             # with symmetry and fixed axis
#             all_positions = [get_symmetry_surrounding_positions(grp_obj_nums[a_i], sym_axis)
#                              for a_i in range(clu_num)]
#         else:
#             # with symmetry and random axis
#             axis = random.choice([-45, 0, 45, 90])
#             all_positions = [get_symmetry_surrounding_positions(grp_obj_nums[a_i], axis)
#                              for a_i in range(clu_num)]
#     else:
#         all_positions = [get_surrounding_positions(group_centers[a_i], cir_so * 1, grp_obj_nums[a_i])
#                          for a_i in range(clu_num)]
#
#     invariant_shape = random.choice(config.all_shapes)
#     invariant_color = random.choice(config.color_large_exclude_gray)
#
#     for a_i in range(clu_num):
#         grp_obj_num = grp_obj_nums[a_i]
#         if "shape" in params and is_positive or (not is_positive and "shape" in cf_params):
#             shapes = [random.choice(logic["shape"]) for _ in range(grp_obj_num)]
#             shapes = data_utils.duplicate_maintain_order(shapes, 2)
#         else:
#             shapes = [random.choice(config.all_shapes) for _ in range(grp_obj_num * 2)]
#         if "shape" in irrel_params:
#             shapes = [invariant_shape] * grp_obj_num * 2
#
#         if "color" in params and is_positive or (not is_positive and "color" in cf_params):
#             colors = [random.choice(logic["color"]) for _ in range(grp_obj_num)]
#             colors = data_utils.duplicate_maintain_order(colors, 2)
#         else:
#             colors = [random.choice(config.color_large_exclude_gray) for _ in range(grp_obj_num * 2)]
#         if "color" in irrel_params:
#             colors = [invariant_color] * grp_obj_num * 2
#
#         if "size" in params and is_positive or (not is_positive and "size" in cf_params):
#             sizes = [obj_size] * grp_obj_num
#             sizes = data_utils.duplicate_maintain_order(sizes, 2)
#         else:
#             sizes = [random.uniform(obj_size * 0.5, obj_size * 1.5) for _ in range(grp_obj_num * 2)]
#         if "size" in irrel_params:
#             sizes = [obj_size] * grp_obj_num
#             sizes = data_utils.duplicate_maintain_order(sizes, 2)
#         positions = all_positions[a_i]
#         grp_ids = [a_i] * grp_obj_num * 2
#         objs = encode_utils.encode_scene(positions, sizes, colors, shapes, grp_ids, is_positive)
#     logics = get_logics(is_positive, params, cf_params, irrel_params)
#     return objs, logics
