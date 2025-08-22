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


def get_symmetry_on_cir_positions(center, radius, num_points, axis=0, min_dist_to_axis=0.1, min_dist_between_pairs=0.2, max_attempts=100):
    """
    Generate symmetric pairs of points on the circumference, with angle_offset sampled as evenly spaced positions.
    axis: symmetry axis in degrees (-45, 0, 45, 90)
    """
    all_positions = []
    min_offset = math.pi / 12  # 15 degrees
    max_offset = math.pi / 12 * 11  # 180 degrees
    axis_rad = math.radians(axis)
    axis_dx = math.cos(axis_rad)
    axis_dy = math.sin(axis_rad)
    axis_point = (0.5, 0.5)
    # Generate evenly spaced offsets in both positive and negative ranges
    offsets_pos = np.linspace(min_offset, max_offset, sum(num_points)+10)

    np.random.shuffle(offsets_pos)  # Shuffle offsets to ensure randomness
    counter = 0
    for grp_pairs in num_points:
        positions = []
        for i in range(grp_pairs):
            angle_offset = offsets_pos[counter]
            counter += 1
            perp_angle = axis_rad + math.pi / 2 + angle_offset
            x = axis_point[0] + radius * math.cos(perp_angle)
            y = axis_point[1] + radius * math.sin(perp_angle)

            # Symmetric point across axis
            px, py = x - axis_point[0], y - axis_point[1]
            proj = px * axis_dx + py * axis_dy
            sx = px - 2 * (proj * axis_dx)
            sy = py - 2 * (proj * axis_dy)
            x_sym = axis_point[0] + sx
            y_sym = axis_point[1] + sy

            # Check minimum distance to axis
            dist_to_axis = abs(px * axis_dy - py * axis_dx)
            dist_to_axis_sym = abs(sx * axis_dy - sy * axis_dx)
            if dist_to_axis < min_dist_to_axis or dist_to_axis_sym < min_dist_to_axis:
                continue

            # Check minimum distance to other positions
            too_close = False
            for (ox, oy) in positions:
                if math.hypot(x - ox, y - oy) < min_dist_between_pairs or math.hypot(x_sym - ox, y_sym - oy) < min_dist_between_pairs:
                    too_close = True
                    break
            if not too_close:
                positions.append((x, y))
                positions.append((x_sym, y_sym))
        all_positions.append(positions)
    return all_positions


def symmetry_solar_sys(obj_size, is_positive, clu_num, params, irrel_params, cf_params, obj_quantity, axis_list):
    cir_so = 0.6 + random.random() * 0.1
    logic = {
        "shape": ["cross", "plus"],
        "color": ["red", "blue", "green", "yellow", "purple"]
    }

    objs = [encode_utils.encode_objs(
        x=0.5,
        y=0.5,
        size=cir_so,
        color=random.choice(config.color_large_exclude_gray),
        shape="circle",
        line_width=-1,
        solid=True,
        group_id=-1
    )]

    dist = 1.2

    if "count" in params and not is_positive:
        clu_num = data_utils.neg_clu_num(clu_num, 1, clu_num + 2)


    # Generate evenly distributed group centers on the circumference
    group_centers = get_circumference_points(clu_num, 0.5, 0.5, cir_so)
    grp_obj_nums = [config.standard_quantity_dict[obj_quantity] - 3 for i in range(clu_num)]

    if "axis" in params and is_positive or (not is_positive and "axis" in cf_params):
        sym_axis = axis_list[0]
    else:
        sym_axis = random.choice(axis_list)

    all_positions = get_symmetry_on_cir_positions(group_centers, cir_so * random.uniform(0.3, 0.6), grp_obj_nums, sym_axis)

    invariant_shape = random.choice(config.all_shapes)
    invariant_color = random.choice(config.color_large_exclude_gray)

    if not is_positive and "symmetry" not in cf_params:
        all_positions = pos_utils.get_almost_symmetry_positions(group_centers, cir_so * random.uniform(0.3, 0.6), grp_obj_nums)
    for a_i in range(clu_num):
        grp_obj_num = grp_obj_nums[a_i]
        if "shape" in params and is_positive or (not is_positive and "shape" in cf_params):
            shapes = [random.choice(logic["shape"]) for _ in range(grp_obj_num)]
            shapes = data_utils.duplicate_maintain_order(shapes, 2)
        else:
            shapes = [random.choice(config.all_shapes) for _ in range(grp_obj_num * 2)]
        if "shape" in irrel_params:
            shapes = [invariant_shape] * grp_obj_num * 2
        if "color" in params and is_positive or (not is_positive and "color" in cf_params):
            colors = [random.choice(logic["color"]) for _ in range(grp_obj_num)]
            colors = data_utils.duplicate_maintain_order(colors, 2)
        else:
            colors = [random.choice(config.color_large_exclude_gray) for _ in range(grp_obj_num * 2)]

        if "color" in irrel_params:
            colors = [invariant_color] * grp_obj_num * 2
        if "size" in params and is_positive or (not is_positive and "size" in cf_params):
            sizes = [obj_size] * grp_obj_num
            sizes = data_utils.duplicate_maintain_order(sizes, 2)
        else:
            sizes = [random.uniform(obj_size * 0.5, obj_size * 1.5) for _ in range(grp_obj_num * 2)]

        if "size" in irrel_params:
            sizes = [obj_size] * grp_obj_num
            sizes = data_utils.duplicate_maintain_order(sizes, 2)

        positions = all_positions[a_i]
        grp_ids = [a_i] * grp_obj_num * 2
        objs += encode_utils.encode_scene(positions, sizes, colors, shapes, grp_ids, is_positive)
    return objs


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


def non_overlap_soloar_sys(params, irrel_params, is_positive, clu_num, obj_quantity, pin):

    obj_size = 0.05
    sym_axis = [-45, 0, 45, 90]
    cf_params = data_utils.get_proper_sublist(params + ["symmetry"])
    objs = symmetry_solar_sys(obj_size, is_positive, clu_num, params, irrel_params, cf_params, obj_quantity, sym_axis)
    logics = get_logics(is_positive, params, cf_params, irrel_params)
    return objs, logics
