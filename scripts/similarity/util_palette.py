# Created by jing at 27.02.25
import math
import random
import numpy as np
from scipy.spatial.distance import cdist

from scripts import config
from scripts.utils import encode_utils, data_utils
from scripts.utils.shape_utils import overlaps, overflow


def generate_positions(n, m, t=0.1, min_range=0.1, max_range=0.9):
    num_points = n * m
    positions = []

    while len(positions) < num_points:
        candidate = np.random.uniform(min_range, max_range, size=(1, 2))

        if len(positions) == 0 or np.min(cdist(positions, candidate)) >= t:
            positions.append(candidate[0])

    positions = np.array(positions)

    return positions.reshape(n, m, 2)


def generate_cluster_positions(clu_num, num_rings, obj_size, num_objects_per_cluster):
    angle_step = 2 * math.pi / clu_num
    angle_gap = angle_step * 0.1  # Reduce overlap by adding a small gap between sectors
    center_x, center_y = 0.5, 0.5
    max_radius = 0.4
    min_radius = 0.15
    cluster_positions = []
    ring_spacing = (max_radius - min_radius) / (num_rings - 1)
    for i in range(clu_num):
        positions = []
        sector_start = i * angle_step + angle_gap / 2
        sector_end = (i + 1) * angle_step - angle_gap / 2
        for ring in range(num_rings):
            radius = min_radius + ring * ring_spacing
            num_objects = min(int(2 * math.pi * radius / (2.0 * obj_size)), num_objects_per_cluster)
            for j in range(num_objects):
                angle = sector_start + (sector_end - sector_start) * (j / max(1, num_objects - 1))
                obj_x = center_x + radius * math.cos(angle)
                obj_y = center_y + radius * math.sin(angle)
                positions.append((obj_x, obj_y))
        cluster_positions.append(positions)
    return cluster_positions


def similarity_palette(obj_size, params, irrel_params, cf_params, is_positive, clu_num, quantity):
    """
    Generate clu_num clusters of objects. Each cluster is formed as a circular sector.
    All the clusters together form a whole circle with objects evenly placed in sectors.

    Parameters:
    - obj_size: Default size of the objects.
    - params: Properties to be fixed (e.g., color, shape, size, count).
    - is_positive: Determines if the pattern follows a positive rule.
    - clu_num: Number of clusters.
    - quantity: Defines the total object quantity in the image ('s' for small, 'm' for medium, 'l' for large).

    Returns:
    - List of objects with their properties and positions.
    """
    objs = []
    num_rings = max(3, config.standard_quantity_dict[quantity]//2)
    num_objects_per_cluster = config.standard_quantity_dict[quantity]//2

    all_positions = generate_cluster_positions(clu_num, num_rings, obj_size, num_objects_per_cluster)
    logics = {
        "shape": random.choices(config.all_shapes, k=clu_num),
        "color": random.choices(config.color_large_exclude_gray, k=clu_num),
    }
    invariant_color = random.choice(config.color_large_exclude_gray)
    invariant_shape = random.choice(config.all_shapes)
    for i in range(clu_num):
        clu_size = len(all_positions[i])
        if "color" in params and is_positive or (not is_positive and "color" in cf_params):
            colors = [logics["color"][i]] * clu_size
        else:
            colors = [random.choice(config.color_large_exclude_gray) for _ in range(clu_size)]
        if "color" in irrel_params:
            colors = [invariant_color] * clu_size
        if "shape" in params and is_positive or (not is_positive and "shape" in cf_params):
            shapes = [logics["shape"][i]] * clu_size
        else:
            shapes = [random.choice(config.all_shapes) for _ in range(clu_size)]
        if "shape" in irrel_params:
            shapes = [invariant_shape] * clu_size

        if "size" in params and is_positive or (not is_positive and "size" in cf_params):
            sizes = [obj_size] * clu_size
        else:
            sizes = [random.uniform(0.5 * obj_size, 1.5 * obj_size) for _ in range(clu_size)]
        if "size" in irrel_params:
            sizes = [obj_size] * clu_size

        group_ids = [i] * clu_size
        objs += encode_utils.encode_scene(all_positions[i], sizes, colors, shapes, group_ids, is_positive)
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
        "principle": "similarity",

    }
    return logic


def non_overlap_palette(params, irrel_params, is_positive, clu_num, quantity, pin):
    obj_size = 0.05
    cf_params = data_utils.get_proper_sublist(params)

    objs = similarity_palette(obj_size, params, irrel_params, cf_params, is_positive, clu_num, quantity)
    # return objs
    t = 0
    tt = 0
    max_try = 1000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = similarity_palette(obj_size, params, irrel_params, cf_params, is_positive, clu_num, quantity)
        if tt > 10:
            tt = 0
            obj_size = obj_size * 0.90
        tt = tt + 1
        t = t + 1
    logics = get_logics(is_positive, params, cf_params, irrel_params)
    return objs, logics
