# Created by jing at 27.02.25
import math
import random
import numpy as np
from scipy.spatial.distance import cdist

from scripts import config
from scripts.utils import encode_utils
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


def similarity_palette(obj_size, params, is_positive, clu_num, quantity):
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
    center_x, center_y = 0.5, 0.5
    max_radius = 0.4
    min_radius = 0.15
    num_rings = 6  # Increased number of rings to distribute objects more evenly
    angle_step = 2 * math.pi / clu_num
    angle_gap = angle_step * 0.1  # Reduce overlap by adding a small gap between sectors

    # Define the number of objects based on quantity parameter
    quantity_map = {"s": 3, "m": 6, "l": 9}  # Number of objects per cluster
    num_objects_per_cluster = quantity_map.get(quantity, 6)  # Default to 'm' if invalid input

    for i in range(clu_num):
        cluster_objects = []

        color = random.choice(config.color_large_exclude_gray) if "color" in params and is_positive else None
        shape = random.choice(config.bk_shapes[1:]) if "shape" in params and is_positive else None
        size = obj_size if "size" in params and is_positive else random.uniform(0.5 * obj_size, 1.5 * obj_size)

        sector_start = i * angle_step + angle_gap / 2
        sector_end = (i + 1) * angle_step - angle_gap / 2

        ring_spacing = (max_radius - min_radius) / (num_rings - 1)

        for ring in range(num_rings):
            radius = min_radius + ring * ring_spacing
            num_objects = min(int(2 * math.pi * radius / (2.0 * obj_size)), num_objects_per_cluster)

            for j in range(num_objects):
                angle = sector_start + (sector_end - sector_start) * (j / max(1, num_objects - 1))
                obj_x = center_x + radius * math.cos(angle)
                obj_y = center_y + radius * math.sin(angle)

                obj_color = color if color else random.choice(config.color_large_exclude_gray)
                obj_shape = shape if shape else random.choice(config.bk_shapes[1:])
                obj_size_variant = size if size else random.uniform(0.5 * obj_size, 1.5 * obj_size)

                cluster_objects.append(encode_utils.encode_objs(
                    x=obj_x,
                    y=obj_y,
                    size=obj_size_variant, color=obj_color, shape=obj_shape,
                    line_width=-1, solid=True,
                ))
        objs.extend(cluster_objects)

    return objs


def non_overlap_palette(params, is_positive, clu_num, quantity):
    obj_size = 0.05
    objs = similarity_palette(obj_size, params, is_positive, clu_num, quantity)
    # return objs
    t = 0
    tt = 0
    max_try = 1000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = similarity_palette(obj_size, params, is_positive, clu_num, quantity)
        if tt > 10:
            tt = 0
            obj_size = obj_size * 0.90
        tt = tt + 1
        t = t + 1
    return objs
