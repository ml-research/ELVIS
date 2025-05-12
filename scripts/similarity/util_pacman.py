# Created by jing at 27.02.25
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


# def similarity_pacman(obj_size, is_positive, clu_num=1, fixed_props="", obj_quantity="s"):
#     if not is_positive:
#         new_clu_num = clu_num
#         while new_clu_num == clu_num:
#             new_clu_num = random.randint(1, clu_num + 1)
#         clu_num = new_clu_num
#
#     colors = random.sample(config.color_large_exclude_gray, clu_num)
#     objs = []
#     angles = [0, 60, 120, 180, 240, 300]
#     positions = generate_positions(clu_num, 5, obj_size)
#
#     if obj_quantity == "s":
#         max_clu_size = 2
#     elif obj_quantity == "m":
#         max_clu_size = 3
#     else:
#         max_clu_size = 5
#     cluster_size = int(random.uniform(1, max_clu_size))
#     # draw circles
#     for clu_i in range(clu_num):
#         if "color" in fixed_props:
#             color = colors[clu_i]
#         else:
#             color = colors[0]
#         if "size" not in fixed_props:
#             obj_size = random.uniform(0.03, 0.1)
#         if "count" in fixed_props:
#             if not is_positive:
#                 cluster_size = int(random.uniform(1, max_clu_size))
#             else:
#                 cluster_size = cluster_size
#         for i in range(cluster_size):
#             obj = encode_utils.encode_objs(x=positions[clu_i, i][0],
#                                            y=positions[clu_i, i][1],
#                                            size=obj_size, color=color, shape="pac_man",
#                                            line_width=-1, solid=True,
#                                            start_angle=angles[clu_i],
#                                            end_angle=angles[clu_i] + 300
#                                            )
#             objs.append(obj)
#     return objs

def similarity_pacman(obj_size, is_positive, clu_num=1, fixed_props="", obj_quantity="s"):

    """
    Generate a set of pacman-shaped objects arranged in clusters.

    Parameters:
    - obj_size: Default size of the objects.
    - is_positive: Determines if the pattern follows a positive rule.
    - clu_num: Number of clusters.
    - fixed_props: Fixed properties such as color, size, or count.
    - obj_quantity: Defines the maximum cluster size ("s" for small, "m" for medium, "l" for large).

    Returns:
    - List of encoded pacman objects.
    """

    # Adjust cluster number if negative pattern
    if not is_positive:
        new_clu_num = clu_num
        while new_clu_num == clu_num:
            new_clu_num = random.randint(1, clu_num + 1)
        clu_num = new_clu_num

    # Define colors and positions
    colors = random.sample(config.color_large_exclude_gray, clu_num)
    angles = [0, 60, 120, 180, 240, 300]  # Fixed angle positions
    positions = generate_positions(clu_num, 5, obj_size)

    # Determine cluster size based on quantity
    max_clu_size = {"s": 2, "m": 3, "l": 5}.get(obj_quantity, 2)
    cluster_size = random.randint(1, max_clu_size)

    objs = []

    # Generate objects
    for clu_i in range(clu_num):
        color = colors[clu_i] if "color" in fixed_props else colors[0]
        obj_size_variant = obj_size if "size" in fixed_props else random.uniform(0.03, 0.1)

        # Adjust cluster size for negative patterns
        if "count" in fixed_props and not is_positive:
            cluster_size = random.randint(1, max_clu_size)

        for i in range(cluster_size):
            obj = encode_utils.encode_objs(
                x=positions[clu_i, i][0],
                y=positions[clu_i, i][1],
                size=obj_size_variant,
                color=color,
                shape="pac_man",
                line_width=-1,
                solid=True,
                start_angle=angles[clu_i],
                end_angle=angles[clu_i] + 300
            )
            objs.append(obj)

    return objs

def non_overlap_pacman(params, is_positive, clu_num, obj_quantity):
    obj_size = 0.05
    objs = similarity_pacman(obj_size, is_positive, clu_num, params, obj_quantity)
    t = 0
    tt = 0
    max_try = 1000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = similarity_pacman(obj_size, is_positive, clu_num, params, obj_quantity)
        if tt > 10:
            tt = 0
            obj_size = obj_size * 0.90
        tt = tt + 1
        t = t + 1
    return objs
