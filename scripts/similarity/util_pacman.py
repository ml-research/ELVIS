# Created by jing at 27.02.25
import random
import numpy as np
from scipy.spatial.distance import cdist

from scripts import config
from scripts.utils import encode_utils, data_utils
from scripts.utils.shape_utils import overlaps, overflow


def generate_positions(num_points, t=0.1, min_range=0.1, max_range=0.9):
    positions = []
    while len(positions) < num_points:
        candidate = np.random.uniform(min_range, max_range, size=(1, 2))

        if len(positions) == 0 or np.min(cdist(positions, candidate)) >= t:
            positions.append(candidate[0])

    positions = np.array(positions)

    return positions


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

def similarity_pacman(obj_size, is_positive, clu_num, params, irrel_params, cf_params, obj_quantity):
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

    standard_quantity_dict = {"s": 5,
                              "m": 10,
                              "l": 12,
                              "xl": 15}

    if not is_positive:
        new_clu_num = clu_num
        while new_clu_num == clu_num:
            new_clu_num = random.randint(2, clu_num + 1)
        clu_num = new_clu_num
    # Define colors and positions
    settings = {
        "angles": [0, 60, 120, 180, 240, 300],
    }
    logics = {
        "shape": "pac_man",
        "color": ["red", "blue", "green", "yellow", "purple", "orange"],
    }

    clu_sizes = []

    for clu_i in range(clu_num):
        # determine cluster size based on parameters
        if is_positive and "count" in params or (not is_positive and "count" in cf_params):
            # For positive patterns, use the fixed cluster size
            cluster_size = standard_quantity_dict[obj_quantity]
        else:
            cluster_size = standard_quantity_dict[obj_quantity] + random.choice([-2, -1, 1, 2])
        clu_sizes.append(cluster_size)

    if "count" in irrel_params:
        clu_sizes = [standard_quantity_dict[obj_quantity]] * clu_num
    raw_positions = generate_positions(sum(clu_sizes), obj_size)
    splitted_positions = np.split(raw_positions, np.cumsum(clu_sizes)[:-1])
    invariant_color = random.choice(config.color_large_exclude_gray)

    objs = []
    # Generate objects
    for clu_i in range(clu_num):
        cluster_size = clu_sizes[clu_i]
        positions = splitted_positions[clu_i]
        # determine color and size based on parameters
        if is_positive and "color" in params or (not is_positive and "color" in cf_params):
            colors = [logics["color"][clu_i]] * cluster_size
        else:
            colors = random.sample([logics["color"][clu_i]] * cluster_size + [logics["color"][(clu_i+1) % len(logics["color"])]] * cluster_size, cluster_size)
        if is_positive and "size" in params or (not is_positive and "size" in cf_params):
            sizes = [obj_size] * cluster_size
        else:
            sizes = [random.uniform(0.5, 1.8) * obj_size for _ in range(cluster_size)]


        shapes = [logics["shape"]] * cluster_size

        if "color" in irrel_params:
            colors = [invariant_color] * cluster_size
        if "size" in irrel_params:
            sizes = [obj_size for _ in range(cluster_size)]

        group_ids = [clu_i] * cluster_size
        start_angles = [settings["angles"][clu_i % len(settings["angles"])]] * cluster_size
        end_angles = [settings["angles"][clu_i % len(settings["angles"])] + 300] * cluster_size
        objs += encode_utils.encode_scene(positions, sizes, colors, shapes, group_ids, is_positive, start_angles, end_angles)
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


def non_overlap_pacman(params, irrel_params, is_positive, clu_num, obj_quantity, pin):
    obj_size = 0.05
    cf_params = data_utils.get_proper_sublist(params)

    objs = similarity_pacman(obj_size, is_positive, clu_num, params, irrel_params, cf_params, obj_quantity)
    t = 0
    tt = 0
    max_try = 1000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = similarity_pacman(obj_size, is_positive, clu_num, params, irrel_params, cf_params, obj_quantity)
        if tt > 10:
            tt = 0
            obj_size = obj_size * 0.90
        tt = tt + 1
        t = t + 1

    logic = get_logics(is_positive, params, cf_params, irrel_params)
    return objs, logic
