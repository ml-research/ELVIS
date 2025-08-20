# Created by jing at 26.02.25

import random
import numpy as np

from scripts import config
from scripts.utils.shape_utils import overlaps, overflow
from scripts.utils import pos_utils, encode_utils, data_utils


def generate_grp_position(grid_positions, used_positions, diameter):
    new_position = random.choice(grid_positions)
    while new_position in used_positions:
        new_position = random.choice(grid_positions)

    used_positions.append(new_position)
    # slightly adjust the position
    direction = random.choice([
        (diameter, 0), (-diameter, 0), (0, diameter), (0, -diameter),
        (diameter, diameter), (-diameter, -diameter), (diameter, -diameter), (-diameter, diameter)
    ])

    new_position[0] += direction[0]
    new_position[1] += direction[1]

    #
    # while count < total_obj_num:
    #     # Pick a random cluster center to start a new cluster.
    #     cluster_x, cluster_y = random.choice(cluster_centers)
    #     cluster_points = [(cluster_x, cluster_y)]
    #     used_centers.add((cluster_x, cluster_y))
    #     # Decide how many circles to generate in this cluster.
    #     num_circles = np.random.randint(min_circles, max_circles + 1)
    #
    #     # Create additional circles in the cluster.
    #     for _ in range(num_circles - 1):
    #         if count >= total_obj_num:
    #             break
    #         directions = [
    #             (diameter, 0), (-diameter, 0), (0, diameter), (0, -diameter),
    #             (diameter, diameter), (-diameter, -diameter), (diameter, -diameter), (-diameter, diameter)
    #         ]
    #         random.shuffle(directions)  # Try different directions at random
    #         for dx, dy in directions:
    #             new_x = cluster_points[-1][0] + dx
    #             new_y = cluster_points[-1][1] + dy
    #             # Check that the new circle is within bounds and not overlapping existing ones.
    #             if (0.05 < new_x < image_size[0] and 0.05 < new_y < image_size[1] and
    #                     all((new_x - cx) ** 2 + (new_y - cy) ** 2 >= diameter ** 2 for cx, cy in used_centers)):
    #                 cluster_points.append((new_x, new_y))
    #                 used_centers.add((new_x, new_y))
    #                 position = (new_x, new_y)
    #                 positions.append(position)
    #                 count += 1
    #                 break
    return new_position, used_positions


def generate_positions(grid_size, image_size, diameter, grp_nums):
    """Generate objects for one group (of a given color) until target_count is reached."""
    positions = []
    # Generate evenly spaced cluster centers within the grid
    grid_spacing = image_size[0] / (grid_size + 1)
    grid_positions = [[grid_spacing * (i + 1), grid_spacing * (j + 1)]
                      for i in range(grid_size) for j in range(grid_size)]
    random.shuffle(grid_positions)
    used_positions = []
    for grp_num in grp_nums:
        for _ in range(grp_num):
            # generate positions for this group
            new_position, used_positions = generate_grp_position(grid_positions, used_positions, diameter)
            positions.append(new_position)
    #
    # while count < total_obj_num:
    #     # Pick a random cluster center to start a new cluster.
    #     cluster_x, cluster_y = random.choice(cluster_centers)
    #     cluster_points = [(cluster_x, cluster_y)]
    #     used_centers.add((cluster_x, cluster_y))
    #     # Decide how many circles to generate in this cluster.
    #     num_circles = np.random.randint(min_circles, max_circles + 1)
    #
    #     # Create additional circles in the cluster.
    #     for _ in range(num_circles - 1):
    #         if count >= total_obj_num:
    #             break
    #         directions = [
    #             (diameter, 0), (-diameter, 0), (0, diameter), (0, -diameter),
    #             (diameter, diameter), (-diameter, -diameter), (diameter, -diameter), (-diameter, diameter)
    #         ]
    #         random.shuffle(directions)  # Try different directions at random
    #         for dx, dy in directions:
    #             new_x = cluster_points[-1][0] + dx
    #             new_y = cluster_points[-1][1] + dy
    #             # Check that the new circle is within bounds and not overlapping existing ones.
    #             if (0.05 < new_x < image_size[0] and 0.05 < new_y < image_size[1] and
    #                     all((new_x - cx) ** 2 + (new_y - cy) ** 2 >= diameter ** 2 for cx, cy in used_centers)):
    #                 cluster_points.append((new_x, new_y))
    #                 used_centers.add((new_x, new_y))
    #                 position = (new_x, new_y)
    #                 positions.append(position)
    #                 count += 1
    #                 break
    return positions


def generate_group_objects(color, target_count, cluster_centers, image_size, diameter,
                           so, min_circles, max_circles, used_centers, fixed_props, group_id):
    """Generate objects for one group (of a given color) until target_count is reached."""
    group_objs = []
    count = 0
    shape = random.choice(config.bk_shapes[1:])
    while count < target_count:
        # Pick a random cluster center to start a new cluster.
        cluster_x, cluster_y = random.choice(cluster_centers)
        cluster_points = [(cluster_x, cluster_y)]
        used_centers.add((cluster_x, cluster_y))

        # Decide how many circles to generate in this cluster.
        num_circles = np.random.randint(min_circles, max_circles + 1)

        # Create additional circles in the cluster.
        for _ in range(num_circles - 1):
            if count >= target_count:
                break
            directions = [
                (diameter, 0), (-diameter, 0), (0, diameter), (0, -diameter),
                (diameter, diameter), (-diameter, -diameter), (diameter, -diameter), (-diameter, diameter)
            ]
            random.shuffle(directions)  # Try different directions at random
            for dx, dy in directions:
                new_x = cluster_points[-1][0] + dx
                new_y = cluster_points[-1][1] + dy
                # Check that the new circle is within bounds and not overlapping existing ones.
                if (0.05 < new_x < image_size[0] and 0.05 < new_y < image_size[1] and
                        all((new_x - cx) ** 2 + (new_y - cy) ** 2 >= diameter ** 2 for cx, cy in used_centers)):
                    cluster_points.append((new_x, new_y))
                    used_centers.add((new_x, new_y))

                    if "shape" in fixed_props:
                        shape = random.choice(config.bk_shapes[1:])

                    obj = encode_utils.encode_objs(x=new_x, y=new_y, size=so, color=color, shape=shape,
                                                   line_width=-1, solid=True, group_id=group_id)
                    group_objs.append(obj)
                    count += 1
                    break
    return group_objs


def similarity_fixed_number(is_positive, obj_size, cluster_num, params, irrel_params, cf_params, quantity, min_circles=3,
                            max_circles=5, diameter=0.08, image_size=(1, 1)):
    """
    Generate a scene by creating multiple groups.

    Parameters:
    - is_positive: Determines whether groups are equal-sized.
    - obj_size: Size of the objects.
    - cluster_num: Number of clusters (2, 3, or 4).
    - fixed_props: Fixed properties for objects.
    - quantity: Defines the total object quantity in the image ('s' for small, 'm' for medium, 'l' for large).
    - grid_size: Defines the number of cluster centers (grid_size x grid_size).
    - min_circles, max_circles: Controls the number of circles per cluster.
    - diameter: Spacing used for circle placement.
    - image_size: Tuple defining the dimensions of the image.

    Returns:
    - List of kandinskyShape objects.
    """
    logic = {
        "shape": ["circle", "square", "triangle", "cross", "star"],
        "color": ["yellow", "blue", "red", "green", "purple", "orange", "pink", "brown"],
    }

    def adjust_count(base):
        """Adjust the count of objects if the pattern is negative."""
        return base if is_positive else max(2, base + random.randint(-5, 5))

    # Define object count ranges based on quantity parameter
    base_count = max(2, config.standard_quantity_dict[quantity]//2 + random.randint(-6, -3))
    if is_positive:
        grp_obj_nums = [base_count] * cluster_num
    else:
        # For negative patterns, adjust the count of objects
        base_count = adjust_count(base_count)
        grp_obj_nums = [base_count] * cluster_num

    shapes = []
    sizes = []
    colors = []
    group_ids = []
    for i, base_count in enumerate(grp_obj_nums):
        if is_positive and "shape" in params or (not is_positive and "shape" in cf_params):
            grp_shapes = [logic["shape"][i]] * base_count
        else:
            grp_shapes = [random.choice(config.all_shapes) for _ in range(base_count)]
        shapes += grp_shapes

        if is_positive and "size" in params or (not is_positive and "size" in cf_params):
            grp_sizes = [obj_size] * base_count
        else:
            grp_sizes = [random.uniform(obj_size * 0.5, obj_size * 1.5) for _ in range(base_count)]
        sizes += grp_sizes

        if is_positive and "color" in params or (not is_positive and "color" in cf_params):
            grp_colors = [logic["color"][i]] * base_count
        else:
            grp_colors = [random.choice(config.color_large_exclude_gray)] * base_count
        colors += grp_colors
        group_ids += [i] * base_count

    if "shape" in irrel_params:
        shapes = [random.choice(config.all_shapes)] * sum(grp_obj_nums)
    if "color" in irrel_params:
        colors = [random.choice(config.color_large_exclude_gray)] * sum(grp_obj_nums)
    if "size" in irrel_params:
        sizes = [random.uniform(obj_size * 0.8, obj_size * 1.5)] * sum(grp_obj_nums)

    grid_size = int(np.sqrt(sum(grp_obj_nums))) + 2
    positions = generate_positions(grid_size, image_size, diameter, grp_obj_nums)

    objs = encode_utils.encode_scene(positions, sizes, colors, shapes, group_ids, is_positive)

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


def non_overlap_fixed_number(params, irrel_params, is_positive, cluster_num, quantity, pin):
    obj_size = 0.05
    cf_params = data_utils.get_proper_sublist(params)
    objs = similarity_fixed_number(is_positive, obj_size, cluster_num, params, irrel_params, cf_params, quantity)
    t = 0
    tt = 0
    max_try = 1000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = similarity_fixed_number(is_positive, obj_size, cluster_num, params, irrel_params, cf_params, quantity)
        if tt > 10:
            tt = 0
            obj_size = obj_size * 0.90
        tt = tt + 1
        t = t + 1
    logics = get_logics(is_positive, params, cf_params, irrel_params)
    return objs, logics
