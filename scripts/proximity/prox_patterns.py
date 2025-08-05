# Created by jing at 25.02.25
from scripts.utils.encode_utils import create_tasks_v2, create_tasks_v3, create_tasks_v4

from scripts.proximity.util_grid_objs import non_overlap_grid
from scripts.proximity.util_red_triangle import non_overlap_red_triangle
from scripts.proximity.util_weird_circle import overlap_circle_features
from scripts.proximity.util_fixed_props import non_overlap_fixed_props
from scripts.proximity.util_big_small import overlap_big_small, non_overlap_big_small_2
from scripts import config

""" 
p: positive
s: size

"""

size_list = ["s", "m", "l"]
qua_list = ["all", "exist"]


def create_tasks(func, obj_size, task_sizes, *args):
    return {f"{func.__name__}_{'_'.join(map(str, args))}_{s}": (lambda p, s=s, args=args: func(obj_size, p, s, *args))
            for s in task_sizes}

def get_patterns(lite=False):
    # Define task functions dynamically
    if lite:
        size_list = ["s"]
        grp_num_range = range(2,4)
        feature_props = ["color", "size", "shape", "count", "position"]
        qua_list = ["exist"]
    else:
        size_list = config.standard_quantity_dict.keys()
        qua_list = ["all", "exist"]
        grp_num_range = range(2, 5)
        feature_props = ["shape", "color", "size", "count", "position"]
    pin = config.prin_in_neg

    all_tasks = []
    all_names = []

    # tasks, names = create_tasks_v4(non_overlap_red_triangle, feature_props, grp_num_range, size_list, qua_list, pin)
    # all_tasks.extend(tasks)
    # all_names.extend(names)

    tasks, names = create_tasks_v3(non_overlap_grid, feature_props, grp_num_range, size_list, pin)
    all_tasks.extend(tasks)
    all_names.extend(names)
    # tasks.update(create_tasks_v2(non_overlap_fixed_props, ["shape", "color"], size_list, pin))
    # tasks.update(create_tasks_v3(overlap_big_small, ["shape", "color", "count"], num_lst, size_list, pin))
    # tasks.update(create_tasks_v4(overlap_circle_features, ["shape", "color"], num_lst, size_list, [0.8, 1, 1.2], pin))


    # Convert tasks to pattern dictionary
    pattern_dicts = [{"name": key, "module": task} for key, task in zip(all_names, all_tasks)]
    return pattern_dicts
