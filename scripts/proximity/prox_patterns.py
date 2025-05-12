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


# Define task functions dynamically
tasks = {}
num_lst = range(2, 5)
# symbolic features
pin = config.prin_in_neg

# color, all
tasks.update(create_tasks_v4(non_overlap_red_triangle, ["shape", "color"], num_lst, size_list, qua_list, pin))
tasks.update(create_tasks_v3(non_overlap_grid, ["shape", "color"], num_lst, size_list, pin))
tasks.update(create_tasks_v2(non_overlap_fixed_props, ["shape", "color"], size_list, pin))
tasks.update(create_tasks_v3(overlap_big_small, ["shape", "color", "count"], num_lst, size_list, pin))
tasks.update(
    create_tasks_v4(overlap_circle_features, ["shape", "color"], num_lst, size_list, [0.8, 1, 1.2], pin))

# Convert tasks to pattern dictionary
pattern_dicts = [{"name": key, "module": task} for key, task in tasks.items()]
