# Created by jing at 01.03.25

from scripts.utils.encode_utils import create_tasks_v2, create_tasks_v3

from scripts.closure.util_pos_triangle import separate_big_triangle
from scripts.closure.util_pos_square import separate_big_square
from scripts.closure.util_pos_circle import non_overlap_big_circle
from scripts.closure.util_feature_triangle import non_overlap_feature_triangle
from scripts.closure.util_feature_square import non_overlap_feature_square
from scripts.closure.util_feature_circle import non_overlap_feature_circle
from scripts import config


def get_patterns():
    size_list = ["s", "m", "l", "xl", "xxl", "xxxl"]
    pin = config.prin_in_neg
    # Define task functions dynamically
    all_tasks = []
    all_names = []

    props = [
        "shape",
        "color",
        "position",
        "size",
        "count"
    ]
    # tasks, names = create_tasks_v3(separate_big_triangle, props, range(1, 4), size_list, pin)
    # all_tasks.extend(tasks)
    # all_names.extend(names)

    # tasks, names = create_tasks_v3(separate_big_square, props, range(1, 3), size_list, pin)
    # all_tasks.extend(tasks)
    # all_names.extend(names)

    tasks, names = create_tasks_v3(non_overlap_big_circle, props, range(1, 3), size_list,pin)
    all_tasks.extend(tasks)
    all_names.extend(names)

    # tasks.update(create_tasks_v2(non_overlap_feature_triangle, ["color", "size"], range(1, 5), pin))
    # tasks.update(create_tasks_v2(non_overlap_feature_square, ["color", "size"], range(1, 5), pin))
    # tasks.update(create_tasks_v2(non_overlap_feature_circle, ["color","shape", "size"], range(1, 4), pin))

    # Convert tasks to pattern dictionary
    all_patterns = [{"name": key, "module": task} for key, task in zip(all_names, all_tasks)]
    return all_patterns
