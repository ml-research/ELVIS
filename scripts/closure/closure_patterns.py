# Created by jing at 01.03.25

from scripts.utils.encode_utils import create_tasks_v2, create_tasks_v3

from scripts.closure.util_pos_triangle import non_overlap_big_triangle
from scripts.closure.util_pos_square import non_overlap_big_square
from scripts.closure.util_pos_circle import non_overlap_big_circle
from scripts.closure.util_feature_triangle import non_overlap_feature_triangle
from scripts.closure.util_feature_square import non_overlap_feature_square
from scripts.closure.util_feature_circle import non_overlap_feature_circle
from scripts import config


size_list = ["s", "m", "l"]
pin = config.prin_in_neg
# Define task functions dynamically
tasks = {}

tasks.update(create_tasks_v3(non_overlap_big_triangle, ["shape", "color", "size"], range(1, 3), size_list, pin))
tasks.update(create_tasks_v3(non_overlap_big_square, ["shape", "color", "size"], range(1, 3), size_list, pin))
tasks.update(create_tasks_v3(non_overlap_big_circle, ["shape", "color", "size"], range(1, 3), size_list,pin))
#
tasks.update(create_tasks_v2(non_overlap_feature_triangle, ["color", "size"], range(1, 5), pin))
tasks.update(create_tasks_v2(non_overlap_feature_square, ["color", "size"], range(1, 5), pin))
tasks.update(create_tasks_v2(non_overlap_feature_circle, ["color","shape", "size"], range(1, 4), pin))


# Convert tasks to pattern dictionary
pattern_dicts = [{"name": key, "module": task} for key, task in tasks.items()]
