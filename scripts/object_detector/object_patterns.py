# Created by MacBook Pro at 20.06.25



from scripts.utils.encode_utils import create_tasks_v2, create_tasks_v3


from scripts.object_detector.util_objects import one_random
from scripts import config

size_list = ["s"]
prop_list = ["shape"]
# Define task functions dynamically

tasks = {}
prin_in_neg = config.prin_in_neg
tasks.update(create_tasks_v3(one_random, prop_list, range(2), size_list, prin_in_neg))

# Convert tasks to pattern dictionary
pattern_dicts = [{"name": key, "module": task} for key, task in tasks.items()]