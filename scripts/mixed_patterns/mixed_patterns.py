# Created by MacBook Pro at 21.07.25

# Created by jing at 25.02.25
from scripts.utils.encode_utils import create_tasks_v2, create_tasks_v3, create_tasks_v4, create_mixed_tasks_v4
from scripts.mixed_patterns.util_mixed import all_combinations_of_wrapped_groups
from scripts import config
""" 
p: positive
s: size

"""
if __name__ == "__main__":
    size_list = ["s", "m", "l"]
    qua_list = ["all", "exist"]
    # Define task functions dynamically
    tasks = {}
    num_lst = range(2, 3)
    # symbolic features
    pin = config.prin_in_neg
    # color, all
    tasks.update(create_mixed_tasks_v4(all_combinations_of_wrapped_groups, ["shape", "color", "size", "count"], num_lst, size_list, qua_list, pin))


    # Convert tasks to pattern dictionary
    pattern_dicts = [{"name": key, "module": task} for key, task in tasks.items()]