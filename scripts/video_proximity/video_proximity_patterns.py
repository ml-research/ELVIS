# Created by MacBook Pro at 14.07.25
from scripts.utils.encode_utils import create_tasks_v2, create_tasks_v3
from scripts.video_proximity.util_scatter_cluster import non_overlap_scatter_cluster
from scripts import config

size_list = ["s", "m", "l"]
pin = config.prin_in_neg
# Define task functions dynamically
tasks = {}
tasks.update(create_tasks_v3(non_overlap_scatter_cluster, ["shape", "color", "size"], range(1, 3), size_list, pin))
# Convert tasks to pattern dictionary
pattern_dicts = [{"name": key, "module": task} for key, task in tasks.items()]
