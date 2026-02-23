"""
Similarity pattern task registration for 3D CLEVR.
Mirrors scripts/similarity/similarity_patterns.py.
"""

from scripts.utils.encode_utils import create_tasks_v3
from scripts.clevr3d.principles.similarity_3d import non_overlap_fixed_number_3d
from scripts.clevr3d import config_3d


def get_patterns(lite=False):
    if lite:
        size_list = ["s"]
        grp_num_range = range(2, 4)
        feature_props = ["color", "size", "shape", "material"]
    else:
        size_list = list(config_3d.standard_quantity_dict.keys())
        grp_num_range = range(2, 5)
        feature_props = ["shape", "color", "size", "material"]

    pin = False

    all_tasks = []
    all_names = []

    tasks, names = create_tasks_v3(
        non_overlap_fixed_number_3d, feature_props, grp_num_range, size_list, pin
    )
    all_tasks.extend(tasks)
    all_names.extend(names)

    pattern_dicts = [{"name": key, "module": task} for key, task in zip(all_names, all_tasks)]
    return pattern_dicts
