"""
Symmetry pattern task registration for 3D CLEVR.
Mirrors scripts/symmetry/symmetry_patterns.py.
"""

from scripts.utils.encode_utils import create_tasks_v3
from scripts.clevr3d.principles.symmetry_3d import (
    non_overlap_axis_symmetry_3d,
    non_overlap_rotational_symmetry_3d,
)
from scripts.clevr3d import config_3d


def get_patterns(lite=False):
    if lite:
        size_list = ["s"]
        grp_num_range = range(2, 4)
        feature_props = ["color", "shape", "material"]
    else:
        size_list = list(config_3d.standard_quantity_dict.keys())
        grp_num_range = range(2, 4)
        feature_props = ["shape", "color", "size", "material"]

    pin = False

    all_tasks = []
    all_names = []

    # Bilateral symmetry
    tasks, names = create_tasks_v3(
        non_overlap_axis_symmetry_3d, feature_props, grp_num_range, size_list, pin
    )
    all_tasks.extend(tasks)
    all_names.extend(names)

    # Rotational symmetry
    tasks, names = create_tasks_v3(
        non_overlap_rotational_symmetry_3d, feature_props, grp_num_range, size_list, pin
    )
    all_tasks.extend(tasks)
    all_names.extend(names)

    pattern_dicts = [{"name": key, "module": task} for key, task in zip(all_names, all_tasks)]
    return pattern_dicts
