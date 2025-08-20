# Created by jing at 25.02.25
import os
from pathlib import Path
import matplotlib

root = Path(__file__).parents[1]


# settings

def get_num_samples(lite=False):
    if lite:
        return 5
    else:
        return 10


img_width = 200
prin_in_neg = False

# quantity settings
standard_quantity_dict = {"s": 5,
                          "m": 8,
                          "l": 12,
                          "xl": 15}


def get_grp_r(r, grp_size):
    standard_group_radius_dict = {"s": 0.08, "m": 0.15, "l": 0.2, "xl": 0.23}
    return standard_group_radius_dict[grp_size]


# -------------------- shape settings --------------------
bk_shapes = ["none", "triangle", "square", "circle"]
all_shapes = [
    "triangle",
    "square",
    "circle",
    "pentagon",
    "hexagon",
    "star",
    "cross",
    "plus",
    "diamond",
    "heart",
    "spade",
    "club",
]
# ---------------------- color settings ------------------
bg_color = (211 / 255, 211 / 255, 211 / 255)
color_matplotlib = {k: tuple(int(v[i:i + 2], 16) for i in (1, 3, 5)) for k, v in
                    list(matplotlib.colors.cnames.items())}
color_matplotlib.pop("darkslategray")
color_matplotlib.pop("lightslategray")
color_matplotlib.pop("black")
color_matplotlib.pop("darkgray")

color_dict_rgb2name = {value: key for key, value in color_matplotlib.items()}
color_large = [k for k, v in list(color_matplotlib.items())]
color_large_exclude_gray = [item for item in color_large if item != "lightgray" and item != "lightgrey"]

# -------------- data path -----------------------
data = root / 'data'
if not os.path.exists(data):
    os.makedirs(data)


def get_raw_patterns_path(remote=False):
    if remote:
        raw_patterns_path = Path('/gen_data')
    else:
        raw_patterns_path = root / 'gen_data'

    if not os.path.exists(raw_patterns_path):
        os.makedirs(raw_patterns_path)
    return raw_patterns_path


# -------------- llm path -----------------------
cache_model_path = data / "llm_pretrained"
if not os.path.exists(cache_model_path):
    os.makedirs(cache_model_path)
# -------------- scripts path -----------------------
scripts = root / 'scripts'
if not os.path.exists(scripts):
    os.mkdir(scripts)

result_path = root / 'results'
figure_path = result_path / 'figures'
os.makedirs(figure_path, exist_ok=True)

# -------------- categories -----------------------
categories = {
    "proximity": ["red_triangle", "grid", "fixed_props", "big_small", "circle_features"],
    "similarity": ["fixed_number", "pacman", "palette"],
    "closure": ["big_triangle", "big_square", "big_circle", "feature_triangle", "feature_square", "feature_circle"],
    "symmetry": ["soloar_sys", "symmetry_circle"],
    "continuity": ["one_split_n", "two_splines", "a_splines", "u_splines", "x_splines"]

}
