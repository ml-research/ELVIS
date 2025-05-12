# Created by jing at 25.02.25
import os
from pathlib import Path
import matplotlib

root = Path(__file__).parents[1]

# settings
num_samples = 10
img_width = 1024
prin_in_neg=False

# -------------------- shape settings --------------------
bk_shapes = ["none", "triangle", "square", "circle"]

# ---------------------- color settings ------------------
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
raw_patterns = data / 'raw_patterns'
if not os.path.exists(raw_patterns):
    os.makedirs(raw_patterns)

# -------------- llm path -----------------------
cache_model_path = data / "llm_pretrained"
if not os.path.exists(cache_model_path):
    os.makedirs(cache_model_path)
# -------------- scripts path -----------------------
scripts = root / 'scripts'
if not os.path.exists(scripts):
    os.mkdir(scripts)
# -------------- results path -----------------------
results = root / "data" / "results"
if not os.path.exists(results):
    os.makedirs(results)

# -------------- categories -----------------------
categories = {
    "proximity": ["red_triangle", "grid", "fixed_props", "big_small", "circle_features"],
    "similarity": ["fixed_number", "pacman", "palette"],
    "closure": ["big_triangle", "big_square", "big_circle", "feature_triangle", "feature_square", "feature_circle"],
    "symmetry": ["soloar_sys", "symmetry_circle"],
    "continuity": ["one_split_n", "two_splines", "a_splines", "u_splines", "x_splines"]

}
