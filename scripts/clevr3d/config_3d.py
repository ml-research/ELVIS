import os
import math
import numpy as np
from pathlib import Path
from scripts import config

# Blender executable path
BLENDER_EXECUTABLE = os.environ.get('BLENDER_PATH', 'blender')

# Optional: Apptainer/Singularity container for Blender (for headless cluster nodes)
# Set BLENDER_CONTAINER env var to the .sif path to run Blender inside a container.
# When set, Blender is invoked as: apptainer exec --bind /mnt <container> blender ...
BLENDER_CONTAINER = os.environ.get('BLENDER_CONTAINER', '')

# -------------------- CLEVR Colors --------------------
CLEVR_COLORS = {
    "gray":   (87, 87, 87),
    "red":    (173, 35, 35),
    "blue":   (42, 75, 215),
    "green":  (29, 105, 20),
    "brown":  (129, 74, 25),
    "purple": (129, 38, 192),
    "cyan":   (41, 208, 208),
    "yellow": (255, 238, 51),
}

CLEVR_COLOR_NAMES = list(CLEVR_COLORS.keys())

# -------------------- CLEVR Shapes --------------------
CLEVR_SHAPES = ["sphere", "cube", "cylinder"]

# -------------------- CLEVR Sizes --------------------
CLEVR_SIZES = {
    "small": 0.35,
    "large": 0.7,
}

CLEVR_SIZE_NAMES = list(CLEVR_SIZES.keys())

# -------------------- CLEVR Materials --------------------
CLEVR_MATERIALS = {
    "metal": {
        "metallic": 1.0,
        "roughness": 0.2,
        "specular": 0.5,
    },
    "rubber": {
        "metallic": 0.0,
        "roughness": 0.6,
        "specular": 0.1,
    },
}

CLEVR_MATERIAL_NAMES = list(CLEVR_MATERIALS.keys())

# -------------------- Camera --------------------
CAMERA = {
    "location": (7.358891, -6.925790, 4.958309),
    "rotation_euler": (1.1526, 0.0, 0.8127),
    "lens": 35.0,
    "sensor_width": 32.0,
}

# -------------------- Lighting --------------------
LIGHTS = [
    {
        "name": "key_light",
        "type": "AREA",
        "location": (6.0, -2.0, 8.0),
        "rotation_euler": (0.6, 0.1, 0.5),
        "energy": 500.0,
        "size": 3.0,
    },
    {
        "name": "fill_light",
        "type": "AREA",
        "location": (-4.0, -3.0, 6.0),
        "rotation_euler": (0.8, -0.2, -0.6),
        "energy": 200.0,
        "size": 5.0,
    },
    {
        "name": "back_light",
        "type": "AREA",
        "location": (0.0, 6.0, 7.0),
        "rotation_euler": (-0.5, 0.0, 3.14),
        "energy": 300.0,
        "size": 4.0,
    },
]

# -------------------- Ground Plane --------------------
GROUND_PLANE = {
    "size": 20.0,
    "color": (0.8, 0.8, 0.8, 1.0),
}

# -------------------- Scene Bounds --------------------
# Objects are placed within this region on the ground plane.
# Wider bounds to accommodate multiple clusters of CLEVR-sized objects.
SCENE_BOUNDS = {
    "x_min": -5.0,
    "x_max": 5.0,
    "y_min": -5.0,
    "y_max": 5.0,
}

# -------------------- Render Settings --------------------
DEFAULT_RESOLUTION = (480, 480)
DEFAULT_RENDER_ENGINE = "CYCLES"  # "BLENDER_EEVEE" for speed

RENDER_SETTINGS = {
    "CYCLES": {
        "samples": 128,
        "use_denoising": True,
    },
    "BLENDER_EEVEE": {
        "samples": 64,
    },
}

# -------------------- Quantity Settings --------------------
# Reuse from 2D config
standard_quantity_dict = config.standard_quantity_dict

# -------------------- Grouping Parameters --------------------
# 3D cluster radii (in world units).
# Must be large enough to fit objects (radius 0.35-0.7) without overlap.
# Rule of thumb: cluster_radius > sqrt(N) * object_diameter for N objects.
# LAYOUT_RADIUS: worst-case for final validation (large object).
# PACKING_RADIUS: used for spatial layout during generation (mean of small/large).
# Scenes are validated after size assignment with actual radii, so layouts using
# PACKING_RADIUS are rejected if any large-large pair ends up too close.
LAYOUT_RADIUS = CLEVR_SIZES["large"]  # 0.7 — worst-case object radius for final validation
PACKING_RADIUS = 0.5      # radius used for spatial layout packing (avg of small/large)
CLUSTER_RADIUS = {"s": 1.8, "m": 2.0, "l": 2.5, "xl": 3.0}
OVERLAP_MARGIN = 0.4      # extra spacing between objects (matches original CLEVR)
CLUSTER_DIST = 5.0        # default minimum distance between cluster centers

# -------------------- Groupable Properties --------------------
GROUPABLE_PROPERTIES = ["shape", "color", "size", "material"]

# -------------------- Data Paths --------------------
def get_raw_patterns_path_3d(remote=False):
    if remote:
        raw_patterns_path = Path('/gen_data_3d')
    else:
        raw_patterns_path = config.root / 'gen_data_3d'
    os.makedirs(raw_patterns_path, exist_ok=True)
    return raw_patterns_path

# -------------------- Categories --------------------
categories_3d = {
    "proximity": ["fixed_props_3d"],
    "similarity": ["fixed_number_3d"],
    "closure": ["big_triangle_3d", "big_square_3d", "big_circle_3d"],
    "symmetry": ["axis_symmetry_3d", "rotational_symmetry_3d"],
    "continuity": ["two_splines_3d", "non_intersected_splines_3d"],
}
