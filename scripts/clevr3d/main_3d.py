"""
Entry point for 3D CLEVR Gestalt pattern generation.
Mirrors scripts/main.py but uses Blender rendering.

Usage:
    python -m scripts.clevr3d.main_3d --principle proximity --img_size 480
    python -m scripts.clevr3d.main_3d --principle all --img_size 480 --lite
    python -m scripts.clevr3d.main_3d --principle all --img_size 480 --render_engine CYCLES
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from scripts.clevr3d import config_3d
from scripts.clevr3d.blender_renderer import render_scene
from scripts.utils import file_utils

from scripts.clevr3d.prox_patterns_3d import get_patterns as get_proximity_patterns
from scripts.clevr3d.similarity_patterns_3d import get_patterns as get_similarity_patterns
from scripts.clevr3d.closure_patterns_3d import get_patterns as get_closure_patterns
from scripts.clevr3d.continuity_patterns_3d import get_patterns as get_continuity_patterns
from scripts.clevr3d.symmetry_patterns_3d import get_patterns as get_symmetry_patterns


def save_patterns_3d(args, pattern_data, pattern, save_path, num_samples, is_positive):
    """Generate and save 3D patterns via Blender rendering."""
    imgs = []
    for example_i in range(num_samples):
        img_path = save_path / f"{example_i:05d}.png"
        data_path = save_path / f"{example_i:05d}.json"

        objs, logics = pattern["module"](is_positive)

        if objs is None:
            print(f"Warning: Failed to generate pattern for {save_path}, skipping example {example_i}")
            continue

        # Render via Blender
        render_scene(
            objs, str(img_path),
            resolution=(args.img_size, args.img_size),
            render_engine=args.render_engine
        )

        # Load rendered image for overview grid
        if img_path.exists():
            image = np.array(Image.open(img_path).convert('RGB'))
        else:
            image = np.zeros((args.img_size, args.img_size, 3), dtype=np.uint8)

        pattern_data["logic_rules"] = logics["rule"]
        file_utils.save_img(img_path, data_path, pattern_data, objs, image)
        imgs.append(image)

    return imgs


def save_task_overview_image(pos_imgs, neg_imgs, save_path, img_size, margin=8, example_num=5):
    """Save a grid overview of positive and negative examples."""
    pos_imgs = pos_imgs[:example_num]
    neg_imgs = neg_imgs[:example_num]
    imgs = pos_imgs + neg_imgs
    imgs = [Image.fromarray(img).resize((img_size, img_size)) for img in imgs]

    grid_rows = 2
    grid_cols = example_num
    total_width = img_size * grid_cols + margin * (grid_cols + 1)
    total_height = img_size * grid_rows + margin * (grid_rows + 1)
    overview_img = Image.new("RGB", (total_width, total_height), (255, 255, 255))

    for idx, img in enumerate(imgs):
        row = idx // example_num
        col = idx % example_num
        x = margin + col * (img_size + margin)
        y = margin + row * (img_size + margin)
        overview_img.paste(img, (x, y))

    overview_img.save(save_path)


def save_principle_patterns_3d(args, principle_name, pattern_dicts):
    """Generate all patterns for a single Gestalt principle."""
    resolution_folder = config_3d.get_raw_patterns_path_3d(args.remote) / f"res_{args.img_size}_pin_False"
    os.makedirs(resolution_folder, exist_ok=True)
    principle_path = resolution_folder / principle_name
    os.makedirs(principle_path, exist_ok=True)

    pattern_counter = 0
    num_samp = 3 if args.lite else 3

    for pattern in tqdm(pattern_dicts, desc=f"Generating 3D {principle_name}"):
        pattern_name = f"{pattern_counter:03d}_" + pattern["name"]

        # Parse pattern properties from name
        group_num = pattern["name"].split("_")[-1]
        pattern_data = {
            "principle": principle_name,
            "id": pattern_counter,
            "num": num_samp,
            "group_num": group_num,
            "qualifier_all": "all" in pattern_name,
            "qualifier_exist": "exist" in pattern_name,
            "prop_shape": "shape" in pattern_name,
            "prop_color": "color" in pattern_name,
            "non_overlap": True,
            "prop_size": "size" in pattern_name,
            "prop_count": "count" in pattern_name,
            "prop_material": "material" in pattern_name,
            "resolution": args.img_size,
            "render_backend": "blender",
        }

        train_path = principle_path / "train" / pattern_name
        test_path = principle_path / "test" / pattern_name
        for p in [train_path / "positive", train_path / "negative",
                   test_path / "positive", test_path / "negative"]:
            os.makedirs(p, exist_ok=True)

        train_pos_imgs = save_patterns_3d(args, pattern_data, pattern,
                                           train_path / "positive", num_samp, True)
        train_neg_imgs = save_patterns_3d(args, pattern_data, pattern,
                                           train_path / "negative", num_samp, False)
        test_pos_imgs = save_patterns_3d(args, pattern_data, pattern,
                                          test_path / "positive", num_samp, True)
        test_neg_imgs = save_patterns_3d(args, pattern_data, pattern,
                                          test_path / "negative", num_samp, False)

        if train_pos_imgs and train_neg_imgs:
            save_task_overview_image(train_pos_imgs, train_neg_imgs,
                                     principle_path / "train" / f"{pattern_name}.png", args.img_size)
        if test_pos_imgs and test_neg_imgs:
            save_task_overview_image(test_pos_imgs, test_neg_imgs,
                                     principle_path / "test" / f"{pattern_name}.png", args.img_size)

        pattern_counter += 1

    print(f"3D {principle_name} pattern generation complete.")


def main(args):
    principles = {
        "proximity": get_proximity_patterns(args.lite),
        "similarity": get_similarity_patterns(args.lite),
        "closure": get_closure_patterns(args.lite),
        "continuity": get_continuity_patterns(args.lite),
        "symmetry": get_symmetry_patterns(args.lite),
    }

    if args.principle == "all":
        for principle_name, pattern_dicts in principles.items():
            save_principle_patterns_3d(args, principle_name, pattern_dicts)
    else:
        if args.principle not in principles:
            print(f"Error: Unknown principle '{args.principle}'. "
                  f"Choose from: {list(principles.keys())} or 'all'")
            sys.exit(1)
        save_principle_patterns_3d(args, args.principle, principles[args.principle])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3D CLEVR Gestalt patterns.")
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--lite", action="store_true")
    parser.add_argument("--principle", type=str, default="all",
                        choices=["all", "proximity", "similarity", "closure",
                                 "continuity", "symmetry"])
    parser.add_argument("--img_size", type=int, default=480,
                        choices=[224, 480, 512, 1024])
    parser.add_argument("--render_engine", type=str, default="BLENDER_EEVEE",
                        choices=["BLENDER_EEVEE", "CYCLES"])
    args = parser.parse_args()

    main(args)
