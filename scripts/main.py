# Created by jing at 25.02.25

import sys
import os
import numpy as np
import cv2

from scripts import config
from scripts.utils import file_utils
from scripts.proximity import prox_patterns
from scripts.similarity import similarity_patterns
from scripts.symmetry import symmetry_patterns
from scripts.continuity import continuity_patterns
from scripts.closure import closure_patterns


def gen_image(objs):
    """
    Generate an image from a list of objects.
    :param objects: List of objects, each defined by a dict with keys x, y, size, rgb_color, shape.
    :param img_size: Size of the output image (img_size x img_size).
    :return: Generated image as a NumPy array.
    """
    img_size = config.img_width
    image = np.zeros((img_size, img_size, 3), dtype=np.uint8) + config.color_matplotlib["lightgray"]
    image = image.astype(np.uint8)
    for obj in objs:
        x = int(obj["x"] * img_size)
        y = int(obj["y"] * img_size)
        size = int(obj["size"] * img_size)
        color = (obj["color_r"], obj["color_g"], obj["color_b"])

        if obj["shape"] == "circle":
            cv2.circle(image, (x, y), size // 2, color, -1)
        elif obj["shape"] == "square":
            top_left = (x - size // 2, y - size // 2)
            bottom_right = (x + size // 2, y + size // 2)
            cv2.rectangle(image, top_left, bottom_right, color, -1)
        elif obj["shape"] == "pac_man":
            cv2.ellipse(image, (x, y), (size // 2, size // 2), 0,
                        obj["start_angle"], obj["end_angle"], color, -1)

        elif obj["shape"] == "triangle":
            half_size = size // 2
            points = np.array([
                [x, y - half_size],
                [x - half_size, y + half_size],
                [x + half_size, y + half_size]
            ])
            cv2.fillPoly(image, [points], color)
    # visual_utils.van(image, "test.png")
    return image


def save_patterns(pattern_data, pattern, save_path, num_samples, is_positive):
    for example_i in range(num_samples):
        img_path = save_path / f"{example_i:05d}.png"
        data_path = save_path / f"{example_i:05d}.json"
        objs = pattern["module"](is_positive)
        # encode symbolic object tensors
        image = gen_image(objs)
        file_utils.save_img(img_path, data_path, pattern_data, objs, image)


def save_principle_patterns(principle_name, pattern_dicts):
    resolution_folder = config.raw_patterns / f"res_{config.img_width}_pin_{config.prin_in_neg}"
    os.makedirs(resolution_folder, exist_ok=True)
    principle_path = resolution_folder / principle_name
    os.makedirs(principle_path, exist_ok=True)

    file_utils.remove_folder(principle_path)

    pattern_counter = 0
    num_samp = config.num_samples
    for pattern in pattern_dicts:
        pattern_counter += 1
        pattern_name = f"{pattern_counter:03d}_" + pattern["name"]
        # Run the save_patterns function if it exists in the script

        group_num = pattern["name"].split("_")[-1]
        qualifier_all = True if "all" in pattern_name else False
        qualifier_exist = True if "exist" in pattern_name else False
        prop_shape = True if "shape" in pattern_name else False
        prop_color = True if "color" in pattern_name else False
        prop_count = True if "count" in pattern_name else False
        prop_size = True if "size" in pattern_name else False
        non_overlap = True if "non_overlap" in pattern_name else False

        pattern_data = {
            "principle": principle_name,
            "id": pattern_counter,
            "num": config.num_samples,
            "group_num": group_num,
            "qualifier_all": qualifier_all,
            "qualifier_exist": qualifier_exist,
            "prop_shape": prop_shape,
            "prop_color": prop_color,
            "non_overlap": non_overlap,
            "prop_size": prop_size,
            "prop_count": prop_count,
            "resolution": config.img_width
        }

        print(f"{pattern_counter}/{len(pattern_dicts)} Generating {principle_name} pattern {pattern_name}...")
        train_path = principle_path / "train" / pattern_name
        test_path = principle_path / "test" / pattern_name

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(train_path / "positive", exist_ok=True)
        os.makedirs(train_path / "negative", exist_ok=True)

        os.makedirs(test_path, exist_ok=True)
        os.makedirs(test_path / "positive", exist_ok=True)
        os.makedirs(test_path / "negative", exist_ok=True)

        save_patterns(pattern_data, pattern, train_path / "positive", num_samples=num_samp, is_positive=True)
        save_patterns(pattern_data, pattern, train_path / "negative", num_samples=num_samp, is_positive=False)
        save_patterns(pattern_data, pattern, test_path / "positive", num_samples=num_samp, is_positive=True)
        save_patterns(pattern_data, pattern, test_path / "negative", num_samples=num_samp, is_positive=False)

    print(f"{principle_name} pattern generation complete.")


def main():
    principles = {
        "proximity": prox_patterns.pattern_dicts,
        "similarity": similarity_patterns.pattern_dicts,
        "symmetry": symmetry_patterns.pattern_dicts,
        "continuity": continuity_patterns.pattern_dicts,
        "closure": closure_patterns.pattern_dicts,
    }
    for principle_name, pattern_dicts in principles.items():
        save_principle_patterns(principle_name, pattern_dicts)


if __name__ == "__main__":
    main()
