# Created by jing at 25.02.25

import sys
import os
import numpy as np
import cv2
import argparse

import math
from rtpt import RTPT
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import matplotlib.transforms as transforms
from matplotlib.path import Path
from PIL import Image
from tqdm import tqdm

from scripts import config
from scripts.utils import file_utils, visual_utils
from scripts.closure import closure_patterns
from scripts.continuity import continuity_patterns
from scripts.proximity import prox_patterns
from scripts.similarity import similarity_patterns
from scripts.symmetry import symmetry_patterns
from scripts.object_detector import object_patterns


# from scripts.mixed_patterns import mixed_patterns

def gen_image_matplotlib(objs):
    import math
    img_size = config.img_width
    dpi = 100
    fig, ax = plt.subplots(figsize=(img_size / dpi, img_size / dpi), dpi=dpi)
    ax.set_xlim(0, img_size)
    ax.set_ylim(0, img_size)
    ax.axis('off')
    ax.set_aspect('equal')

    # Background
    bg_color = config.bg_color
    ax.add_patch(
        patches.Rectangle((0, 0), img_size, img_size, color=bg_color, zorder=0)
    )

    for obj in objs:
        x = obj["x"] * img_size
        y = obj["y"] * img_size
        size = obj["size"] * img_size
        color = (obj["color_r"] / 255, obj["color_g"] / 255, obj["color_b"] / 255)
        shape = obj["shape"]

        patch = None

        if shape == "circle":
            patch = patches.Circle((x, y), size / 2, color=color)
        elif shape == "square":
            patch = patches.Rectangle((x - size / 2, y - size / 2), size, size, color=color)
        elif shape == "triangle":
            half = size / 2
            points = [
                (x, y + half),
                (x - half, y - half),
                (x + half, y - half)
            ]
            patch = patches.Polygon(points, color=color)
        elif shape == "pac_man":
            patch = patches.Wedge((x, y), size / 2, obj.get("start_angle", 30), obj.get("end_angle", 330), color=color)
        elif shape == "pentagon":
            points = [
                (x + size / 2 * math.cos(2 * math.pi * i / 5 - math.pi / 2), y + size / 2 * math.sin(2 * math.pi * i / 5 - math.pi / 2))
                for i in range(5)
            ]
            patch = patches.Polygon(points, color=color)
        elif shape == "hexagon":
            points = [
                (x + size / 2 * math.cos(2 * math.pi * i / 6), y + size / 2 * math.sin(2 * math.pi * i / 6))
                for i in range(6)
            ]
            patch = patches.Polygon(points, color=color)
        elif shape == "star":
            # 5-pointed star
            points = []
            for i in range(10):
                angle = i * math.pi / 5 - math.pi / 2
                r = size / 2 if i % 2 == 0 else size / 4
                points.append((x + r * math.cos(angle), y + r * math.sin(angle)))
            patch = patches.Polygon(points, color=color)
        elif shape == "diamond":
            points = [
                (x, y - size / 2),
                (x - size / 2, y),
                (x, y + size / 2),
                (x + size / 2, y)
            ]
            patch = patches.Polygon(points, color=color)
        elif shape == "ellipse":
            patch = patches.Ellipse((x, y), size, size / 2, color=color)
        elif shape == "cross":
            # Cross shape (X)
            lw = size / 5
            for angle in [45, -45]:
                trans = transforms.Affine2D().rotate_deg_around(x, y, angle) + ax.transData
                rect = patches.Rectangle((x - size / 2, y - lw / 2), size, lw, color=color, transform=trans)
                ax.add_patch(rect)
            continue
        elif shape == "plus":
            # Plus shape (+)
            lw = size / 5
            ax.add_patch(patches.Rectangle((x - lw / 2, y - size / 2), lw, size, color=color))
            ax.add_patch(patches.Rectangle((x - size / 2, y - lw / 2), size, lw, color=color))
            continue
        elif shape == "heart":
            t = np.linspace(0, 2 * np.pi, 100)
            x_ = (size / 30) * 16 * np.sin(t) ** 3
            y_ = (size / 30) * (13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
            points = np.column_stack((x + x_, y + y_))
            patch = patches.Polygon(points, color=color)

        elif shape == "spade":
            # Spade: upright heart top + curved, flared stem + horizontal base
            t = np.linspace(0, 2 * np.pi, 100)
            x_ = (size / 30) * 16 * np.sin(t) ** 3
            y_ = -(size / 30) * (13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
            top = np.column_stack((x + x_, y + y_ + size * 0.12))  # shift up a bit
            patch = patches.Polygon(top, color=color)
            ax.add_patch(patch)

            # Adjusted stem to connect with heart
            stem_h = size / 2.8
            stem_top_w = size / 1000
            stem_bot_w = size / 5
            stem_top_y = y - size * 0.1  # Move up to connect with heart
            stem_bot_y = stem_top_y - stem_h

            # Polygon for curved, flared stem
            stem_points = [
                (x - stem_top_w / 2, stem_top_y),
                (x - stem_bot_w / 2, stem_bot_y),
                (x + stem_bot_w / 2, stem_bot_y),
                (x + stem_top_w / 2, stem_top_y),
            ]
            for i in range(1, 4):
                interp = i / 4
                left_x = x - (stem_top_w / 2) * (1 - interp) - (stem_bot_w / 2) * interp
                right_x = x + (stem_top_w / 2) * (1 - interp) + (stem_bot_w / 2) * interp
                y_interp = stem_top_y * (1 - interp) + stem_bot_y * interp
                stem_points.insert(i, (left_x, y_interp))
                stem_points.insert(-i, (right_x, y_interp))
            stem_patch = patches.Polygon(stem_points, color=color)
            ax.add_patch(stem_patch)

            continue

        elif shape == "club":
            # Club: three circles + flared, curved stem + thick Bezier stems between circles
            r = size / 4.9
            centers = [
                (x, y + r * 1.2),  # top
                (x - r * np.sin(np.pi / 2), y - r * np.cos(np.pi / 6) * 0.8),  # left
                (x + r * np.sin(np.pi / 2), y - r * np.cos(np.pi / 6) * 0.8),  # right
            ]
            for cx, cy in centers:
                ax.add_patch(patches.Circle((cx, cy), r, color=color))

            # Flared, curved stem
            stem_h = size / 2.8
            stem_top_w = size / 1000
            stem_bot_w = size / 4
            stem_top_y = y - size * 0.05
            stem_bot_y = stem_top_y - stem_h * 1.2

            n_interp = 16
            stem_points_left = []
            stem_points_right = []
            for i in range(n_interp + 1):
                t = i / n_interp
                t_pow = t ** 2.2  # Power curve for width
                width = (1 - t_pow) * stem_top_w + t_pow * stem_bot_w
                y_interp = stem_top_y * (1 - t) + stem_bot_y * t
                left_x = x - width / 2
                right_x = x + width / 2
                stem_points_left.append((left_x, y_interp))
                stem_points_right.append((right_x, y_interp))
            stem_points = stem_points_left + stem_points_right[::-1]
            stem_patch = patches.Polygon(stem_points, color=color)
            ax.add_patch(stem_patch)

            #
            # stem_points = [
            #     (x - stem_top_w / 2, stem_top_y),
            #     (x - stem_bot_w / 2, stem_bot_y),
            #     (x + stem_bot_w / 2, stem_bot_y),
            #     (x + stem_top_w / 2, stem_top_y),
            # ]
            # for i in range(1, 4):
            #     interp = i / 4
            #     left_x = x - (stem_top_w / 2) * (1 - interp) - (stem_bot_w / 2) * interp
            #     right_x = x + (stem_top_w / 2) * (1 - interp) + (stem_bot_w / 2) * interp
            #     y_interp = stem_top_y * (1 - interp) + stem_bot_y * interp
            #     stem_points.insert(i, (left_x, y_interp))
            #     stem_points.insert(-i, (right_x, y_interp))
            # stem_patch = patches.Polygon(stem_points, color=color)
            # ax.add_patch(stem_patch)

            # Thick Bezier stems between each pair of circles (center to center)
            pairs = [(0, 1), (1, 2), (2, 0)]
            for i, j in pairs:
                cx1, cy1 = centers[i]
                cx2, cy2 = centers[j]
                start = (cx1, cy1)
                end = (cx2, cy2)
                mx, my = (cx1 + cx2) / 2, (cy1 + cy2) / 2
                nx, ny = -(cy2 - cy1), (cx2 - cx1)
                norm = np.hypot(nx, ny)
                nx, ny = nx / norm, ny / norm
                ctrl = (mx + nx * r * 0.8, my + ny * r * 0.8)
                verts = [start, ctrl, end]
                codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                ax.add_patch(patches.PathPatch(Path(verts, codes), color=color, lw=5, capstyle='round'))
            continue


        else:
            continue

        if patch is not None:
            ax.add_patch(patch)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = plt.imread(buf)
    buf.close()
    img = (img[:, :, :3] * 255).astype(np.uint8)
    img = img[:img_size, :img_size, :]
    return img


def save_patterns(pattern_data, pattern, save_path, num_samples, is_positive):
    imgs = []
    for example_i in range(num_samples):
        img_path = save_path / f"{example_i:05d}.png"
        data_path = save_path / f"{example_i:05d}.json"
        objs, logics = pattern["module"](is_positive)
        # encode symbolic object tensors
        image = visual_utils.gen_image_cv2_full(objs)
        # image = visual_utils.add_label_on_img(image, logics)
        pattern_data["logic_rules"] = logics["rule"]
        file_utils.save_img(img_path, data_path, pattern_data, objs, image)
        imgs.append(image)
    return imgs


def save_task_overview_image(pos_imgs, neg_imgs, save_path, img_size, margin=8, example_num=5):
    # Prepare images
    pos_imgs = pos_imgs[:example_num]
    neg_imgs = neg_imgs[:example_num]
    imgs = pos_imgs + neg_imgs
    imgs = [img[..., ::-1] if img.shape[-1] == 3 else img for img in imgs]
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


def save_principle_patterns(args, principle_name, pattern_dicts):
    resolution_folder = config.get_raw_patterns_path(args.remote) / f"res_{config.img_width}_pin_{config.prin_in_neg}"
    os.makedirs(resolution_folder, exist_ok=True)
    principle_path = resolution_folder / principle_name
    os.makedirs(principle_path, exist_ok=True)
    file_utils.remove_folder(principle_path)
    pattern_counter = 0
    num_samp = config.get_num_samples(args.lite)
    rtpt = RTPT(name_initials='JS', experiment_name=f'Elvis-Gen-{principle_name}', max_iterations=len(pattern_dicts))
    rtpt.start()
    for pattern in tqdm(pattern_dicts):
        rtpt.step()
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
            "num": config.get_num_samples(args.lite),
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
        # print(f"{pattern_counter}/{len(pattern_dicts)} Generating {principle_name} pattern {pattern_name}...")
        train_path = principle_path / "train" / pattern_name
        test_path = principle_path / "test" / pattern_name
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(train_path / "positive", exist_ok=True)
        os.makedirs(train_path / "negative", exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
        os.makedirs(test_path / "positive", exist_ok=True)
        os.makedirs(test_path / "negative", exist_ok=True)
        train_pos_imgs = save_patterns(pattern_data, pattern, train_path / "positive", num_samples=num_samp, is_positive=True)
        train_neg_imgs = save_patterns(pattern_data, pattern, train_path / "negative", num_samples=num_samp, is_positive=False)
        test_pos_imgs = save_patterns(pattern_data, pattern, test_path / "positive", num_samples=num_samp, is_positive=True)
        test_neg_imgs = save_patterns(pattern_data, pattern, test_path / "negative", num_samples=num_samp, is_positive=False)
        # Save overview images
        save_task_overview_image(train_pos_imgs, train_neg_imgs, principle_path / "train" / f"{pattern_name}.png", config.img_width)
        save_task_overview_image(test_pos_imgs, test_neg_imgs, principle_path / "test" / f"{pattern_name}.png", config.img_width)
        pattern_counter += 1
    print(f"{principle_name} pattern generation complete.")


def main(args):
    principles = {
        # "od": object_patterns.pattern_dicts,
        "proximity": prox_patterns.get_patterns(args.lite),
        "similarity": similarity_patterns.get_patterns(args.lite),
        "closure": closure_patterns.get_patterns(args.lite),
        "continuity": continuity_patterns.get_patterns(args.lite),
        "symmetry": symmetry_patterns.get_patterns(args.lite),
        # "mixed":mixed_patterns.pattern_dicts
    }
    if args.principle == "all":
        for principle_name, pattern_dicts in principles.items():
            save_principle_patterns(args, principle_name, pattern_dicts)
    else:
        save_principle_patterns(args, args.principle, principles[args.principle])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline models with CUDA support.")
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--lite", action="store_true")
    parser.add_argument("--principle", type=str)
    parser.add_argument("--labelOn", action="store_true", help="Show labels on the generated images.")
    args = parser.parse_args()

    main(args)
    # draw_club_playground()
