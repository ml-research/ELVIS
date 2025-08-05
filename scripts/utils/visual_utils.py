# Created by jing at 25.02.25

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import math
from scripts import config


def img_padding(img, pad_width=2):
    if img.ndim == 3:
        pad_img = np.pad(img, pad_width=(
            (pad_width, pad_width), (pad_width, pad_width), (0, 0)),
                         constant_values=255)
    elif img.ndim == 2:
        pad_img = np.pad(img, pad_width=(
            (pad_width, pad_width), (pad_width, pad_width)),
                         constant_values=255)

    else:
        raise ValueError()
    return pad_img


def visual_np_array(array, filename=None):
    if filename is not None:
        # save the image
        # Convert array to image
        image = Image.fromarray(array)
        # Save as PNG
        image.save(filename)
    plt.axis('off')


def hconcat_imgs(img_list):
    padding_imgs = []
    for img in img_list:
        padding_imgs.append(img_padding(img))
    img = np.hstack(padding_imgs).astype(np.uint8)

    return img


def van(array, file_name=None):
    plt.clf()  # Clear current figure
    if isinstance(array, list):
        hconcat = hconcat_imgs(array)
        visual_np_array(hconcat.squeeze(), file_name)
    elif len(array.shape) == 2:
        visual_np_array(array.squeeze(), file_name)
    elif len(array.shape) == 3:
        visual_np_array(array.squeeze(), file_name)
    elif len(array.shape) == 4:
        visual_np_array(array[0].squeeze(), file_name)


def add_label_on_img(image, pattern_data):
    is_positive = pattern_data.get("is_positive", True)
    fixed_props = pattern_data.get("fixed_props", [])
    irrel_props = pattern_data.get("irrel_params", [])
    cf_params = pattern_data.get("cf_params", [])

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    black = (0, 0, 0)
    red = (0, 0, 255)
    bg_color = (255, 255, 255)
    gray = (128, 128, 128)
    margin = 10
    line_spacing = 10

    labels = ([f"{str(is_positive)}", f"{pattern_data['principle']}"] +
              [f"{str(prop)}" for prop in fixed_props] + [f"{str(p)}" for p in irrel_props])
    text_sizes = [cv2.getTextSize("(IR)" +label, font, font_scale, thickness)[0] for label in labels]
    max_width = max(w for w, h in text_sizes)
    total_height = sum(h for w, h in text_sizes) + line_spacing * (len(labels) - 1)

    cv2.rectangle(image, (margin - 5, margin - 5),
                  (margin + max_width + 5, margin + total_height + 5),
                  bg_color, -1)

    y = margin
    for i, label in enumerate(labels):
        color = black
        prefix = "(R)"
        if not is_positive and label not in cf_params + ["False"]:
            color = red
            prefix= "(R)"
        if (label not in fixed_props and label in irrel_props) or (label in irrel_props and label not in cf_params):
            color = gray
            prefix= "(IR)"
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.putText(image, prefix+label, (margin, y + text_h), font, font_scale, color, thickness, cv2.LINE_AA)
        y += text_h + line_spacing

    return image


def gen_image_cv2_full(objs):
    img_size = config.img_width
    image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    bg_color = tuple(int(255 * c) if isinstance(c, float) else int(c) for c in config.bg_color[:3])
    image[:] = bg_color

    for obj in objs:
        x = int(obj["x"] * img_size)
        y = int(obj["y"] * img_size)
        size = int(obj["size"] * img_size)
        color = (int(obj["color_b"]), int(obj["color_g"]), int(obj["color_r"]))  # cv2 uses BGR

        shape = obj["shape"]

        if shape == "circle":
            cv2.circle(image, (x, y), size // 2, color, -1)
        elif shape == "square":
            top_left = (x - size // 2, y - size // 2)
            bottom_right = (x + size // 2, y + size // 2)
            cv2.rectangle(image, top_left, bottom_right, color, -1)
        elif shape == "triangle":
            half = size // 2
            points = np.array([
                [x, y - half],
                [x - half, y + half],
                [x + half, y + half]
            ], np.int32)
            cv2.fillPoly(image, [points], color)
        elif shape == "pac_man":
            start_angle = int(obj.get("start_angle", 30))
            end_angle = int(obj.get("end_angle", 330))
            cv2.ellipse(image, (x, y), (size // 2, size // 2), 0, start_angle, end_angle, color, -1)
        elif shape == "pentagon":
            points = []
            for i in range(5):
                angle = 2 * math.pi * i / 5 - math.pi / 2
                px = int(x + size / 2 * math.cos(angle))
                py = int(y + size / 2 * math.sin(angle))
                points.append([px, py])
            cv2.fillPoly(image, [np.array(points, np.int32)], color)
        elif shape == "hexagon":
            points = []
            for i in range(6):
                angle = 2 * math.pi * i / 6
                px = int(x + size / 2 * math.cos(angle))
                py = int(y + size / 2 * math.sin(angle))
                points.append([px, py])
            cv2.fillPoly(image, [np.array(points, np.int32)], color)
        elif shape == "star":
            points = []
            for i in range(10):
                angle = i * math.pi / 5 - math.pi / 2
                r = size / 2 if i % 2 == 0 else size / 4
                px = int(x + r * math.cos(angle))
                py = int(y + r * math.sin(angle))
                points.append([px, py])
            cv2.fillPoly(image, [np.array(points, np.int32)], color)
        elif shape == "diamond":
            points = np.array([
                [x, y - size // 2],
                [x - size // 2, y],
                [x, y + size // 2],
                [x + size // 2, y]
            ], np.int32)
            cv2.fillPoly(image, [points], color)
        elif shape == "ellipse":
            cv2.ellipse(image, (x, y), (size // 2, size // 4), 0, 0, 360, color, -1)
        elif shape == "cross":
            lw = size // 5
            # Draw two rectangles rotated by 45 and -45 degrees
            M1 = cv2.getRotationMatrix2D((x, y), 45, 1.0)
            M2 = cv2.getRotationMatrix2D((x, y), -45, 1.0)
            rect = np.array([
                [x - size // 2, y - lw // 2],
                [x + size // 2, y - lw // 2],
                [x + size // 2, y + lw // 2],
                [x - size // 2, y + lw // 2]
            ], np.int32)
            rect1 = cv2.transform(np.array([rect]), M1)[0]
            rect2 = cv2.transform(np.array([rect]), M2)[0]
            cv2.fillPoly(image, [rect1], color)
            cv2.fillPoly(image, [rect2], color)
        elif shape == "plus":
            lw = size // 5
            cv2.rectangle(image, (x - lw // 2, y - size // 2), (x + lw // 2, y + size // 2), color, -1)
            cv2.rectangle(image, (x - size // 2, y - lw // 2), (x + size // 2, y + lw // 2), color, -1)
        elif shape == "heart":
            # Approximate heart with polygon
            t = np.linspace(0, 2 * np.pi, 100)
            x_ = (size / 30) * 16 * np.sin(t) ** 3
            y_ = (size / 30) * (13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
            points = np.column_stack((x + x_, y + y_)).astype(np.int32)
            cv2.fillPoly(image, [points], color)
        elif shape == "spade":
            # Upright spade: heart top + downward stem
            t = np.linspace(0, 2 * np.pi, 100)
            x_ = (size / 30) * 16 * np.sin(t) ** 3
            y_ = (size / 30) * (13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
            points = np.column_stack((x + x_, y + y_ - size * 0.12)).astype(np.int32)
            cv2.fillPoly(image, [points], color)

            # Stem (downward, flared)
            stem_h = size / 2.8
            stem_top_w = size / 1000
            stem_bot_w = size / 5
            stem_top_y = y + size * 0.1
            stem_bot_y = stem_top_y + stem_h
            stem_points = np.array([
                [x - stem_top_w / 2, stem_top_y],
                [x - stem_bot_w / 2, stem_bot_y],
                [x + stem_bot_w / 2, stem_bot_y],
                [x + stem_top_w / 2, stem_top_y]
            ], np.int32)
            cv2.fillPoly(image, [stem_points], color)
        elif shape == "club":
            r = size / 4.9
            centers = [
                (x, y - r * 1.0),  # top
                (x - r * np.sin(np.pi / 2), y + r * np.cos(np.pi / 6) * 0.6),  # left
                (x + r * np.sin(np.pi / 2), y + r * np.cos(np.pi / 6) * 0.6),  # right
            ]
            for cx, cy in centers:
                cv2.circle(image, (int(cx), int(cy)), int(r), color, -1)
            # Stem (downward)
            stem_h = size / 2.8
            stem_top_w = size / 1000
            stem_bot_w = size / 4
            stem_top_y = y + size * 0.05
            stem_bot_y = stem_top_y + stem_h * 1.2
            stem_points = np.array([
                [x - stem_top_w / 2, stem_top_y],
                [x - stem_bot_w / 2, stem_bot_y],
                [x + stem_bot_w / 2, stem_bot_y],
                [x + stem_top_w / 2, stem_top_y]
            ], np.int32)
            cv2.fillPoly(image, [stem_points], color)
        # else: skip unknown shapes

    return image
