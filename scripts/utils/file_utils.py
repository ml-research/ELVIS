# Created by jing at 25.02.25
import os
import json
from PIL import Image
import numpy as np
import torch
import shutil

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


def hconcat_imgs(img_list):
    padding_imgs = []
    for img in img_list:
        padding_imgs.append(img_padding(img))
    img = np.hstack(padding_imgs).astype(np.uint8)

    return img


def save_img(img_path, data_path, pattern_data, img_data, image):

    # save image
    Image.fromarray(image).save(img_path)

    # save data
    pattern_data["img_data"] = img_data
    with open(data_path, 'w') as f:
        json.dump(pattern_data, f)


def remove_folder(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)  # Removes folder and all its contents
        print(f"Folder '{folder_path}' removed successfully.")
    else:
        print(f"Folder '{folder_path}' does not exist.")

def is_png_file(filename):
    """Check if file is a PNG image"""
    return filename.lower().endswith('.png')



def count_images(folder_path):
    """Count the number of PNG images in a folder."""
    if not os.path.exists(folder_path):
        return 0
    return sum(1 for f in os.listdir(folder_path) if is_png_file(f))