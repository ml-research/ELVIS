# Created by jing at 25.02.25

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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
