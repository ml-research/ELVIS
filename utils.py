# Created by MacBook Pro at 21.11.25

from rtpt import RTPT
from tqdm import tqdm
import argparse
import json
import wandb
import torch


def init_wandb(model_name, batch_size, epochs, principle, img_num, img_size, num_classes):
    # Initialize Weights & Biases (WandB)
    wandb.init(project=f"elvis-{model_name}-{principle}_{img_num}", config={
        "batch_size": batch_size,
        "image_size": img_size,
        "num_classes": num_classes,
        "epochs": epochs
    })


def load_image_as_tensor(image_path):
    from PIL import Image
    from torchvision import transforms

    img = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    img_tensor = transform(img)
    return img_tensor


def load_boxes_and_groups(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    boxes = []
    group_ids = []
    for item in data['img_data']:
        x = item['x']
        y = item['y']
        w = item['size']
        h = item['size']
        boxes.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
        group_ids.append(item['group_id'])
    ## convert box value to 0-255 range
    boxes = [[b[0] * 255, b[1] * 255, b[2] * 255, b[3] * 255] for b in boxes]
    boxes_tensor = torch.tensor(boxes, dtype=torch.int32)
    group_ids_tensor = torch.tensor(group_ids, dtype=torch.int64)
    return boxes_tensor, group_ids_tensor


def get_pattern_data(pattern_folder, with_neg=False):
    # sample = {
    #     'image': Tensor(3, H, W),
    #     'boxes': Tensor(N, 4),  # (x1,y1,x2,y2)
    #     'group_ids': Tensor(N)
    # },
    samples = []
    # load positive images as tensors from pattern_folder/positive
    positive_folder = pattern_folder / "positive"
    positive_images = sorted([p for p in positive_folder.iterdir() if p.suffix in [".png", ".jpg", ".jpeg"]])
    for img_path in positive_images:
        img = load_image_as_tensor(img_path)
        boxes, group_ids = load_boxes_and_groups(img_path.with_suffix(".json"))
        samples.append({
            'image': img,
            'boxes': boxes,
            'group_ids': group_ids
        })
    return samples
