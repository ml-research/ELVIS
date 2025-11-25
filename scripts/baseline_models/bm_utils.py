# Created by MacBook Pro at 25.11.25

# from transformers import AutoProcessor, GPT5ForConditionalGeneration
import torch
import wandb
from pathlib import Path
from PIL import Image
from rtpt import RTPT
import json
from scripts.baseline_models import conversations
from scripts.utils import data_utils
import os
from datetime import datetime
from openai import OpenAI
import base64
from io import BytesIO
import re
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment


def init_wandb(batch_size, principle):
    wandb.init(project=f"GPT5-Gestalt-{principle}", config={"batch_size": batch_size})


def load_patterns(principle_path, start_num, task_num):

    pattern_folders = sorted([p for p in (principle_path / "train").iterdir() if p.is_dir()], key=lambda x: x.stem)
    if not pattern_folders:
        print("No pattern folders found in", principle_path)
        return

    if task_num != "full":
        if task_num != "end":
            task_num = int(task_num)
            pattern_folders = pattern_folders[start_num:start_num + task_num]
        else:
            pattern_folders = pattern_folders[start_num:]

    return pattern_folders


def load_images(image_dir, num_samples=5):
    image_paths = sorted(Path(image_dir).glob("*.png"))[:num_samples]
    return [Image.open(img_path).convert("RGB").resize((224, 224)) for img_path in image_paths]


def load_jsons(json_dir, num_samples=5):
    json_paths = sorted(Path(json_dir).glob("*.json"))[:num_samples]
    json_data = []
    for json_path in json_paths:
        with open(json_path, 'r') as f:
            data = json.load(f)
            json_data.append(data)
    return json_data