# Created by MacBook Pro at 25.11.25
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset
import random
import pickle


def rgb2patch(rgb_img, input_type):
    patch_set, positions, sizes = preprocess_rgb_image_to_patch_set_torch(rgb_img)
    positions = positions[0]
    sizes = sizes[0]
    if input_type == "pos":
        patch_set = patch_set[0][:, :, :2]
    elif input_type == "pos_color":
        patch_set = patch_set[0][:, :, :5]
    elif input_type == "pos_color_size":
        patch_set = patch_set[0]
    elif input_type == "color_size":
        patch_set = patch_set[0][:, :, 2:]
    else:
        raise ValueError

    return patch_set, positions, sizes

class ContextContourDataset(Dataset):
    def __init__(self, root_dir, orders, input_type, sample_size, device, task_num=20, data_num=100000, split="train", test_ratio=0.2, remove_cache=False):
        self.root_dir = Path(root_dir)
        self.data = []
        self.input_type = input_type
        self.data_num = data_num
        self.task_num = task_num
        # Cache file path
        cache_name = f"cache_{input_type}_ss{sample_size}_tn{task_num}_dn{data_num}.pkl"
        cache_path = self.root_dir / cache_name

        if remove_cache and cache_path.exists():
            print(f"Removing existing cache at {cache_path}")
            cache_path.unlink()

        if cache_path.exists():
            print(f"Loading cached dataset from {cache_path}")
            with open(cache_path, "rb") as f:
                self.data = pickle.load(f)
        else:
            print("No cache found, processing dataset...")
            self.data = []
            self._load(sample_size, orders, device)
            self.balance_labels()
            with open(cache_path, "wb") as f:
                pickle.dump(self.data, f)
            print(f"Dataset cached to {cache_path}")


        split_idx = int(len(self.data) * (1 - test_ratio))
        if split == "train":
            self.data = self.data[:split_idx]
        else:
            self.data = self.data[split_idx:]

    def _get_bbox_corners(self, x, y, size):
        half_size = size / 2
        # Return 4 corners: top-left, top-right, bottom-right, bottom-left
        return [
            [x - half_size, y - half_size],
            [x + half_size, y - half_size],
            [x + half_size, y + half_size],
            [x - half_size, y + half_size],
        ]

    def _get_task_dirs(self, orders):
        task_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        task_dirs = [task_dirs[i] for i in orders]
        task_dirs = task_dirs[:self.task_num] if self.task_num < len(task_dirs) else task_dirs

        return task_dirs

    def _process_label_dir(self, labeled_dir, device, file_sample_size=3, sample_size=10):
        json_files = sorted(labeled_dir.glob("*.json"))
        png_files = sorted(labeled_dir.glob("*.png"))
        file_indices = list(range(len(json_files)))
        if len(file_indices) > file_sample_size:
            file_indices = random.sample(file_indices, file_sample_size)

        for f_i in file_indices:
            with open(json_files[f_i]) as f:
                metadata = json.load(f)
            objects = metadata.get("img_data", [])
            obj_imgs = patch_preprocess.img_path2obj_images(png_files[f_i], device)
            if len(objects) != len(obj_imgs) or len(objects) < 2:
                continue
            objects, obj_imgs, permutes = patch_preprocess.align_data_and_imgs(objects, obj_imgs)
            self._process_object_pairs(objects, obj_imgs, sample_size, device)

    def _process_object_pairs(self, objects, obj_imgs, sample_size, device="cpu", ):
        patches = [patch_preprocess.rgb2patch(img, self.input_type)[0].to(device) for img in obj_imgs]
        pair_indices = [(i, j) for i in range(len(objects)) for j in range(len(objects)) if i != j]
        # if len(pair_indices) > sample_size:
        #     pair_indices = random.sample(pair_indices, sample_size)
        for i, j in pair_indices:
            obj_i = objects[i]
            obj_j = objects[j]
            c_i = patches[i]
            c_j = patches[j]
            others = [patches[k] for k in range(len(objects)) if k != i and k != j]
            if others:
                others_tensor = torch.stack(others, dim=0).to(device)
            else:
                others_tensor = torch.zeros((1, 6, 16, c_i.shape[-1]), device=device)
            label = 1 if obj_i["group_id"] == obj_j["group_id"] and obj_i["group_id"] != -1 else 0
            self.data.append((c_i, c_j, others_tensor, label))

    def _load(self, sample_size, orders, device="cpu"):
        task_dirs = self._get_task_dirs(orders)
        total_tasks = 0
        for task_dir in task_dirs:
            if len(self.data) > self.data_num:
                break
            print(f"Processing task directory: {task_dir}, current data size: {len(self.data)}")
            total_tasks+=1
            for label_dir in ["positive", "negative"]:
                labeled_dir = task_dir / label_dir
                if not labeled_dir.exists():
                    continue
                self._process_label_dir(labeled_dir, device, sample_size=sample_size)
        print(f"Total tasks processed: {total_tasks}, total data collected: {len(self.data)}")
    def balance_labels(self):
        # Separate data by label
        positives = [d for d in self.data if d[3] == 1]
        negatives = [d for d in self.data if d[3] == 0]
        self.data = positives + negatives
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        c_i, c_j, others, label = self.data[idx]
        return c_i, c_j, others, torch.tensor(label, dtype=torch.float32)

