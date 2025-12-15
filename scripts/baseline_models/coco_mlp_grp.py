# Created by MacBook Pro at 28.11.25

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
import math
from PIL import Image, ImageDraw, ImageFont
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim


from scripts import config


def extract_box_feat(ann, img_w, img_h):
    """
    简单 box 几何特征:
    - 归一化中心
    - 归一化宽高
    - log area, aspect ratio
    """
    x, y, w, h = ann["bbox"]
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    w_n = w / img_w
    h_n = h / img_h
    area = (w * h) / (img_w * img_h + 1e-8)
    log_area = torch.log(torch.tensor(area + 1e-8, dtype=torch.float32))
    aspect = w_n / (h_n + 1e-8)
    return torch.tensor([cx, cy, w_n, h_n, log_area, aspect], dtype=torch.float32)



class CocoTaskPairDataset(Dataset):
    """
    Pair-level 数据集:
    每个样本 = (pair_feat, label)
    - label=1: 两个 object 属于同一个人工 task group
    - label=0: 跨 group 或 task vs background
    """

    def __init__(
        self,
        coco_ann_file: Path,
        task_group_file: Path,
        image_ids_subset=None,
        max_neg_per_pos: int = 3,
        seed: int = 0,
    ):
        super().__init__()
        self.coco = COCO(str(coco_ann_file))

        with open(task_group_file, "r") as f:
            tg = json.load(f)

        img_to_task_groups = defaultdict(list)
        for g in tg["task_groups"]:
            img_to_task_groups[g["image_id"]].append(g["member_ann_ids"])

        if image_ids_subset is None:
            self.image_ids = sorted(img_to_task_groups.keys())
        else:
            self.image_ids = [i for i in image_ids_subset if i in img_to_task_groups]

        self.pairs = []  # list of (feat_i, feat_j, label)
        rng = random.Random(seed)

        for img_id in self.image_ids:
            task_groups = img_to_task_groups[img_id]
            if len(task_groups) == 0:
                continue

            img_info = self.coco.loadImgs(img_id)[0]
            img_w, img_h = img_info["width"], img_info["height"]

            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            anns = self.coco.loadAnns(ann_ids)
            if len(anns) < 2:
                continue

            # ann_id -> feat, group_id (or None)
            ann_id_to_feat = {}
            ann_id_to_group = {}  # -1 = background
            for a in anns:
                feat = extract_box_feat(a, img_w, img_h)
                ann_id_to_feat[a["id"]] = feat
                ann_id_to_group[a["id"]] = -1  # default bg

            for gid, group_ann_ids in enumerate(task_groups):
                for aid in group_ann_ids:
                    if aid in ann_id_to_group:
                        ann_id_to_group[aid] = gid

            # 按 group 拆开
            group_to_members = defaultdict(list)
            bg_ids = []
            for aid in ann_id_to_feat.keys():
                g = ann_id_to_group[aid]
                if g == -1:
                    bg_ids.append(aid)
                else:
                    group_to_members[g].append(aid)

            # === Positive pairs: 同一 task group 内的所有 pairs ===
            pos_pairs = []
            for gid, members in group_to_members.items():
                if len(members) < 2:
                    continue
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        ai, aj = members[i], members[j]
                        fi = ann_id_to_feat[ai]
                        fj = ann_id_to_feat[aj]
                        pair_feat = torch.cat([fi, fj, torch.abs(fi - fj)], dim=0)
                        pos_pairs.append((pair_feat, 1.0))

            # === Negative pairs: 跨 group + task vs bg ===
            neg_candidates = []

            # (1) task vs bg
            task_ids = [aid for aid, g in ann_id_to_group.items() if g != -1]
            for aid in task_ids:
                for bid in bg_ids:
                    fi = ann_id_to_feat[aid]
                    fj = ann_id_to_feat[bid]
                    pair_feat = torch.cat([fi, fj, torch.abs(fi - fj)], dim=0)
                    neg_candidates.append((pair_feat, 0.0))

            # (2) task vs task (不同 group)
            group_ids = [gid for gid in group_to_members.keys()]
            for i in range(len(group_ids)):
                for j in range(i + 1, len(group_ids)):
                    gi, gj = group_ids[i], group_ids[j]
                    for ai in group_to_members[gi]:
                        for aj in group_to_members[gj]:
                            fi = ann_id_to_feat[ai]
                            fj = ann_id_to_feat[aj]
                            pair_feat = torch.cat([fi, fj, torch.abs(fi - fj)], dim=0)
                            neg_candidates.append((pair_feat, 0.0))

            # 下采样 negative，避免极度不平衡
            max_neg = max_neg_per_pos * max(1, len(pos_pairs))
            if len(neg_candidates) > max_neg:
                neg_pairs = rng.sample(neg_candidates, max_neg)
            else:
                neg_pairs = neg_candidates

            self.pairs.extend(pos_pairs)
            self.pairs.extend(neg_pairs)

        print(f"[CocoTaskPairDataset] images={len(self.image_ids)}, pairs={len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        feat, label = self.pairs[idx]
        return feat, torch.tensor([label], dtype=torch.float32)


def build_coco_pair_datasets(
    coco_ann_file: Path,
    task_group_file: Path,
    train_ratio: float = 0.8,
    seed: int = 0,
):
    with open(task_group_file, "r") as f:
        tg = json.load(f)

    img_ids = sorted({g["image_id"] for g in tg["task_groups"]})
    rng = random.Random(seed)
    rng.shuffle(img_ids)

    split = int(len(img_ids) * train_ratio)
    train_ids = img_ids[:split]
    val_ids = img_ids[split:]

    train_dataset = CocoTaskPairDataset(
        coco_ann_file, task_group_file, image_ids_subset=train_ids, seed=seed
    )
    val_dataset = CocoTaskPairDataset(
        coco_ann_file, task_group_file, image_ids_subset=val_ids, seed=seed + 1
    )
    return train_dataset, val_dataset


class ProximityMLP(nn.Module):
    """
    简单 MLP same-group scorer
    输入维度:
      feat_i (6) + feat_j (6) + |feat_i - feat_j| (6) = 18
    """

    def __init__(self, in_dim: int = 18, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),  # logit
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # [B]


def train_proximity_mlp(
    coco_ann_file: Path,
    task_group_file: Path,
    out_path: Path,
    batch_size: int = 256,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda",
):
    train_ds, val_ds = build_coco_pair_datasets(coco_ann_file, task_group_file)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ProximityMLP(in_dim=18, hidden_dim=64).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0

        for feats, labels in train_loader:
            feats = feats.to(device)
            labels = labels.to(device).squeeze(1)

            logits = model(feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * feats.size(0)
            n_samples += feats.size(0)

        avg_train_loss = total_loss / max(1, n_samples)

        # --- validation ---
        model.eval()
        val_loss = 0.0
        val_n = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats = feats.to(device)
                labels = labels.to(device).squeeze(1)

                logits = model(feats)
                loss = criterion(logits, labels)
                val_loss += loss.item() * feats.size(0)
                val_n += feats.size(0)

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.numel()

        avg_val_loss = val_loss / max(1, val_n)
        val_acc = correct / max(1, total)

        print(f"[Epoch {epoch}/{epochs}] "
              f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, val_acc={val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, out_path)
        print(f"[train_proximity_mlp] Saved best model to {out_path}")
    else:
        print("[train_proximity_mlp] No best state? Something is off.")

    return model


if __name__ == "__main__":
    coco_root = config.get_coco_path() / "original"
    coco_ann_file = coco_root / "annotations" / "instances_val2017.json"
    task_group_file = coco_root / "annotations" / "task_groups_val2017.json"
    out_model = coco_root / "models" / "coco_proximity_mlp.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_proximity_mlp(
        coco_ann_file=coco_ann_file,
        task_group_file=task_group_file,
        out_path=out_model,
        device=device,
    )