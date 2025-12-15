# Created by MacBook Pro at 27.11.25
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


from scripts import config


# ---------------------------
# 1. IOU 计算与匹配
# ---------------------------




def bbox_iou_xywh(boxA, boxB):
    """COCO 格式 bbox: [x, y, w, h]"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]
    union = boxA_area + boxB_area - inter_area + 1e-8

    return inter_area / union


def match_predictions_to_gt(pred_boxes, gt_boxes, iou_thresh=0.5):
    """
    简单 greedy 匹配：每个 GT 最多匹配一个预测。
    pred_boxes: list of [x,y,w,h]
    gt_boxes  : list of [x,y,w,h]
    返回：
        matched_gt_idx: len = len(pred_boxes)，
                        若匹到某个 GT，则为其 index，否则为 -1
    """
    matched_gt_idx = [-1] * len(pred_boxes)
    if len(gt_boxes) == 0:
        return matched_gt_idx

    gt_used = [False] * len(gt_boxes)

    # 可以按 score 排序匹配，这里简单按原顺序来
    for i, pb in enumerate(pred_boxes):
        best_iou = 0.0
        best_j = -1
        for j, gb in enumerate(gt_boxes):
            if gt_used[j]:
                continue
            iou = bbox_iou_xywh(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_thresh:
            matched_gt_idx[i] = best_j
            gt_used[best_j] = True

    return matched_gt_idx


# ---------------------------
# 2. 简单 proximity grouping（只基于中心点距离）
# ---------------------------

def simple_proximity_groups(objects, dist_thresh=0.15):
    """
    objects: list of dicts, 每个 dict 至少包含:
        {
          "cx_norm": float [0,1],
          "cy_norm": float [0,1],
          "kind": "gt_task" or "pred",
          "id": arbitrary
        }

    dist_thresh: 归一化距离阈值，越小 group 越小。

    返回：
        group_ids: list[int], len = len(objects)
    """
    n = len(objects)
    if n == 0:
        return []

    # 构建一个简单的连通分量：距离 < 阈值 的点认为属于同一 group
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # pair-wise 计算距离
    for i in range(n):
        for j in range(i + 1, n):
            dx = objects[i]["cx_norm"] - objects[j]["cx_norm"]
            dy = objects[i]["cy_norm"] - objects[j]["cy_norm"]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= dist_thresh:
                union(i, j)

    # 压缩 + group id 映射
    roots = [find(i) for i in range(n)]
    root_to_gid = {}
    group_ids = []
    next_gid = 0
    for r in roots:
        if r not in root_to_gid:
            root_to_gid[r] = next_gid
            next_gid += 1
        group_ids.append(root_to_gid[r])

    return group_ids


# ---------------------------
# 3. Dataset loading function
# ---------------------------

def load_coco_task_grouping_data(
        coco_ann_file: Path,
        task_group_file: Path,
):
    """
    Load COCO annotations and task group annotations.

    Args:
        coco_ann_file: Path to COCO annotations JSON file
        task_group_file: Path to task groups JSON file

    Returns:
        dict containing:
            - coco: COCO object
            - img_to_task_groups: dict mapping image_id to list of task groups
            - eval_image_ids: sorted list of image IDs with task groups
    """
    coco = COCO(str(coco_ann_file))

    # 载入 task group 标注
    with open(task_group_file, "r") as f:
        tg = json.load(f)

    # image_id -> [task_group_1_ann_ids, task_group_2_ann_ids, ...]
    img_to_task_groups = defaultdict(list)
    for g in tg["task_groups"]:
        img_to_task_groups[g["image_id"]].append(g["member_ann_ids"])

    eval_image_ids = sorted(img_to_task_groups.keys())

    return {
        "coco": coco,
        "img_to_task_groups": img_to_task_groups,
        "eval_image_ids": eval_image_ids,
    }




def prepare_image_objects(
        coco,
        img_id: int,
        task_groups: list,
):
    """
    Prepare object data for a single image.

    Args:
        coco: COCO object
        img_id: Image ID
        task_groups: List of task groups (each is a list of annotation IDs)

    Returns:
        dict containing:
            - img_info: COCO image info dict
            - objects: List of object dicts with normalized centers and metadata
            - task_gt_id_set: Set of annotation IDs that are task objects
            - ann_id_to_gt_group: Mapping from annotation ID to GT group index
    """
    img_info = coco.loadImgs(img_id)[0]
    w, h = img_info["width"], img_info["height"]

    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)

    if len(anns) == 0:
        return None

    # 这一张图里所有属于任务 group 的 GT id
    task_gt_id_set = set()
    for g in task_groups:
        task_gt_id_set.update(g)

    # Build ann_id to GT task group mapping
    ann_id_to_gt_group = {}
    for group_idx, g in enumerate(task_groups):
        for ann_id in g:
            ann_id_to_gt_group[ann_id] = group_idx

    # 构建 objects 列表（所有 GT）
    objects = []
    for a in anns:
        x, y, bw, bh = a["bbox"]
        cx = (x + bw / 2.0) / w
        cy = (y + bh / 2.0) / h
        is_task = (a["id"] in task_gt_id_set)
        gt_group_id = ann_id_to_gt_group.get(a["id"], -1)  # -1 for non-task
        objects.append({
            "cx_norm": cx,
            "cy_norm": cy,
            "ann_id": a["id"],
            "is_task": is_task,
            "bbox": a["bbox"],
            "gt_group_id": gt_group_id,
        })

    return {
        "img_info": img_info,
        "objects": objects,
        "task_gt_id_set": task_gt_id_set,
        "ann_id_to_gt_group": ann_id_to_gt_group,
    }


# ---------------------------
# 4. 主评估函数：
#    标准 precision vs group-aware precision
# ---------------------------

def evaluate_group_aware_precision(
        coco_ann_file: Path,
        task_group_file: Path,
        det_result_file: Path,
        iou_thresh: float = 0.5,
        score_thresh: float = 0.3,
        dist_thresh: float = 0.15,
):
    coco = COCO(str(coco_ann_file))

    # 载入 task group 标注
    with open(task_group_file, "r") as f:
        tg = json.load(f)

    # image_id -> list of task groups (each group = list of ann_id)
    img_to_task_groups = defaultdict(list)
    for g in tg["task_groups"]:
        img_to_task_groups[g["image_id"]].append(g["member_ann_ids"])

    # 载入检测结果
    with open(det_result_file, "r") as f:
        dets = json.load(f)

    img_to_preds = defaultdict(list)
    for d in dets:
        if d["score"] < score_thresh:
            continue
        img_to_preds[d["image_id"]].append(d)

    # 只在有 task group 标注的图上评估
    eval_image_ids = sorted(img_to_task_groups.keys())

    total_TP = 0
    total_FP = 0
    total_TaskFP = 0
    total_IgnorableFP = 0

    for img_id in eval_image_ids:
        task_groups = img_to_task_groups[img_id]
        if len(task_groups) == 0:
            continue

        img_info = coco.loadImgs(img_id)[0]
        w, h = img_info["width"], img_info["height"]

        # all GT for IoU matching
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)

        gt_boxes = [a["bbox"] for a in anns]
        gt_ids = [a["id"] for a in anns]

        # 标出哪些 GT 是任务 group 成员
        task_gt_id_set = set()
        for g in task_groups:
            task_gt_id_set.update(g)

        # 预测框
        preds = img_to_preds.get(img_id, [])
        pred_boxes = [p["bbox"] for p in preds]

        # ——— 标准 TP / FP ———
        matched_gt_idx = match_predictions_to_gt(pred_boxes, gt_boxes, iou_thresh=iou_thresh)

        img_TP = 0
        img_FP = 0

        # 为 grouping 准备 object 列表
        objects_for_grouping = []

        # (1) 先把所有任务 GT 加进去
        for a in anns:
            if a["id"] not in task_gt_id_set:
                continue
            x, y, bw, bh = a["bbox"]
            cx = (x + bw / 2.0) / w
            cy = (y + bh / 2.0) / h
            objects_for_grouping.append({
                "cx_norm": cx,
                "cy_norm": cy,
                "kind": "gt_task",
                "ann_id": a["id"],
            })

        # (2) 再把所有预测框加进去
        for idx, p in enumerate(preds):
            x, y, bw, bh = p["bbox"]
            cx = (x + bw / 2.0) / w
            cy = (y + bh / 2.0) / h
            objects_for_grouping.append({
                "cx_norm": cx,
                "cy_norm": cy,
                "kind": "pred",
                "pred_idx": idx,  # 索引回 pred_boxes / matched_gt_idx
            })

        # 如果这一张根本没任务 GT，跳过
        if not any(obj["kind"] == "gt_task" for obj in objects_for_grouping):
            continue

        # grouping
        group_ids = simple_proximity_groups(objects_for_grouping, dist_thresh=dist_thresh)

        # Precompute: 每个 group 里是否有任务 GT
        group_has_task_gt = defaultdict(bool)
        for obj, gid in zip(objects_for_grouping, group_ids):
            if obj["kind"] == "gt_task":
                group_has_task_gt[gid] = True

        # 统计标准 TP/FP & 分 Task-FP / Ignorable-FP
        img_TaskFP = 0
        img_IgnorableFP = 0

        for i, pb in enumerate(pred_boxes):
            if matched_gt_idx[i] != -1:
                img_TP += 1
            else:
                img_FP += 1
                # 找到这个预测在 objects_for_grouping 中的 group
                # kind="pred" 且 pred_idx = i
                for obj, gid in zip(objects_for_grouping, group_ids):
                    if obj.get("kind") == "pred" and obj.get("pred_idx") == i:
                        if group_has_task_gt[gid]:
                            img_TaskFP += 1
                        else:
                            img_IgnorableFP += 1
                        break

        total_TP += img_TP
        total_FP += img_FP
        total_TaskFP += img_TaskFP
        total_IgnorableFP += img_IgnorableFP

    # 汇总指标
    eps = 1e-8
    prec_bbox = total_TP / (total_TP + total_FP + eps)
    prec_group = total_TP / (total_TP + total_TaskFP + eps)

    print("===== Evaluation over {} images =====".format(len(eval_image_ids)))
    print(f"TP          = {total_TP}")
    print(f"FP (all)    = {total_FP}")
    print(f"  Task-FP   = {total_TaskFP}")
    print(f"  Ignorable = {total_IgnorableFP}")
    print(f"Precision (bbox standard)   = {prec_bbox:.4f}")
    print(f"Precision (group-aware)     = {prec_group:.4f}")
    print("=================================")
    return {
        "TP": total_TP,
        "FP": total_FP,
        "TaskFP": total_TaskFP,
        "IgnorableFP": total_IgnorableFP,
        "prec_bbox": prec_bbox,
        "prec_group": prec_group,
    }


def evaluate_gt_task_grouping(
        coco_ann_file: Path,
        task_group_file: Path,
        dist_thresh: float = 0.15,
        min_task_per_group: int = 1,
        iou_thresh: float = 0.5,
        visualize: bool = True,
        num_vis: int = 10,
):
    """
    GT-only 实验：
    - 不看 detector 预测，只用 GT 和你标的 task groups
    - 问题：在一张图里，proximity grouping + 简单 group 规则，
      能否从所有 GT 物体中，挑出"任务对象"（is_task）？

    指标：
      - Group-level TP/FP/FN based on IoU matching between predicted groups and GT groups
      - precision: TP / (TP + FP)
      - recall: TP / (TP + FN)
      - F1
    """
    # Load data
    data = load_coco_task_grouping_data(coco_ann_file, task_group_file)
    coco = data["coco"]
    img_to_task_groups = data["img_to_task_groups"]
    eval_image_ids = data["eval_image_ids"]

    total_TP = 0
    total_FP = 0
    total_FN = 0

    # Store visualization data
    vis_data = []

    for img_id in eval_image_ids:
        task_groups = img_to_task_groups[img_id]
        if len(task_groups) == 0:
            continue

        # Prepare image objects
        img_data = prepare_image_objects(coco, img_id, task_groups)
        if img_data is None:
            continue

        objects = img_data["objects"]
        img_info = img_data["img_info"]

        # 在这一张图的所有 GT 上做 proximity grouping
        group_ids = simple_proximity_groups(objects, dist_thresh=dist_thresh)

        # 统计每个 group 中有多少 task 对象
        group_task_count = defaultdict(int)
        for obj, gid in zip(objects, group_ids):
            if obj["is_task"]:
                group_task_count[gid] += 1

        # 根据 group_task_count 决定 group 是否是"任务 group"
        group_is_task = {}
        for gid, cnt in group_task_count.items():
            group_is_task[gid] = (cnt >= min_task_per_group)

        # 没有出现过的 gid 默认 False（纯背景 group）
        for gid in set(group_ids):
            if gid not in group_is_task:
                group_is_task[gid] = False

        # Build GT group bounding boxes
        gt_task_objects = [obj for obj in objects if obj["is_task"]]
        gt_groups_dict = defaultdict(list)
        gt_group_boxes = {}

        for obj in gt_task_objects:
            gt_group_id = obj["gt_group_id"]
            gt_groups_dict[gt_group_id].append(obj)

        for gt_gid, group_objs in gt_groups_dict.items():
            min_x = min(obj["bbox"][0] for obj in group_objs)
            min_y = min(obj["bbox"][1] for obj in group_objs)
            max_x = max(obj["bbox"][0] + obj["bbox"][2] for obj in group_objs)
            max_y = max(obj["bbox"][1] + obj["bbox"][3] for obj in group_objs)
            gt_width = max_x - min_x
            gt_height = max_y - min_y
            gt_group_boxes[gt_gid] = [min_x, min_y, gt_width, gt_height]

        # Build predicted group bounding boxes (only task groups)
        pred_groups = defaultdict(list)
        for obj, gid in zip(objects, group_ids):
            pred_groups[gid].append(obj)

        pred_task_groups = {}  # {pred_gid: [x, y, w, h]}
        for gid, group_objs in pred_groups.items():
            if not group_is_task[gid]:
                continue
            # Only consider groups with all task objects
            gt_task_statuses = [obj["is_task"] for obj in group_objs]
            if not all(gt_task_statuses):
                continue

            min_x = min(obj["bbox"][0] for obj in group_objs)
            min_y = min(obj["bbox"][1] for obj in group_objs)
            max_x = max(obj["bbox"][0] + obj["bbox"][2] for obj in group_objs)
            max_y = max(obj["bbox"][1] + obj["bbox"][3] for obj in group_objs)
            pred_width = max_x - min_x
            pred_height = max_y - min_y
            pred_task_groups[gid] = [min_x, min_y, pred_width, pred_height]

        # Match predicted groups to GT groups using IoU
        matched_gt_groups = set()
        img_TP = 0
        img_FP = 0

        for pred_gid, pred_box in pred_task_groups.items():
            # Find which GT group(s) the objects in this pred group belong to
            group_objs = pred_groups[pred_gid]
            gt_group_ids_in_pred = set(obj["gt_group_id"] for obj in group_objs if obj["is_task"])

            # Only consider as potential TP if all objects belong to same GT group
            if len(gt_group_ids_in_pred) == 1:
                gt_gid = list(gt_group_ids_in_pred)[0]
                if gt_gid in gt_group_boxes and gt_gid not in matched_gt_groups:
                    gt_box = gt_group_boxes[gt_gid]
                    iou = bbox_iou_xywh(pred_box, gt_box)

                    if iou >= iou_thresh:
                        img_TP += 1
                        matched_gt_groups.add(gt_gid)
                        continue

            # If we get here, it's a false positive
            img_FP += 1

        # FN: GT groups that were not matched
        img_FN = len(gt_group_boxes) - len(matched_gt_groups)

        total_TP += img_TP
        total_FP += img_FP
        total_FN += img_FN

        # Store data for visualization
        vis_data.append({
            "img_id": img_id,
            "img_info": img_info,
            "objects": objects,
            "group_ids": group_ids,
            "group_is_task": group_is_task,
            "num_gt_groups": len(task_groups),
        })

    # 汇总指标
    eps = 1e-8
    precision = total_TP / (total_TP + total_FP + eps)
    recall = total_TP / (total_TP + total_FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    print("===== GT-only Task Group Evaluation (Group-level) =====")
    print(f"Images evaluated       : {len(eval_image_ids)}")
    print(f"  TP (matched groups)  : {total_TP}")
    print(f"  FP (extra groups)    : {total_FP}")
    print(f"  FN (missed groups)   : {total_FN}")
    print(f"Precision (group)      : {precision:.4f}")
    print(f"Recall (group)         : {recall:.4f}")
    print(f"F1 (group)             : {f1:.4f}")
    print("========================================================")

    # Visualize results
    if visualize and len(vis_data) > 0:
        coco_root = config.get_coco_path() / "original"
        img_dir = coco_root / "val2017"

        visualize_task_grouping(vis_data, img_dir, num_vis=num_vis)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": total_TP,
        "FP": total_FP,
        "FN": total_FN,
        "num_images": len(eval_image_ids),
    }


def visualize_task_grouping(vis_data, img_dir, num_vis=10):
    """
    Visualize ground truth vs predictions in 3 panels side-by-side:
    Left: All object bounding boxes (task objects in green, non-task in gray)
    Middle: Ground truth task groups (using actual annotated groups, all in green)
    Right: Predicted proximity groups - TP in green (if matches GT group position/size), FP in yellow (otherwise)
    Each group shown as one merged bounding box.
    """
    # Randomly sample images to visualize
    vis_samples = random.sample(vis_data, min(num_vis, len(vis_data)))

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    except:
        font = ImageFont.load_default()
        font_large = ImageFont.load_default()

    # IoU threshold: predicted group box must have IoU >= this with GT group box to be TP
    iou_thresh = 0.8

    for idx, sample in enumerate(vis_samples):
        img_id = sample["img_id"]
        img_info = sample["img_info"]
        objects = sample["objects"]
        group_ids = sample["group_ids"]
        group_is_task = sample["group_is_task"]
        num_gt_groups = sample["num_gt_groups"]

        # Load image
        img_path = img_dir / img_info["file_name"]
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            continue

        img_orig = Image.open(img_path).convert("RGB")
        width, height = img_orig.size

        # Create three images: all objects, GT groups, and predictions
        img_all = img_orig.copy()
        img_gt = img_orig.copy()
        img_pred = img_orig.copy()

        draw_all = ImageDraw.Draw(img_all, "RGBA")
        draw_gt = ImageDraw.Draw(img_gt, "RGBA")
        draw_pred = ImageDraw.Draw(img_pred, "RGBA")

        # Draw all objects (left panel)
        task_obj_count = 0
        non_task_obj_count = 0

        for obj in objects:
            x, y, w, h = obj["bbox"]
            is_task = obj["is_task"]

            if is_task:
                color = (0, 255, 0)  # Green for task objects
                label = "T"
                task_obj_count += 1
            else:
                color = (128, 128, 128)  # Gray for non-task objects
                label = "N"
                non_task_obj_count += 1

            draw_all.rectangle([x, y, x + w, y + h],
                             outline=color + (255,), width=2)
            draw_all.text((x, y - 22), label,
                         fill=color + (255,), font=font)

        # Draw ground truth groups (middle panel)
        gt_task_objects = [obj for obj in objects if obj["is_task"]]
        gt_color = (0, 255, 0)  # Green for all GT task groups

        gt_groups_dict = defaultdict(list)
        gt_group_boxes = {}  # Store merged bbox of each GT group: {gt_gid: [x, y, w, h]}

        if gt_task_objects:
            # Group by actual GT group ID from annotation
            for obj in gt_task_objects:
                gt_group_id = obj["gt_group_id"]
                gt_groups_dict[gt_group_id].append(obj)

            # Draw each actual GT task group with green color and compute merged bbox
            for gt_gid, group_objs in gt_groups_dict.items():
                min_x = min(obj["bbox"][0] for obj in group_objs)
                min_y = min(obj["bbox"][1] for obj in group_objs)
                max_x = max(obj["bbox"][0] + obj["bbox"][2] for obj in group_objs)
                max_y = max(obj["bbox"][1] + obj["bbox"][3] for obj in group_objs)

                # Store GT group merged bbox in [x, y, w, h] format
                gt_width = max_x - min_x
                gt_height = max_y - min_y
                gt_group_boxes[gt_gid] = [min_x, min_y, gt_width, gt_height]

                draw_gt.rectangle([min_x, min_y, max_x, max_y],
                                outline=gt_color + (255,), width=4)
                draw_gt.text((min_x, min_y - 25), f"GT-G{gt_gid} ({len(group_objs)})",
                            fill=gt_color + (255,), font=font)

        # Draw predictions (right panel)
        pred_groups = defaultdict(list)
        for obj, gid in zip(objects, group_ids):
            pred_groups[gid].append(obj)

        tp_count = 0
        fp_count = 0
        matched_gt_groups = set()  # Track which GT groups have been matched

        # Draw each predicted proximity group
        for gid, group_objs in pred_groups.items():
            # Check if this group contains any task objects
            gt_task_statuses = [obj["is_task"] for obj in group_objs]
            has_any_task = any(gt_task_statuses)
            all_task = all(gt_task_statuses)

            # Only draw groups that contain at least one task object
            if not has_any_task:
                continue

            # Merge all boxes in this predicted group
            min_x = min(obj["bbox"][0] for obj in group_objs)
            min_y = min(obj["bbox"][1] for obj in group_objs)
            max_x = max(obj["bbox"][0] + obj["bbox"][2] for obj in group_objs)
            max_y = max(obj["bbox"][1] + obj["bbox"][3] for obj in group_objs)

            # Predicted group merged bbox in [x, y, w, h] format
            pred_box = [min_x, min_y, max_x - min_x, max_y - min_y]

            # Check if this predicted group matches a GT group
            is_tp = False
            matched_gt_gid = None

            if all_task:
                # Find which GT group(s) these objects belong to
                gt_group_ids_in_pred = set(obj["gt_group_id"] for obj in group_objs)

                # If all objects belong to the same GT group, check IoU
                if len(gt_group_ids_in_pred) == 1:
                    gt_gid = list(gt_group_ids_in_pred)[0]
                    if gt_gid in gt_group_boxes and gt_gid not in matched_gt_groups:
                        gt_box = gt_group_boxes[gt_gid]
                        iou = bbox_iou_xywh(pred_box, gt_box)

                        # TP only if IoU is high enough (position, width, height all similar)
                        if iou >= iou_thresh:
                            is_tp = True
                            matched_gt_gid = gt_gid
                            matched_gt_groups.add(gt_gid)

            if is_tp:
                # TP: matches a GT group in position and size
                color = (0, 255, 0)  # Green for TP
                label = f"TP-G{gid}→GT{matched_gt_gid} ({len(group_objs)})"
                tp_count += 1
            else:
                # FP: either mixed objects, multiple GT groups, or poor IoU match
                color = (255, 255, 0)  # Yellow for FP
                task_in_group = sum(gt_task_statuses)
                if all_task:
                    # All task but failed IoU check or multiple GT groups
                    label = f"FP-G{gid} ({len(group_objs)}, poor match)"
                else:
                    label = f"FP-G{gid} ({task_in_group}/{len(group_objs)})"
                fp_count += 1

            draw_pred.rectangle([min_x, min_y, max_x, max_y],
                              outline=color + (255,), width=4)
            draw_pred.text((min_x, min_y - 25), label,
                          fill=color + (255,), font=font)

        # Add titles and stats
        draw_all.text((10, 10), "All Objects", fill=(255, 255, 255, 255), font=font_large)
        draw_all.text((10, 45), f"Task: {task_obj_count}, Non-task: {non_task_obj_count}",
                     fill=(255, 255, 255, 255), font=font)

        draw_gt.text((10, 10), "Ground Truth (Task Only)", fill=(255, 255, 255, 255), font=font_large)
        draw_gt.text((10, 45), f"Task groups: {num_gt_groups}, Task objects: {len(gt_task_objects)}",
                    fill=(255, 255, 255, 255), font=font)

        draw_pred.text((10, 10), "Predictions (Proximity Groups)", fill=(255, 255, 255, 255), font=font_large)
        draw_pred.text((10, 45), f"TP groups: {tp_count}, FP groups: {fp_count}",
                      fill=(255, 255, 255, 255), font=font)

        # Create 3-panel side-by-side image
        combined = Image.new('RGB', (width * 3, height))
        combined.paste(img_all.convert("RGB"), (0, 0))
        combined.paste(img_gt.convert("RGB"), (width, 0))
        combined.paste(img_pred.convert("RGB"), (width * 2, 0))

        # Draw separator lines
        draw_combined = ImageDraw.Draw(combined)
        draw_combined.line([(width, 0), (width, height)], fill=(255, 255, 255), width=3)
        draw_combined.line([(width * 2, 0), (width * 2, height)], fill=(255, 255, 255), width=3)

        # Save visualization
        out_dir = config.get_proj_output_path() / "coco_task_group_vis"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_file = out_dir / f"task_group_vis_{idx}_{img_id}.jpg"
        combined.save(output_file)
        print(f"Saved visualization to: {output_file}")

    print(f"\n[visualize_task_grouping] Saved {len(vis_samples)} visualizations")

    #     dist_thresh=0.15,
    #

if __name__ == "__main__":
    coco_ann_file = config.get_coco_path() / "original" / "annotations" / "instances_val2017.json"
    task_group_file = config.get_coco_path() / "original" / "annotations" / "task_groups_val2017.json"

    print("\n=== Evaluating GT Task Grouping ===")
    evaluate_gt_task_grouping(
        coco_ann_file=coco_ann_file,
        task_group_file=task_group_file,
        dist_thresh=0.15,
        min_task_per_group=1,
        iou_thresh=0.8,
        visualize=True,
        num_vis=10,
    )