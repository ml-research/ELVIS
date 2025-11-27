# Created by MacBook Pro at 27.11.25


import json
import os
from collections import defaultdict
from pathlib import Path
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

from pycocotools.coco import COCO

from scripts import config


COCO_ROOT = config.get_coco_path() / "original"
ann_file = COCO_ROOT / "annotations" / "instances_val2017.json"
img_dir = COCO_ROOT / "val2017"
out_path = COCO_ROOT / "annotations" / "task_groups_val2017.json"

coco = COCO(str(ann_file))

# Load existing annotations if file exists
all_groups = []
annotated_image_ids = set()

if out_path.exists():
    print(f"Loading existing annotations from {out_path}")
    with open(out_path, "r") as f:
        existing_data = json.load(f)
        all_groups = existing_data.get("task_groups", [])
        # Get unique image IDs from the images list (not from groups)
        annotated_image_ids = {img["image_id"] for img in existing_data.get("images", [])}
    print(f"Found {len(annotated_image_ids)} already annotated images with {len(all_groups)} total groups")
else:
    print("No existing annotations found, starting fresh")

# 你可以只选一部分 image_id，比如之前的 coco_prox_100 那批
img_ids = coco.getImgIds()
random.shuffle(img_ids)  # Randomize the order
img_ids = img_ids[:100]   # 先标 50 张玩玩

for img_id in img_ids:
    # Skip already annotated images
    if img_id in annotated_image_ids:
        print(f"Skipping image {img_id} (already annotated)")
        continue

    img_info = coco.loadImgs(img_id)[0]
    file_name = img_info["file_name"]
    img_path = img_dir / file_name

    img = cv2.imread(str(img_path))
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)

    # 只标注物体数量在5-15的图片
    if len(anns) < 5 or len(anns) > 15:
        continue

    # 可选：只显示某些类别
    # anns = [a for a in anns if a["category_id"] in allowed_cat_ids]

    # 显示图像和 bbox
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    ax = plt.gca()

    for idx, ann in enumerate(anns):
        x,y,w,h = ann["bbox"]
        rect = patches.Rectangle((x, y), w, h, linewidth=1.5,
                                 edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        # 在框旁边打 idx 编号 - 增大字体
        ax.text(x, y, str(idx), color='yellow', fontsize=18, weight='bold',
                bbox=dict(facecolor='black', alpha=0.7, pad=2))

    # 在左上角显示总物体数
    ax.text(10, 30, f'Total Objects: {len(anns)}', color='white', fontsize=20, weight='bold',
            bbox=dict(facecolor='red', alpha=0.7, pad=5))

    plt.title(f"image_id={img_id}, file={file_name}")
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=False)

    print(f"Image {img_id} ({file_name}), total objects: {len(anns)}")
    print("输入任务 group，格式例如：")
    print("  0 1 2; 5 6 7")
    print("表示两个 group: {0,1,2} 和 {5,6,7}")
    print("直接回车表示跳过这张图，输入 q 退出标注。")
    user_in = input("task groups indices: ").strip()

    plt.close()

    if user_in.lower() == 'q':
        break
    if user_in == "":
        continue

    # 解析输入
    group_strs = [s.strip() for s in user_in.split(";") if s.strip()]
    groups_added = 0
    for gid, g_str in enumerate(group_strs):
        idx_list = [int(s) for s in g_str.split(",") if s.strip().isdigit()]
        if len(idx_list) == 0:
            continue
        member_ann_ids = [anns[i]["id"] for i in idx_list]
        all_groups.append({
            "image_id": img_id,
            "group_id": gid,
            "member_ann_ids": member_ann_ids,
            "is_task": True
        })
        groups_added += 1

    # Mark this image as annotated
    annotated_image_ids.add(img_id)

    # Save after each image annotation to preserve work
    out = {
        "images": [{"image_id": iid} for iid in sorted(annotated_image_ids)],
        "task_groups": all_groups
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"✓ Saved progress: Added {groups_added} groups for image {img_id}. Total: {len(all_groups)} groups across {len(annotated_image_ids)} images\n")

# Final save (redundant but safe)
out = {
    "images": [{"image_id": iid} for iid in annotated_image_ids],
    "task_groups": all_groups
}
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"Final save: {len(all_groups)} task groups for {len(annotated_image_ids)} images to {out_path}")
