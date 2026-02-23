"""
Overlay bounding boxes from scene JSON onto rendered images.

Usage:
    python -m scripts.clevr3d.visualize_bboxes [--input_dir DIR] [--output_dir DIR]
"""

import json
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm


def draw_bboxes(img_path, json_path, out_path):
    """Draw bounding boxes from JSON onto the rendered image."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    with open(json_path) as f:
        data = json.load(f)

    objects = data.get("objects", data.get("img_data", []))
    if not objects:
        return

    draw = ImageDraw.Draw(img, "RGBA")

    # Color by group_id
    group_ids = sorted(set(o["group_id"] for o in objects))
    cmap = cm.get_cmap("tab10")
    gid_color = {gid: tuple(int(255 * c) for c in cmap(i % 10)[:3])
                 for i, gid in enumerate(group_ids)}

    for i, obj in enumerate(objects):
        x = obj["x"]
        y = obj["y"]
        size = obj["size"]

        x1 = int((x - size / 2) * w)
        y1 = int((y - size / 2) * h)
        x2 = int((x + size / 2) * w)
        y2 = int((y + size / 2) * h)

        gid = obj["group_id"]
        color = gid_color.get(gid, (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=2)
        # Semi-transparent fill so groups are visually obvious
        draw.rectangle([x1, y1, x2, y2], fill=color + (40,))

        label = f"g{gid}"
        draw.text((x1 + 2, y1 - 12), label, fill=color + (255,))

    # Draw a legend in the top-left corner
    legend_y = 5
    for gid in group_ids:
        color = gid_color[gid]
        count = sum(1 for o in objects if o["group_id"] == gid)
        draw.rectangle([5, legend_y, 20, legend_y + 12], fill=color + (200,),
                        outline=color + (255,))
        draw.text((24, legend_y), f"group {gid} ({count} obj)", fill=(255, 255, 255, 255))
        legend_y += 16

    img.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="gen_data_3d/test_renders")
    parser.add_argument("--output_dir", default="gen_data_3d/test_renders_bbox")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    png_files = sorted(input_dir.rglob("*.png"))
    print(f"Found {len(png_files)} images in {input_dir}")

    count = 0
    for png in png_files:
        json_path = png.with_suffix(".json")
        if not json_path.exists():
            continue

        rel = png.relative_to(input_dir)
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        draw_bboxes(png, json_path, out_path)
        count += 1

    print(f"Saved {count} annotated images to {output_dir}/")


if __name__ == "__main__":
    main()
