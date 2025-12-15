"""
Interactive NYU RGB-D labeling helper.

Usage examples:
  python nyu_rgbd_labeling.py --mat_file path/to/nyu_depth_v2_labeled.mat --out labels_nyu.json

The script will load images and bounding boxes from the .mat file, show one annotated image at a time
and ask you to type comma-separated group ids for all bounding boxes (example: "0,0,1,1").
Commands: "s" to skip image, "q" to quit (saves progress), "r" to redraw.
"""
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import h5py

def load_nyu_data(mat_file):
    """Load images and bounding boxes from NYU RGB-D .mat file"""
    with h5py.File(mat_file, 'r') as f:
        # Extract images and labels from .mat file
        images = np.array(f['images'])  # Should be (3, H, W, N) array in HDF5
        labels = np.array(f['labels'])  # Should be (H, W, N) array with object instance IDs

        # HDF5 format might have different axis order, transpose if needed
        if images.ndim == 4:
            # Convert from (3, H, W, N) to (H, W, 3, N)
            images = np.transpose(images, (1, 2, 0, 3))

    # Convert to format: {image_idx: [[x,y,w,h], ...]}
    mapping = {}

    for img_idx in range(images.shape[3]):
        # Extract bounding boxes from label mask
        label_mask = labels[:, :, img_idx]
        unique_labels = np.unique(label_mask)
        unique_labels = unique_labels[unique_labels > 0]  # Remove background

        bboxes = []
        for label_id in unique_labels:
            mask = (label_mask == label_id)
            if not mask.any():
                continue

            # Find bounding box coordinates
            rows, cols = np.where(mask)
            y_min, y_max = rows.min(), rows.max()
            x_min, x_max = cols.min(), cols.max()

            # Convert to [x, y, w, h] format
            bbox = [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]
            bboxes.append(bbox)

        if len(bboxes) > 0:
            mapping[f"image_{img_idx:04d}"] = bboxes

    return mapping, images, labels

def get_image_from_mat(images, img_idx):
    """Extract PIL Image from .mat images array"""
    # images is (H, W, 3, N), we want (H, W, 3) for img_idx
    img_array = images[:, :, :, img_idx]
    # Normalize to 0-255 range if needed
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
    else:
        img_array = img_array.astype(np.uint8)
    return Image.fromarray(img_array)

def draw_bboxes_on_image(img, bboxes, draw_idx=True):
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()
    for idx, bb in enumerate(bboxes):
        x, y, w, h = [int(round(v)) for v in bb]
        color = (255, 0, 0, 200)
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        if draw_idx:
            draw.text((x, max(0, y - 20)), f"{idx}", fill=(255,255,255,255), font=font)
    return img

def prompt_for_groups(file_name, bboxes):
    n = len(bboxes)
    print(f"\nImage: {file_name} | {n} boxes")
    for i, bb in enumerate(bboxes):
        print(f"  [{i}] bbox = {bb}")
    print("Enter group ids as comma-separated integers (e.g. 0,0,1,1). Commands: s=skip, q=quit(save), r=redraw")
    while True:
        raw = input("groups> ").strip()
        if raw.lower() == "s":
            return None  # skip
        if raw.lower() == "q":
            return "QUIT"
        if raw.lower() == "r":
            return "REDRAW"
        parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
        if len(parts) != n:
            print(f"Expected {n} values, got {len(parts)}. Try again or use 's' to skip.")
            continue
        try:
            gids = [int(p) for p in parts]
            return gids
        except ValueError:
            print("All entries must be integers. Try again.")

def main(args):
    mat_file = Path(args.mat_file)
    out_json = Path(args.out_json)

    mapping, images, labels = load_nyu_data(mat_file)
    print(f"Loaded {len(mapping)} images from {mat_file}")

    # Load existing results if present (resume)
    results = {}
    if out_json.exists():
        try:
            results = json.loads(out_json.read_text())
            print(f"Resuming: loaded {len(results)} existing annotations from {out_json}")
        except Exception:
            results = {}

    image_list = sorted(mapping.keys())
    for file_name in image_list:
        if file_name in results and not args.overwrite:
            continue

        # Extract image index from filename
        img_idx = int(file_name.split("_")[1])

        bboxes = mapping[file_name]
        if len(bboxes) == 0:
            print(f"No boxes for {file_name}, skipping")
            continue

        img = get_image_from_mat(images, img_idx)
        annotated = draw_bboxes_on_image(img.copy(), bboxes)

        # Optionally save annotated preview to out_dir for user to open
        preview_dir = out_json.parent / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_path = preview_dir / f"preview_{file_name}.png"
        annotated.save(preview_path)
        print(f"Preview saved to: {preview_path} (also opened)")

        # Show image (may open external viewer)
        annotated.show()

        while True:
            res = prompt_for_groups(file_name, bboxes)
            if res == "QUIT":
                print("Quitting and saving progress...")
                out_json.write_text(json.dumps(results, indent=2))
                return
            if res == "REDRAW":
                annotated.show()
                continue
            if res is None:
                print("Skipped.")
                break
            # Valid group ids list
            results[file_name] = res
            print(f"Saved groups for {file_name}: {res}")
            break

        # Save after each image to allow resuming
        out_json.write_text(json.dumps(results, indent=2))

    print(f"All done. Saved {len(results)} annotations to {out_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive NYU RGB-D grouping labeling")
    parser.add_argument("--mat_file", type=str, default=None, help="Path to NYU .mat file")
    parser.add_argument("--out_json", type=str, default="nyu_group_labels.json", help="Output JSON to store group labels")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing entries in output")
    args = parser.parse_args()

    # Set default paths
    if args.mat_file is None:
        args.mat_file = Path(__file__).parent / ".." / "storage" / "nyu_depth_v2_labeled.mat"
    if args.out_json == "nyu_group_labels.json":
        args.out_json = Path(__file__).parent / ".." / "storage" / "nyu_depth_v2_labels.json"

    main(args)
