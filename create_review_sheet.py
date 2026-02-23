#!/usr/bin/env python3
"""
Create visual comparison review sheets for Gestalt principle datasets.

For each of the 5 principles (proximity, similarity, closure_triangle, continuity, symmetry),
generates a comparison image grid showing positive vs negative samples with render and bbox
visualizations side by side, annotated with metadata from the JSON files.

Outputs:
  - gen_data_3d/review_sheets/{principle}_comparison.png  (one per principle)
  - gen_data_3d/review_sheets/all_principles_overview.png (combined overview)
"""

import json
import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# === Configuration ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RENDER_DIR = os.path.join(BASE_DIR, "gen_data_3d", "test_renders")
BBOX_DIR = os.path.join(BASE_DIR, "gen_data_3d", "test_renders_bbox")
OUTPUT_DIR = os.path.join(BASE_DIR, "gen_data_3d", "review_sheets")

PRINCIPLES = ["proximity", "similarity", "closure_triangle", "continuity", "symmetry"]
NUM_SAMPLES = 3  # Number of samples per category to display
CELL_SIZE = 3.0  # Inches per image cell


def load_json(json_path):
    """Load and parse a JSON metadata file."""
    with open(json_path, "r") as f:
        return json.load(f)


def extract_metadata(data):
    """Extract relevant metadata from a JSON data dict."""
    objects = data.get("objects", [])
    logics = data.get("logics", {})

    group_ids = sorted(set(obj.get("group_id", -1) for obj in objects))
    num_objects = len(objects)
    is_positive = logics.get("is_positive", None)
    principle = logics.get("principle", "unknown")

    return {
        "group_ids": group_ids,
        "num_objects": num_objects,
        "is_positive": is_positive,
        "principle": principle,
    }


def build_annotation(label, metadata):
    """Build a multi-line annotation string for display below an image."""
    groups_str = "{" + ", ".join(str(g) for g in metadata["group_ids"]) + "}"
    n_obj = metadata["num_objects"]
    is_pos = metadata["is_positive"]
    lines = [
        label,
        "groups: " + groups_str,
        "objects: " + str(n_obj),
        "is_positive: " + str(is_pos),
    ]
    return "\n".join(lines)


def create_principle_sheet(principle):
    """
    Create a comparison sheet for a single principle.

    Layout:
      Title row: principle name
      Row 1: 3 positive renders | 3 positive bbox images  (6 across)
      Row 2: 3 negative renders | 3 negative bbox images  (6 across)

    Each image is annotated with metadata from its JSON file.
    """
    ncols = NUM_SAMPLES * 2  # 3 renders + 3 bbox = 6
    nrows = 2  # positive row, negative row

    # Extra vertical space for title and annotations
    fig_width = ncols * CELL_SIZE
    fig_height = nrows * (CELL_SIZE + 0.9) + 1.0

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))

    title_text = "Principle: " + principle.replace("_", " ").title()
    fig.suptitle(title_text, fontsize=20, fontweight="bold", y=0.98)

    for row_idx, category in enumerate(["positive", "negative"]):
        for sample_idx in range(NUM_SAMPLES):
            sample_name = "{:05d}".format(sample_idx)

            # --- Paths ---
            render_path = os.path.join(
                RENDER_DIR, principle, "train", category, sample_name + ".png"
            )
            bbox_path = os.path.join(
                BBOX_DIR, principle, "train", category, sample_name + ".png"
            )
            json_path = os.path.join(
                RENDER_DIR, principle, "train", category, sample_name + ".json"
            )

            # --- Load metadata ---
            metadata = extract_metadata(load_json(json_path))
            label = "Positive" if category == "positive" else "Negative"
            annotation = build_annotation(label, metadata)

            # --- Render image (left half of the row) ---
            ax_render = axes[row_idx, sample_idx]
            if os.path.exists(render_path):
                img = mpimg.imread(render_path)
                ax_render.imshow(img)
            else:
                ax_render.text(0.5, 0.5, "MISSING", ha="center", va="center",
                               transform=ax_render.transAxes, fontsize=12, color="red")
            ax_render.set_xticks([])
            ax_render.set_yticks([])
            # Color-code the border
            border_color = "green" if category == "positive" else "red"
            for spine in ax_render.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2)

            # Title on top for first row only
            if row_idx == 0 and sample_idx == 0:
                ax_render.set_title("Renders", fontsize=12, fontweight="bold", pad=8)

            # Annotation below
            ax_render.set_xlabel(annotation, fontsize=7, labelpad=6,
                                 ha="center", multialignment="left",
                                 fontfamily="monospace")

            # --- Bbox image (right half of the row) ---
            ax_bbox = axes[row_idx, NUM_SAMPLES + sample_idx]
            if os.path.exists(bbox_path):
                img_bbox = mpimg.imread(bbox_path)
                ax_bbox.imshow(img_bbox)
            else:
                ax_bbox.text(0.5, 0.5, "MISSING", ha="center", va="center",
                             transform=ax_bbox.transAxes, fontsize=12, color="red")
            ax_bbox.set_xticks([])
            ax_bbox.set_yticks([])
            for spine in ax_bbox.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2)

            if row_idx == 0 and sample_idx == 0:
                ax_bbox.set_title("Bounding Boxes", fontsize=12, fontweight="bold", pad=8)

            ax_bbox.set_xlabel(annotation, fontsize=7, labelpad=6,
                               ha="center", multialignment="left",
                               fontfamily="monospace")

    # Add row labels on the left side
    for row_idx, cat_label in enumerate(["Positive Samples", "Negative Samples"]):
        color = "green" if row_idx == 0 else "red"
        axes[row_idx, 0].set_ylabel(cat_label, fontsize=13, fontweight="bold",
                                     color=color, labelpad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, principle + "_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved: " + out_path)
    return out_path


def create_combined_overview(principle_sheets):
    """
    Create a single combined overview image that stacks all principle sheets vertically.
    Reads the saved per-principle PNGs and composites them.
    """
    images = []
    for path in principle_sheets:
        img = mpimg.imread(path)
        images.append(img)

    # All images may have different widths due to tight_layout; pad to max width
    max_width = max(im.shape[1] for im in images)
    padded = []
    for img in images:
        h, w = img.shape[:2]
        if w < max_width:
            channels = img.shape[2] if img.ndim == 3 else 1
            if img.ndim == 3:
                pad = np.ones((h, max_width - w, channels), dtype=img.dtype)
            else:
                pad = np.ones((h, max_width - w), dtype=img.dtype)
            img = np.concatenate([img, pad], axis=1)
        padded.append(img)

    # Add a thin separator between principle sections
    separator_height = 20
    if padded[0].ndim == 3:
        channels = padded[0].shape[2]
        sep = np.ones((separator_height, max_width, channels), dtype=padded[0].dtype) * 0.85
    else:
        sep = np.ones((separator_height, max_width), dtype=padded[0].dtype) * 0.85

    combined_parts = []
    for i, img in enumerate(padded):
        combined_parts.append(img)
        if i < len(padded) - 1:
            combined_parts.append(sep)

    combined = np.concatenate(combined_parts, axis=0)

    # Save combined overview
    out_path = os.path.join(OUTPUT_DIR, "all_principles_overview.png")
    fig, ax = plt.subplots(1, 1, figsize=(combined.shape[1] / 150, combined.shape[0] / 150),
                           dpi=150)
    ax.imshow(combined)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)
    print("  Saved combined overview: " + out_path)
    return out_path


def main():
    print("=" * 60)
    print("Gestalt Principle Review Sheet Generator")
    print("=" * 60)
    print()

    # Verify directories exist
    for d, name in [(RENDER_DIR, "test_renders"), (BBOX_DIR, "test_renders_bbox")]:
        if not os.path.isdir(d):
            print("ERROR: " + name + " directory not found at " + d)
            return

    # Generate per-principle sheets
    principle_sheets = []
    for principle in PRINCIPLES:
        print("[" + principle + "] Generating comparison sheet...")
        path = create_principle_sheet(principle)
        principle_sheets.append(path)

    print()
    print("Generating combined overview...")
    combined_path = create_combined_overview(principle_sheets)

    print()
    print("=" * 60)
    print("Done! Output files:")
    for path in principle_sheets:
        print("  " + path)
    print("  " + combined_path)
    print("=" * 60)


if __name__ == "__main__":
    main()
