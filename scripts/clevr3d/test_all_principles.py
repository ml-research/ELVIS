"""
Generate positive and negative examples for all 5 Gestalt principles
with train/test split.

Output structure:
    gen_data_3d/test_renders/{principle}/train/{positive,negative}/{00000..00009}.{png,json}
    gen_data_3d/test_renders/{principle}/test/{positive,negative}/{00000..00009}.{png,json}

Run from the ELVIS directory:
    python -m scripts.clevr3d.test_all_principles
"""

import json
import os
import traceback
from pathlib import Path

from scripts.clevr3d import config_3d
from scripts.clevr3d.blender_renderer import render_scene

OUT_DIR = Path("gen_data_3d/test_renders")
RES = (480, 480)
NUM_SAMPLES_PER_SPLIT = 10  # 10 train + 10 test = 20 per label

results = {}


def render_samples(principle_name, generator_fn, gen_kwargs_pos, gen_kwargs_neg):
    """Generate train/test splits with NUM_SAMPLES_PER_SPLIT pos + neg each."""
    print(f"\n{'='*60}")
    print(f"  {principle_name.upper()}")
    print(f"{'='*60}")

    for split in ["train", "test"]:
        print(f"\n  --- {split} ---")
        for label, is_pos, gen_kwargs in [
            ("positive", True, gen_kwargs_pos),
            ("negative", False, gen_kwargs_neg),
        ]:
            sub_dir = OUT_DIR / principle_name / split / label
            os.makedirs(sub_dir, exist_ok=True)

            ok_count = 0
            fail_count = 0

            for i in range(NUM_SAMPLES_PER_SPLIT):
                kw = {**gen_kwargs, "is_positive": is_pos}
                tag = f"{principle_name}/{split}/{label}/{i:05d}"
                try:
                    objs, logics = generator_fn(**kw)
                    if objs is None:
                        print(f"  [{tag}] FAILED: generator returned None")
                        fail_count += 1
                        continue

                    n_objs = len(objs)
                    out_path = sub_dir / f"{i:05d}.png"
                    render_scene(objs, str(out_path), resolution=RES)

                    if out_path.exists():
                        kb = out_path.stat().st_size / 1024
                        print(f"  [{tag}] OK: {n_objs} objects ({kb:.0f} KB)")
                        ok_count += 1

                        json_path = sub_dir / f"{i:05d}.json"
                        with open(json_path, 'w') as f:
                            json.dump({"objects": objs, "logics": logics}, f, indent=2)
                    else:
                        print(f"  [{tag}] FAILED: render produced no file")
                        fail_count += 1
                except Exception as e:
                    print(f"  [{tag}] ERROR: {e}")
                    traceback.print_exc()
                    fail_count += 1

            key = f"{principle_name}_{split}_{label}"
            results[key] = f"{ok_count}/{NUM_SAMPLES_PER_SPLIT} ok"
            print(f"  => {split}/{label}: {ok_count}/{NUM_SAMPLES_PER_SPLIT} succeeded")


# ============================================================
# 1. PROXIMITY
# ============================================================
from scripts.clevr3d.adapt_2d_to_3d import proximity_adapted

render_samples(
    "proximity",
    proximity_adapted,
    gen_kwargs_pos=dict(params=["shape", "color"], irrel_params=["material"],
                        clu_num=2, obj_quantities="s", pin=False),
    gen_kwargs_neg=dict(params=["shape", "color"], irrel_params=["material"],
                        clu_num=2, obj_quantities="s", pin=False),
)

# ============================================================
# 2. SIMILARITY
# ============================================================
from scripts.clevr3d.adapt_2d_to_3d import similarity_adapted

render_samples(
    "similarity",
    similarity_adapted,
    gen_kwargs_pos=dict(params=["shape", "color"], irrel_params=["material"],
                        clu_num=2, obj_quantities="s", pin=False),
    gen_kwargs_neg=dict(params=["shape", "color"], irrel_params=["material"],
                        clu_num=2, obj_quantities="s", pin=False),
)

# ============================================================
# 3. CLOSURE — triangle
# ============================================================
from scripts.clevr3d.adapt_2d_to_3d import closure_triangle_adapted

render_samples(
    "closure_triangle",
    closure_triangle_adapted,
    gen_kwargs_pos=dict(params=["shape", "color"], irrel_params=["material"],
                        clu_num=1, obj_quantities="m", pin=False),
    gen_kwargs_neg=dict(params=["shape", "color"], irrel_params=["material"],
                        clu_num=1, obj_quantities="m", pin=False),
)

# ============================================================
# 4. CONTINUITY — two splines
# ============================================================
from scripts.clevr3d.adapt_2d_to_3d import continuity_adapted

render_samples(
    "continuity",
    continuity_adapted,
    gen_kwargs_pos=dict(params=["shape", "color"], irrel_params=["material"],
                        clu_num=2, obj_quantities="s", pin=False),
    gen_kwargs_neg=dict(params=["shape", "color"], irrel_params=["material"],
                        clu_num=2, obj_quantities="s", pin=False),
)

# ============================================================
# 5. SYMMETRY — bilateral axis
# ============================================================
from scripts.clevr3d.adapt_2d_to_3d import symmetry_adapted

render_samples(
    "symmetry",
    symmetry_adapted,
    gen_kwargs_pos=dict(params=["shape", "color"], irrel_params=["material"],
                        clu_num=1, obj_quantities="s", pin=False),
    gen_kwargs_neg=dict(params=["shape", "color"], irrel_params=["material"],
                        clu_num=1, obj_quantities="s", pin=False),
)

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")
total_ok = 0
total_expected = 0
for key, status in results.items():
    ok = int(status.split("/")[0])
    total_ok += ok
    total_expected += NUM_SAMPLES_PER_SPLIT
    icon = "OK" if ok == NUM_SAMPLES_PER_SPLIT else "PARTIAL" if ok > 0 else "FAIL"
    print(f"  [{icon}] {key}: {status}")

print(f"\n  Total: {total_ok}/{total_expected} renders succeeded")
print(f"\nOutput directory: {OUT_DIR}/")

for principle_dir in sorted(OUT_DIR.iterdir()):
    if not principle_dir.is_dir():
        continue
    for split_dir in sorted(principle_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        for label_dir in sorted(split_dir.iterdir()):
            if label_dir.is_dir():
                pngs = list(label_dir.glob("*.png"))
                print(f"  {label_dir.relative_to(OUT_DIR)}/  ({len(pngs)} images)")
