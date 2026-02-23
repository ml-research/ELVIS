# CLEVR-3D Gestalt Scene Generation: Unified Strategy Documentation

**Purpose:** This document explains how the 3D CLEVR scene generation pipeline (`scripts/clevr3d/`) reuses the **same conceptual strategy** as the original 2D generation pipeline (`scripts/proximity/`, `scripts/similarity/`, etc.) for placing objects and assigning groups. The 3D pipeline is a faithful translation of the 2D logic into a Blender-rendered 3D world.

---

## Table of Contents

1. [High-Level Architecture Comparison](#1-high-level-architecture-comparison)
2. [Shared Generation Strategy (Overview)](#2-shared-generation-strategy)
3. [Per-Principle Breakdown (with Visual Examples)](#3-per-principle-breakdown)
   - 3.1 Proximity
   - 3.2 Similarity
   - 3.3 Closure
   - 3.4 Continuity
   - 3.5 Symmetry
4. [Object Encoding: 2D vs 3D Fields](#4-object-encoding)
5. [Position Generation: Coordinate Translation](#5-position-generation)
6. [Negative Example Strategy](#6-negative-example-strategy)
7. [Scene JSON Structure](#7-scene-json-structure)
8. [Summary of Equivalences](#8-summary)

---

## 1. High-Level Architecture Comparison

The two pipelines have a mirrored directory structure by design:

```
2D Pipeline (Original)                    3D Pipeline (New)
========================                  ========================
scripts/proximity/                        scripts/clevr3d/principles/proximity_3d.py
  util_red_triangle.py
  util_fixed_props.py
  util_grid_objs.py

scripts/similarity/                       scripts/clevr3d/principles/similarity_3d.py
  util_fixed_number.py
  util_palette.py

scripts/closure/                          scripts/clevr3d/principles/closure_3d.py
  util_pos_triangle.py
  util_pos_square.py
  util_pos_circle.py

scripts/continuity/                       scripts/clevr3d/principles/continuity_3d.py
  util_two_splines.py
  util_a_splines.py

scripts/symmetry/                         scripts/clevr3d/principles/symmetry_3d.py
  util_symmetry_bilateral.py
  util_solar_system.py

scripts/utils/encode_utils.py             scripts/clevr3d/encode_utils_3d.py
scripts/config.py                         scripts/clevr3d/config_3d.py
```

**Key insight:** Each 3D principle file contains the same function signatures and logic flow as its 2D counterpart, but outputs 3D coordinates and adds CLEVR-specific properties (material, 3D size).

---

## 2. Shared Generation Strategy

Both pipelines follow the **exact same 4-step strategy** for every Gestalt principle:

```
+-------------------------------------------------------------------+
|  STEP 1: GENERATE POSITIONS (spatial arrangement)                 |
|    2D: (x, y) in [0,1] normalized canvas                         |
|    3D: (x, y, z) in [-5,5] x [-5,5] ground plane, z = radius    |
+-------------------------------------------------------------------+
           |
           v
+-------------------------------------------------------------------+
|  STEP 2: ASSIGN PROPERTIES per group (shape, color, size, ...)    |
|    - fixed_props: same value within group (defines grouping)      |
|    - irrel_params: same value across ALL objects (noise control)   |
|    - random: different random values per object (visual variety)   |
|    2D: shape, color, size                                         |
|    3D: shape, color, size, MATERIAL (new groupable property)      |
+-------------------------------------------------------------------+
           |
           v
+-------------------------------------------------------------------+
|  STEP 3: ENCODE OBJECTS into scene dict                           |
|    2D: encode_utils.encode_scene(positions, sizes, colors, ...)   |
|    3D: encode_utils_3d.encode_scene_3d(positions, sizes, ...)     |
|    3D also projects to 2D for backward-compatible x/y/size fields |
+-------------------------------------------------------------------+
           |
           v
+-------------------------------------------------------------------+
|  STEP 4: VALIDATE (no overlaps, all in bounds)                    |
|    2D: scripts/utils/shape_utils.overlaps()                       |
|    3D: scripts/clevr3d/shape_utils_3d.validate_scene_no_overlap() |
+-------------------------------------------------------------------+
```

### The Positive/Negative Logic is Identical

Both pipelines use the same `fixed_props`, `irrel_params`, `cf_params` (counterfactual) system:

| Parameter | Positive Example | Negative Example |
|-----------|-----------------|------------------|
| `fixed_props` | Group-defining properties are **consistent** within each group | These properties are **violated** (randomized) |
| `irrel_params` | Held constant across all objects | Held constant across all objects |
| `cf_params` | N/A | Subset of properties that are **kept consistent** even in negative (to create specific counterfactual) |
| Spatial structure | Follows the Gestalt principle | Either broken or preserved (depending on what's being tested) |

This logic is **literally the same code pattern** in both pipelines.

---

## 3. Per-Principle Breakdown

### 3.1 Proximity

**Core idea (both 2D and 3D):** Objects in the same group are placed **close together** in tight clusters, separated from other clusters by a large gap.

#### 2D Implementation (`scripts/proximity/util_red_triangle.py`)

```python
# Generate cluster centers far apart
# Place objects tightly around each cluster center
# group_id assigned per cluster
```

- Canvas: normalized [0,1] x [0,1]
- Cluster radius: `config.get_grp_r()` ~ 0.08-0.23
- Object size: ~0.05

#### 3D Implementation (`scripts/clevr3d/principles/proximity_3d.py`)

```python
# Same logic:
# 1. generate_cluster_center() - place cluster centers far apart
# 2. generate_cluster_positions() - pack objects tightly in each cluster
# 3. Assign properties using identical fixed_props/irrel_params logic
# 4. encode_scene_3d() - encode with 3D + backward-compatible 2D fields
```

- Ground plane: [-5,5] x [-5,5]
- Cluster radius: `CLUSTER_RADIUS` = 1.8-3.0 (scaled to 3D world)
- Object radius: 0.35 (CLEVR "small")
- Cluster separation: `CLUSTER_DIST` = 5.0

#### Visual Comparison

**3D Positive (Proximity)** - Two tight clusters, clearly separated:

```
    [gen_data_3d/test_renders/proximity/train/positive/00000.png]

    Objects form two distinct spatial clusters:
    - Left cluster: 5 purple spheres packed together
    - Right cluster: 5 cyan cylinders packed together
    Each cluster = one group (group_id 0 and 1)
```

**3D Negative (Proximity)** - Objects scattered randomly, no clear clusters:

```
    [gen_data_3d/test_renders/proximity/train/negative/00000.png]

    Objects of mixed shapes/colors spread uniformly across the scene.
    No perceivable spatial clusters.
```

**Bounding box visualization confirming groups:**

```
    [gen_data_3d/test_renders_bbox/proximity/train/positive/00000.png]

    Blue boxes = group 0 (cyan cylinders, right cluster)
    Orange boxes = group 1 (purple spheres, left cluster)
```

---

### 3.2 Similarity

**Core idea (both 2D and 3D):** Objects are grouped by **shared visual properties** (shape, color, etc.), NOT by spatial position. Positions are random.

#### 2D Implementation (`scripts/similarity/util_fixed_number.py`)

```python
# Scatter all objects randomly
# Group defined by shared properties (e.g., same shape + same color)
# Positive: groups have consistent properties
# Negative: properties are randomized across objects
```

#### 3D Implementation (`scripts/clevr3d/principles/similarity_3d.py`)

```python
def similarity_fixed_number_3d(fixed_props, irrel_params, cf_params, clu_num,
                                is_positive, obj_quantities):
    # SAME LOGIC:
    # 1. Generate random positions for ALL objects (no clustering!)
    positions = generate_random_positions(total_objects, base_radius)

    # 2. Assign properties per group - IDENTICAL logic to 2D
    for g_i in range(clu_num):
        for _ in range(group_size):
            for prop in ["shape", "color", "size", "material"]:
                if prop in irrel_params:
                    val = irrel_values[prop]           # same for all
                elif is_positive and prop in fixed_props:
                    val = group_values[prop][g_i]      # same within group
                elif not is_positive and prop in cf_params:
                    val = group_values[prop][g_i]      # kept in negative
                else:
                    val = random.choice(PROPERTY_POOLS[prop])  # random
```

Note: The 3D version adds `"material"` as a new groupable property (metal vs rubber), which has no 2D equivalent. This is the **only new feature** -- the strategy is unchanged.

#### Visual Comparison

**3D Positive (Similarity)** - Same type objects scattered, grouped by property:

```
    [gen_data_3d/test_renders/similarity/train/positive/00000.png]

    Group 0: purple cubes (scattered across scene)
    Group 1: green cylinders (scattered across scene)
    Grouped by shared color+shape, NOT by spatial proximity
```

**3D Negative (Similarity)** - No consistent property grouping:

```
    [gen_data_3d/test_renders/similarity/train/negative/00000.png]

    Mixed shapes, colors, and materials.
    No two-or-more objects consistently share properties.
```

---

### 3.3 Closure

**Core idea (both 2D and 3D):** Objects are placed along the edges of a geometric shape outline (triangle, square, circle), forming a "closed" shape with gaps at corners.

#### 2D Implementation (`scripts/closure/util_pos_triangle.py`, etc.)

```python
# Place objects along triangle/square/circle edges
# Leave gaps at corners (the "closure" effect)
# All objects share properties (same group)
```

#### 3D Implementation (`scripts/clevr3d/principles/closure_3d.py`)

Uses `pos_utils_3d.generate_closure_positions()` which implements the **same edge-distribution logic**:

```python
def generate_closure_positions(shape_type, center, size, num_objects, obj_radius=0.35):
    if shape_type == "triangle":
        # Distribute objects across 3 edges
        per_edge_base = max(num_objects // 3, 1)
        # Place along edge with gap_frac at corners
        for edge_i in range(3):
            for j in range(n_on_edge):
                t = gap_frac + usable_frac * (j / max(n_on_edge - 1, 1))
                x = v1[0] + t * (v2[0] - v1[0])
                y = v1[1] + t * (v2[1] - v1[1])
```

This is **the same parametric edge interpolation** as the 2D version, just with 3D coordinates.

#### Visual Comparison

**3D Positive (Closure - Triangle)** - Objects form a triangle outline:

```
    [gen_data_3d/test_renders/closure_triangle/train/positive/00000.png]

    9 brown cubes arranged along the edges of a triangle.
    Gaps visible at the three corners.
    All objects in group 0 (single group forming the shape).
```

**3D Negative (Closure)** - Objects scattered randomly:

```
    [gen_data_3d/test_renders/closure_triangle/train/negative/00000.png]

    Objects of mixed types placed randomly.
    No perceivable geometric outline.
```

**Bounding box visualization:**

```
    [gen_data_3d/test_renders_bbox/closure_triangle/train/positive/00000.png]

    All blue boxes (group 0) - single group forming the triangle outline.
```

---

### 3.4 Continuity

**Core idea (both 2D and 3D):** Objects are placed along smooth curves (splines), creating a "continuation" pattern.

#### 2D Implementation (`scripts/continuity/util_two_splines.py`)

```python
# Generate random control points
# Fit spline through control points
# Sample object positions along the spline
# Each spline = one group
```

#### 3D Implementation (`scripts/clevr3d/principles/continuity_3d.py`)

```python
def continuity_splines_3d(...):
    for spline_i in range(num_splines):
        if is_positive or ("continuity" in cf_params):
            # Generate spline control points - SAME LOGIC
            control_points = _generate_control_points(num_control, z_range=(...))
            # Sample positions along spline - SAME LOGIC
            raw_positions = generate_spline_3d(control_points, objects_per_spline * 3)
            # Sub-sample evenly - SAME LOGIC
            candidates = raw_positions[::step]
            positions = _filter_positions_no_overlap(candidates, ...)
        else:
            # Negative: random scatter - SAME LOGIC
            positions = generate_random_positions(...)
```

The `generate_spline_3d()` function uses `scipy.interpolate.make_interp_spline` -- the same approach as 2D, extended to 3D coordinates.

#### Visual Comparison

**3D Positive (Continuity)** - Two interleaving curves:

```
    [gen_data_3d/test_renders/continuity/train/positive/00000.png]

    Group 0: 5 gray spheres arranged along a smooth curve
    Group 1: 5 red cylinders arranged along another smooth curve
    The two curves weave through the scene.
```

**3D Negative (Continuity)** - No curve structure:

```
    [gen_data_3d/test_renders/continuity/train/negative/00000.png]

    Objects of mixed types placed randomly.
    No perceivable curves or continuation paths.
```

---

### 3.5 Symmetry

**Core idea (both 2D and 3D):** Objects are arranged with either bilateral (mirror) or rotational symmetry.

#### 2D Implementation (`scripts/symmetry/util_symmetry_bilateral.py`)

```python
# Bilateral: mirror positions across an axis
# Rotational: place arms at equal angular intervals from center
# Paired/corresponding objects share properties
```

#### 3D Implementation (`scripts/clevr3d/principles/symmetry_3d.py`)

**Bilateral symmetry:**
```python
def symmetry_bilateral_3d(fixed_props, irrel_params, cf_params, is_positive, obj_quantities):
    # SAME LOGIC as 2D:
    if is_positive or ("symmetry" in cf_params):
        positions = generate_symmetric_positions_bilateral(
            center, num_per_side, spread=3.5, axis=axis, obj_radius=base_radius
        )
    else:
        # Break symmetry: random positions
        positions = generate_random_positions(total, base_radius)

    # Paired objects share properties - SAME as 2D
    for i in range(total):
        pair_idx = i // 2
        if is_positive and prop in fixed_props:
            if i % 2 == 0:
                val = random.choice(PROPERTY_POOLS[prop])  # generate for first
            else:
                val = _pair_cache[(pair_idx, prop)]         # reuse for mirror
```

**Rotational symmetry:**
```python
def symmetry_rotational_3d(...):
    # SAME LOGIC as 2D rotational_symmetry_pattern:
    # Objects arranged in n-fold rotational arms
    # Corresponding objects in each arm share properties
    positions = generate_symmetric_positions_rotational(
        center, group_size, radius=3.0, n_fold=n_fold, obj_radius=base_radius
    )
```

#### Visual Comparison

**3D Positive (Symmetry)** - Bilaterally symmetric arrangement:

```
    [gen_data_3d/test_renders/symmetry/train/positive/00000.png]

    10 blue spheres arranged in bilateral symmetry.
    Mirror pairs visible: left-right correspondence.
    All objects in group 0 (single symmetric group).
```

**3D Negative (Symmetry)** - No symmetry:

```
    [gen_data_3d/test_renders/symmetry/train/negative/00000.png]

    Objects of mixed types placed randomly.
    No perceivable symmetry axis.
```

---

## 4. Object Encoding: 2D vs 3D Fields

The 3D encoder (`encode_utils_3d.py`) produces objects with **all 2D fields preserved** plus additional 3D fields:

```
2D Fields (backward compatible):       3D Fields (new):
================================       ================
x          (projected 2D x)            x_3d        (world x)
y          (projected 2D y)            y_3d        (world y)
size       (projected 2D size)         z_3d        (world z)
color_name                             size_3d     ("small"/"large")
color_r, color_g, color_b              radius_3d   (0.35 or 0.7)
shape                                  material    ("metal"/"rubber")
line_width (-1)                        rotation    (z-axis degrees)
solid      (True)                      render_backend ("blender")
start_angle (0)
end_angle   (360)
group_id
```

The 2D fields are computed via **camera projection** from the 3D coordinates, so the scene JSON is compatible with both the 2D analysis pipeline and the 3D renderer.

---

## 5. Position Generation: Coordinate Translation

| Concept | 2D | 3D |
|---------|----|----|
| Canvas | [0,1] x [0,1] | [-5,5] x [-5,5] ground plane |
| Object size | ~0.05 radius | 0.35 or 0.7 radius |
| Overlap check | `shape_utils.overlaps()` | `shape_utils_3d.check_overlap_3d()` |
| Cluster radius | 0.08-0.23 | 1.8-3.0 |
| Cluster separation | ~0.3 | 5.0 |
| Z coordinate | N/A | Fixed at `obj_radius` (on ground) |
| Random position | `random.uniform(0,1)` | `random.uniform(-5,5)` |
| Spline fitting | scipy B-spline (2D) | scipy B-spline (3D, z fixed) |

The **ratios are preserved**: cluster radius / canvas size, object size / cluster radius, inter-cluster distance / cluster radius are all proportionally equivalent.

---

## 6. Negative Example Strategy

Both pipelines use the **identical counterfactual strategy**:

```
            Positive Example                    Negative Example
            ================                    ================

Proximity:  Objects clustered together    -->   Objects scattered randomly
            Same props within cluster           (spatial structure broken)

Similarity: Shared properties per group   -->   Properties randomized
            Random positions                    (property grouping broken)

Closure:    Objects form shape outline     -->   Objects placed randomly
            Same props on outline               (geometric structure broken)

Continuity: Objects along smooth curves    -->   Objects scattered randomly
            Each curve = one group              (curve structure broken)

Symmetry:   Mirror/rotational symmetry    -->   Random positions
            Paired objects match                (symmetry broken)
```

The `cf_params` (counterfactual parameters) system allows **fine-grained negative control**: you can break ONLY the spatial structure while keeping properties consistent, or break ONLY properties while keeping spatial structure. This is **the same mechanism in both pipelines**.

---

## 7. Scene JSON Structure

Both 2D and 3D produce the same JSON format:

```json
{
  "objects": [
    {
      "x": 0.716,             // 2D projected x (backward compatible)
      "y": 0.434,             // 2D projected y
      "size": 0.055,          // 2D projected size
      "color_name": "cyan",
      "color_r": 41, "color_g": 208, "color_b": 208,
      "shape": "cylinder",
      "group_id": 0,          // GROUP ASSIGNMENT (same meaning in both)

      // --- 3D-specific fields (new, absent in 2D) ---
      "x_3d": 0.207,
      "y_3d": 3.423,
      "z_3d": 0.35,
      "size_3d": "small",
      "radius_3d": 0.35,
      "material": "metal",
      "rotation": 196.05,
      "render_backend": "blender"
    }
  ],
  "logics": {
    "rule": "group_target(X):-has_shape(X,...),principle(proximity).",
    "is_positive": true,
    "fixed_props": ["shape", "color"],
    "cf_params": ["shape", "proximity"],
    "irrel_params": ["material"],
    "principle": "proximity"
  }
}
```

The `logics` block uses the **same Prolog-style rule format** as the 2D pipeline.

---

## 8. Summary of Equivalences

| Aspect | 2D Pipeline | 3D Pipeline | Same Strategy? |
|--------|-------------|-------------|----------------|
| **Proximity** grouping | Tight 2D clusters | Tight 3D clusters on ground plane | YES |
| **Similarity** grouping | Random positions, shared props | Random positions, shared props (+material) | YES |
| **Closure** shapes | Edge interpolation (triangle/square/circle) | Same edge interpolation in 3D | YES |
| **Continuity** curves | B-spline through control points | B-spline through 3D control points | YES |
| **Symmetry** (bilateral) | Mirror across axis | Mirror across axis on ground plane | YES |
| **Symmetry** (rotational) | Arms at equal angles | Arms at equal angles on ground plane | YES |
| Positive/Negative logic | fixed_props + cf_params | fixed_props + cf_params | IDENTICAL |
| Object encoding | `encode_scene()` | `encode_scene_3d()` (superset) | COMPATIBLE |
| Overlap validation | 2D circle overlap | 3D sphere overlap | EQUIVALENT |
| Config (quantity dict) | `{"s":5, "m":8, "l":12, "xl":15}` | **Reuses 2D config directly** | IDENTICAL |
| Group ID meaning | Cluster/property group index | Cluster/property group index | IDENTICAL |

### Bottom Line

The 3D pipeline is a **direct translation** of the 2D pipeline into CLEVR-style 3D scenes. The generation strategy -- how positions are computed, how groups are assigned, how positive/negative examples are constructed -- is **the same algorithm** operating in a different coordinate system and rendered with Blender instead of matplotlib.

The only additions in 3D are:
1. **Material** as a new groupable property (metal vs rubber)
2. **Camera projection** to generate backward-compatible 2D coordinates
3. **Blender rendering** instead of 2D shape drawing

---

## Visual Reference Sheets

The following pre-generated review sheets (in `gen_data_3d/review_sheets/`) show side-by-side positive vs negative examples for each principle, with bounding boxes overlaid:

- `all_principles_overview.png` - All 5 principles at a glance
- `proximity_comparison.png` - Proximity positive (green border) vs negative (red border)
- `similarity_comparison.png` - Similarity positive vs negative
- `closure_triangle_comparison.png` - Closure positive vs negative
- `continuity_comparison.png` - Continuity positive vs negative
- `symmetry_comparison.png` - Symmetry positive vs negative

These sheets confirm visually that the 3D patterns match the expected Gestalt grouping behavior from the 2D pipeline.
