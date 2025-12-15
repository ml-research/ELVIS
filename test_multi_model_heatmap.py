#!/usr/bin/env python3
"""
Quick test to verify the multi-model heatmap generation logic
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Simulate data for 5 models
model_names = ["ViT", "LLaVA", "InternVL3-2B", "InternVL3-78B", "GPT-4o"]

fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)
factor_labels = ['Shape', 'Color', 'Size']
upper_mask = np.triu(np.ones((3, 3), dtype=bool), k=1)

for idx, model_name in enumerate(model_names):
    ax = axes[idx]

    # Simulate some data (lower triangle only)
    mean_combo = np.random.uniform(0.3, 0.8, (3, 3))
    mean_combo = np.tril(mean_combo)  # Keep only lower triangle
    mean_combo[np.triu_indices(3, k=1)] = np.nan

    mean_df = pd.DataFrame(mean_combo, index=factor_labels, columns=factor_labels)

    # Plot heatmap
    sns.heatmap(mean_df, annot=True, fmt=".3f", cmap="RdBu_r",
               vmin=0.3, vmax=0.8, ax=ax,
               cbar=(idx == 4),  # Only show colorbar on last subplot
               cbar_kws={'label': 'Mean F1 Score'} if idx == 4 else None,
               linewidths=0.8, linecolor='black', mask=upper_mask,
               annot_kws={'size': 12})

    ax.set_title(model_name, fontsize=16, fontweight='bold', pad=10)
    ax.set_xlabel("Factor", fontsize=11)
    if idx == 0:
        ax.set_ylabel("Factor", fontsize=11)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    if idx == 0:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    # Add sample counts
    total_counts = np.random.randint(50, 200, (3, 3))
    for i in range(3):
        for j in range(3):
            if i >= j:
                n = int(total_counts[i, j])
                ax.text(j + 0.5, i + 0.7, f"n={n}", ha='center', va='top',
                       fontsize=7, color='dimgray')

plt.suptitle("Mean Pairwise Factor Combination Performance Across All Principles",
            fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()

output_path = "/Users/jing/PycharmProjects/ELVIS/test_multi_model_output.pdf"
plt.savefig(output_path, format="pdf", bbox_inches="tight")
print(f"✓ Test figure saved to: {output_path}")
plt.close()

