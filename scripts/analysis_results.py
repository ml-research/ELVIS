# Created by jing at 03.03.25
import argparse
import os
from scripts import config
import ace_tools_open as tools
import json
import pandas as pd
import scipy.stats as stats
import torch
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib

model_dict = {
    "vit_base_patch16_224": {"model": "vit", "img_num": 3},
    # "vit_base_patch16_224/100": {"model": "vit", "img_num": 100},
    "llava-onevision-qwen2-7b": {"model": "llava", "img_num": 3},
    "InternVL3-2B": {"model": "internVL3_2B", "img_num": 3},
    "InternVL3-78B": {"model": "internVL3_78B", "img_num": 3},
    "GPT-5": {"model": "gpt5", "img_num": 3},

}

principles = ["proximity", "similarity", "closure", "symmetry", "continuity"]


def draw_line_chart(x, y, xlabel, ylabel, title, save_path=None, msg=""):
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, marker='o', linestyle='-')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True)
    if msg:
        plt.text(0.05, 0.8, msg, fontsize=12, ha='left', va='center', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


def analysis_llava(principle, model_name):
    path = config.results / principle
    json_path = path / f"{model_name}_evaluation_results.json"
    data = json.load(open(json_path, "r"))
    accs = torch.tensor([v["accuracy"] / 100 for k, v in data.items()])
    f1s = torch.tensor([v["f1_score"] for k, v in data.items()])
    # Convert JSON to DataFrame

    # Calculate performance statistics
    mean_accuracy = accs.mean()
    mean_f1 = f1s.mean()
    std_accuracy = accs.std()
    std_f1 = f1s.std()

    # Compute confidence intervals
    confidence = 0.95
    n = len(accs)
    ci_accuracy = stats.t.interval(confidence, n - 1, loc=mean_accuracy, scale=std_accuracy / np.sqrt(n))
    ci_f1 = stats.t.interval(confidence, n - 1, loc=mean_f1, scale=std_f1 / np.sqrt(n))

    # Redraw the bar chart without std lines and save as PDF
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Define colors
    accuracy_color = "#1f77b4"  # Blue
    f1_color = "#ff7f0e"  # Orange

    # Plot accuracy distribution with mean
    axes[0].hist(accs, bins=10, edgecolor="black", alpha=0.7, color=accuracy_color)
    axes[0].axvline(mean_accuracy, color='red', linestyle='dashed', linewidth=1, label=f"Mean: {mean_accuracy:.3f}")
    axes[0].set_title(f"Accuracy Distribution ({principle})", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Accuracy", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].legend()

    # Plot F1-score distribution with mean
    axes[1].hist(f1s, bins=10, edgecolor="black", alpha=0.7, color=f1_color)
    axes[1].axvline(mean_f1, color='blue', linestyle='dashed', linewidth=1, label=f"Mean: {mean_f1:.3f}")
    axes[1].set_title(f"F1 Score Distribution ({principle})", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("F1 Score", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].legend()

    # Save the figure as PDF
    pdf_filename = f"{principle}_{model_name}_model_performance_charts.pdf"
    plt.tight_layout()
    plt.savefig(path / pdf_filename, format="pdf")

    # Format confidence intervals using ± notation
    accuracy_conf_interval = (mean_accuracy - ci_accuracy[0], ci_accuracy[1] - mean_accuracy)
    f1_conf_interval = (mean_f1 - ci_f1[0], ci_f1[1] - mean_f1)

    # Create a formatted table
    formatted_performance_table = pd.DataFrame({
        "Metric": ["Accuracy", "F1 Score"],
        "Mean ± Std": [f"{mean_accuracy:.3f} ± {std_accuracy:.3f}", f"{mean_f1:.3f} ± {std_f1:.3f}"],
        "95% Confidence Interval": [f"{mean_accuracy:.3f} ± {accuracy_conf_interval[0]:.3f}",
                                    f"{mean_f1:.3f} ± {f1_conf_interval[0]:.3f}"]
    })

    # Display the formatted table
    tools.display_dataframe_to_user(name=f"Formatted Performance Table {principle}",
                                    dataframe=formatted_performance_table)


# def draw_f1_heat_map(csv_files, model_names, gestalt_principles):
#     """
#     Draws and saves a heatmap showing F1 scores of models across tasks,
#     grouping tasks into 5 areas corresponding to different Gestalt principles.
#
#     Args:
#         csv_files (dict): Dictionary where keys are Gestalt principles and values are lists of CSV files.
#         model_names (list): List of model names.
#         gestalt_principles (dict): Dictionary mapping Gestalt principles to column ranges.
#     """
#
#     path = config.results
#     category_f1_scores = {model: pd.Series(dtype=float) for model in model_names}
#
#     for principle, principle_csv_files in csv_files.items():
#         for file in principle_csv_files:
#             df = pd.read_csv(file, index_col=0)  # Load CSV and set first column as index (model names)
#
#             # Categorize tasks
#             def categorize_task(task_name):
#                 for category in config.categories[principle]:
#                     if category in task_name:
#                         return category
#
#             # Identify model name from file name
#             if "vit_3" in file.name:
#                 model_name = "vit_3"
#             elif "vit_100" in file.name:
#                 model_name = "vit_100"
#             else:
#                 model_name = "llava"
#
#             df["Category"] = df.index.map(categorize_task)
#             category_avg_f1 = df.groupby("Category")["F1 Score"].mean()
#             category_f1_scores[model_name] = pd.concat([category_f1_scores[model_name], category_avg_f1])
#
#     # Convert dictionary to DataFrame for heatmap
#     heatmap_data = pd.DataFrame(category_f1_scores)
#
#     # Plot the heatmap
#     plt.figure(figsize=(12, 6))
#     ax = sns.heatmap(heatmap_data.T, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.8, cbar_kws={'label': 'F1 Score'})
#
#     # **Add vertical separators for Gestalt principles**
#     column_positions = [0]  # Start at the first column
#     col_names = list(heatmap_data.columns)
#
#     # Compute column splits based on Gestalt principles
#     for principle, categories in gestalt_principles.items():
#         end_col_idx = max([col_names.index(cat) for cat in categories if cat in col_names], default=0)
#         column_positions.append(end_col_idx + 1)
#
#     # Draw vertical lines between sections
#     for pos in column_positions[1:-1]:  # Avoid the last position
#         ax.axvline(pos, color='black', linestyle='dashed', linewidth=1.5)
#
#     # **Label each Gestalt principle in the center of its area**
#     mid_positions = [(column_positions[i] + column_positions[i+1]) / 2 for i in range(len(column_positions)-1)]
#     plt.xticks(mid_positions, list(gestalt_principles.keys()), fontsize=12, rotation=30)
#
#     # Labels
#     plt.xlabel("Gestalt Principles", fontsize=12)
#     plt.ylabel("Models", fontsize=12)
#
#     # Save the heatmap as a PDF file
#     heat_map_filename = path / "f1_heat_map.pdf"
#     plt.tight_layout()
#     plt.savefig(heat_map_filename, format="pdf", bbox_inches="tight")
#
#     print(f"Heatmap saved to: {heat_map_filename}")


def draw_f1_heat_map(csv_files, model_names, gestalt_principles):
    # Function to categorize tasks
    category_acc_scores = {
        "vit_base_patch16_224": pd.Series(dtype=float),
        # "vit_base_patch16_224": pd.Series(dtype=float),
        "llava-onevision-qwen2-7b": pd.Series(dtype=float),
        "InternVL3-2B": pd.Series(dtype=float),
        "InternVL3-78B": pd.Series(dtype=float),
        "GPT-5": pd.Series(dtype=float),
        # "llava-onevision-qwen2-7b-si-hf/3": pd.Series(dtype=float)
    }

    for principle, principle_csv_files in csv_files.items():
        tmp_df = pd.read_csv(principle_csv_files[0], index_col=0)  # Load CSV and set first column as index (model names)
        for file in principle_csv_files:
            df = pd.read_csv(file, index_col=0)  # Load CSV and set first column as index (model names)

            def categorize_task(task_name):
                for category in categories:
                    if category in task_name:
                        return category

            if "vit_3" in file.name:
                model_name = "vit_base_patch16_224"
            elif "llava" in file.name:
                model_name = "llava-onevision-qwen2-7b"
            elif "internVL3_78B" in file.name:
                model_name = "InternVL3-78B"
            elif "internVL3_2B" in file.name:
                model_name = "InternVL3-2B"
            elif "gpt5" in file.name:
                model_name = "GPT-5"
            else:
                raise ValueError("Unknown model in file name:", file.name)
            # df = df.reindex(tmp_df.index, fill_value=0)
            categories = config.categories[principle]
            df["Category"] = df.index.map(categorize_task)
            category_avg_f1 = df.groupby("Category")["F1 Score"].mean()

            # iterative the category_avg_f1, check the name if it is in the config.name_map, then replace it with the value
            if "symmetry" in str(file):
                for cat in category_avg_f1.index:
                    if cat in config.name_map:
                        new_name = config.name_map[cat]
                        category_avg_f1 = category_avg_f1.rename(index={cat: new_name})

            category_acc_scores[model_name] = pd.concat([category_acc_scores[model_name], category_avg_f1])  # Store results
    # Convert dictionary to DataFrame for heatmap
    heatmap_data = pd.DataFrame(category_acc_scores)

    # Adjust figure size dynamically based on the number of columns
    plt.figure(figsize=(max(15, len(heatmap_data.columns) * 1.5), 4))  # Auto-scale width
    ax = sns.heatmap(heatmap_data.T, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.8,
                     cbar_kws={'label': 'F1 Score'}
                     )
    ax2 = ax.twiny()  # Create a secondary x-axis
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)  # Increase y-axis label font size

    counts = 0
    principle_pos = []
    principle_names = []
    # Compute column splits based on Gestalt principles
    for principle, categories in gestalt_principles.items():
        principle_names.append(principle)
        principle_pos.append(counts + len(categories) / 2)
        if principle == "continuity": continue
        # Draw vertical lines between sections
        pos = int(counts + len(categories))
        ax.axvline(pos, color='black', linestyle='dashed', linewidth=1.5)
        counts += len(categories)
    # Increase the font size of x ticks below the chart
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16, rotation=30, ha="right")

    ax2.set_xticks(principle_pos)
    ax2.set_xticklabels(principle_names, rotation=0, fontsize=12, fontweight="bold")
    ax2.set_xlim(ax.get_xlim())  # Align with main x-axis
    # ax2.set_xlabel("Gestalt Principles", fontsize=12, fontweight="bold")

    # plt.xlabel("Category", fontsize=12)
    plt.ylabel("Models", fontsize=12)

    # Adjust the layout to remove extra space
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.2)  # Reduce right margin

    # Save the heatmap
    heat_map_filename = config.figure_path / f"f1_heat_map.pdf"
    plt.savefig(heat_map_filename, format="pdf", bbox_inches="tight")  # Ensures no extra space

    print(f"Heatmap saved to: {heat_map_filename}")


def json_to_csv(json_data, csv_file_path):
    # Convert JSON to DataFrame
    df = pd.DataFrame(json_data).T
    df["accuracy"] /= 100  # Convert accuracy to percentage
    # Calculate performance statistics
    # mean_accuracy = df["accuracy"].mean()
    precision = df["precision"].values
    recall = df["recall"].values
    f1_score = 2 * (precision * recall) / ((precision + recall) + 1e-20)
    # Save F1-score to CSV with row names (index)
    f1_score_df = pd.DataFrame({"F1 Score": f1_score}, index=df.index)
    f1_score_df.to_csv(csv_file_path, index=True)
    print(f"F1-score data saved to {csv_file_path}")
    return df, f1_score


def json_to_csv_NEUMANN(json_data, principle, csv_file_path):
    # Convert JSON to DataFrame
    df = pd.DataFrame(json_data[principle]).T
    df["accuracy"] /= 100  # Convert accuracy to percentage
    # Calculate performance statistics
    # mean_accuracy = df["accuracy"].mean()
    precision = df["precision"].values
    recall = df["recall"].values
    f1_score = 2 * (precision * recall) / ((precision + recall) + 1e-20)
    # Save F1-score to CSV with row names (index)
    f1_score_df = pd.DataFrame({"F1 Score": f1_score}, index=df.index)
    f1_score_df.to_csv(csv_file_path, index=True)
    print(f"F1-score data saved to {csv_file_path}")
    return df, f1_score


def json_to_csv_llava(json_data, csv_file_path):
    f1_score = torch.tensor([v["f1_score"] for k, v in json_data.items()])
    # Convert JSON to DataFrame
    # Remove the "logic_rules" field from each entry
    for key in json_data.keys():
        if "logic_rules" in json_data[key]:
            del json_data[key]["logic_rules"]

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(json_data, orient="index")

    df["accuracy"] /= 100  # Convert accuracy to percentage
    # Calculate performance statistics
    # mean_accuracy = df["accuracy"].mean()
    precision = df["precision"].values
    recall = df["recall"].values
    f1_score = 2 * (precision * recall) / ((precision + recall) + 1e-20)
    # Save F1-score to CSV with row names (index)
    f1_score_df = pd.DataFrame({"F1 Score": f1_score}, index=df.index)
    f1_score_df.to_csv(csv_file_path, index=True)
    print(f"F1-score data saved to {csv_file_path}")
    return df, f1_score


def draw_bar_chart(df, f1_score, model_name, path, principle):
    f1_score = np.nan_to_num(f1_score)
    mean_f1 = f1_score.mean()
    mean_accuracy = df["accuracy"].mean()
    mean_precision = df["precision"].mean()
    mean_recall = df["recall"].mean()

    std_accuracy = df["accuracy"].std()
    std_f1 = f1_score.std()
    std_precision = df["precision"].std()
    std_recall = df["recall"].std()

    # Redraw the bar chart without std lines and save as PDF
    fig, axes = plt.subplots(4, 1, figsize=(12, 20))

    # Define colors
    accuracy_color = "#1f77b4"  # Blue
    f1_color = "#ff7f0e"  # Orange
    precision_color = "#2ca02c"  # Green
    recall_color = "#d62728"  # Red

    # Plot accuracy distribution with mean
    axes[0].hist(df["accuracy"], bins=10, edgecolor="black", alpha=0.7, color=accuracy_color)
    axes[0].axvline(mean_accuracy, color='red', linestyle='dashed', linewidth=1, label=f"Mean: {mean_accuracy:.2f}")
    axes[0].set_title(f"Accuracy Distribution ({principle})", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Accuracy", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].legend()

    # Plot F1-score distribution with mean
    axes[1].hist(df["f1_score"], bins=10, edgecolor="black", alpha=0.7, color=f1_color)
    axes[1].axvline(mean_f1, color='blue', linestyle='dashed', linewidth=1, label=f"Mean: {mean_f1:.2f}")
    axes[1].set_title(f"F1 Score Distribution ({principle})", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("F1 Score", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].legend()

    # Plot Precision distribution with mean
    axes[2].hist(df["precision"], bins=10, edgecolor="black", alpha=0.7, color=precision_color)
    axes[2].axvline(mean_precision, color='black', linestyle='dashed', linewidth=1, label=f"Mean: {mean_precision:.2f}")
    axes[2].set_title(f"Precision Distribution ({principle})", fontsize=14, fontweight='bold')
    axes[2].set_xlabel("Precision", fontsize=12)
    axes[2].set_ylabel("Frequency", fontsize=12)
    axes[2].legend()

    # Plot Recall distribution with mean
    axes[3].hist(df["recall"], bins=10, edgecolor="black", alpha=0.7, color=recall_color)
    axes[3].axvline(mean_recall, color='purple', linestyle='dashed', linewidth=1, label=f"Mean: {mean_recall:.2f}")
    axes[3].set_title(f"Recall Distribution ({principle})", fontsize=14, fontweight='bold')
    axes[3].set_xlabel("Recall", fontsize=12)
    axes[3].set_ylabel("Frequency", fontsize=12)
    axes[3].legend()

    # Save the figure as PDF
    pdf_filename = f"{model_name}_model_performance_charts.pdf"
    plt.tight_layout()
    plt.savefig(path / pdf_filename, format="pdf")

    # Create Line Chart for Precision & Recall without Standard Deviation
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    x = np.arange(len(df))  # X-axis index

    axes[0].plot(x, df["precision"], label="Precision", color=precision_color)
    axes[0].plot(x, df["recall"], label="Recall", color=recall_color)
    axes[0].set_title(f"Precision & Recall Over Samples ({principle})", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Sample Index", fontsize=12)
    axes[0].set_ylabel("Score", fontsize=12)
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Plot Precision - Recall Difference
    diff = np.abs(df["precision"] - df["recall"])
    axes[1].plot(x, diff, label="Precision - Recall", color='black')
    axes[1].set_title(f"Precision - Recall Difference ({principle})", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Sample Index", fontsize=12)
    axes[1].set_ylabel("Difference", fontsize=12)
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # Save the performance line chart as PDF
    line_chart_filename = f"performance_line_chart.pdf"
    plt.tight_layout()
    plt.savefig(path / line_chart_filename, format="pdf")

    # Create a formatted table
    formatted_performance_table = pd.DataFrame({
        "Metric": ["Accuracy", "F1 Score", "Precision", "Recall"],
        "Mean ± Std": [
            f"{mean_accuracy:.2f} ± {std_accuracy:.2f}",
            f"{mean_f1:.2f} ± {std_f1:.2f}",
            f"{mean_precision:.2f} ± {std_precision:.2f}",
            f"{mean_recall:.2f} ± {std_recall:.2f}"
        ]
    })

    # Display the formatted table
    tools.display_dataframe_to_user(name=f"Formatted Performance Table {principle}",
                                    dataframe=formatted_performance_table)


def analysis_models(args):
    csv_files = {}
    save_path = config.figure_path / f"all_category_all_principles_heat_map.pdf"
    for principle in principles:
        csv_files[principle] = []
        # path = config.results / principle
        for model_name, model_info in model_dict.items():
            json_path = get_results_path(args.remote, principle, model_info["model"], model_info["img_num"])
            per_task_data = get_per_task_data(json_path, principle)
            # replace the soloar with solar if exists in the keys of the per_task_data
            per_task_data = {re.sub(r"soloar", "solar", k): v for k, v in per_task_data.items()}
            # replace intersected_n_splines in the keys with with_intersected_n_splines of the per_task_data if intersected_n_splines in the keys but non_intersected_n_splines not in the keys
            # per_task_data = {re.sub(r"intersected_n_splines", "with_intersected_n_splines", k): v for k, v in per_task_data.items()}
            new_per_task_data = {}
            for k, v in per_task_data.items():
                if "non_intersected_n_splines" in k:
                    new_per_task_data[k] = v
                elif "intersected_n_splines" in k:
                    new_key = k.replace("intersected_n_splines", "with_intersected_n_splines")
                    new_per_task_data[new_key] = v
                else:
                    new_per_task_data[k] = v
            per_task_data = new_per_task_data
            csv_file_name = config.result_path / principle / f"{model_info['model']}_{model_info['img_num']}.csv"
            if model_name == "llava":
                df, f1_score = json_to_csv_llava(per_task_data, csv_file_name)
            else:
                df, f1_score = json_to_csv(per_task_data, csv_file_name)

            csv_files[principle].append(csv_file_name)
    draw_f1_heat_map(csv_files, list(model_dict.keys()), config.categories)


def get_category_accuracy(task_results, categories=None):
    if categories is None:
        categories = ['shape', 'color', 'count', '_s_', '_m_', '_l_',
                      "non_overlap_red_triangle", "non_overlap_grid",
                      "non_overlap_fixed_props", "overlap_big_small",
                      "overlap_circle_features"]
    acc_results = {}
    for cat in categories:
        acc_results[cat] = np.mean([v['accuracy'] for k, v in task_results.items() if cat in k])
    return acc_results


def print_category_accuracy(task_results, categories=None):
    """
    Calculate and print mean and std accuracy for each category keyword in task names.
    Args:
        task_results: dict, key is task_name, value is a dict with at least 'accuracy'
        categories: list of category keywords to check (default: ['shape', 'color', 'count', 's', 'm', 'l'])
    """
    import numpy as np
    if categories is None:
        categories = ['shape', 'color', 'count', '_s_', '_m_', '_l_',
                      "non_overlap_red_triangle", "non_overlap_grid",
                      "non_overlap_fixed_props", "overlap_big_small",
                      "overlap_circle_features"]
    for cat in categories:
        accs = [v['accuracy'] for k, v in task_results.items() if cat in k]
        if accs:
            print(f"{cat}: mean accuracy = {np.mean(accs):.3f}, std = {np.std(accs):.3f}, n = {len(accs)}")
        else:
            print(f"{cat}: no tasks found")


def analysis_result(principles, data):
    accuracies = [v['accuracy'] for v in data.values()]
    f1_scores = [v['f1_score'] for v in data.values()]

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)

    print(f"Accuracy: mean={mean_acc:.3f}, std={std_acc:.3f}")
    print(f"F1 Score: mean={mean_f1:.3f}, std={std_f1:.3f}")


# def draw_category_subfigures(results, save_path=None):
#     models = list(results.keys())
#     categories = list(next(iter(results.values())).keys())
#     n_cats = len(categories)
#     n_cols = min(6, n_cats)
#     n_rows = int(np.ceil(n_cats / n_cols))
#
#     fig_width = 15
#     fig_height = 2.5 * n_rows
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
#
#     palette = sns.color_palette("Set2", n_colors=len(models))
#
#     for idx, cat in enumerate(categories):
#         row, col = divmod(idx, n_cols)
#         ax = axes[row][col]
#         values = [results[model].get(cat, np.nan) for model in models]
#         bars = ax.bar(range(len(models)), values, color=palette)
#         ax.set_title(cat, fontsize=10, fontweight='bold', pad=6)
#         ax.set_ylim(0, 100)
#         ax.set_ylabel('Acc.', fontsize=8)
#         ax.yaxis.set_label_coords(-0.08, 0.5)
#         ax.set_xticks(range(len(models)))
#         ax.set_xticklabels(models, fontsize=8, rotation=20)
#         ax.tick_params(axis='y', labelsize=8)
#         yticks = ax.get_yticks()
#         ax.set_yticks(yticks[::2])
#         for i, v in enumerate(values):
#             ax.text(i, v + 2, f"{v:.1f}", ha='center', va='bottom', fontsize=7)
#         ax.margins(y=0.15)
#         # Remove top and right spines
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         # Add 50% dashed line
#         ax.axhline(50, color='gray', linestyle='--', linewidth=1)
#     # Hide unused subplots
#     for idx in range(n_cats, n_rows * n_cols):
#         row, col = divmod(idx, n_cols)
#         fig.delaxes(axes[row][col])
#     fig.suptitle(f'Mean Accuracy by Category', fontsize=12, fontweight='bold', y=0.99)
#     plt.tight_layout(rect=[0, 0, 1, 0.97], pad=1.0, h_pad=1.2, w_pad=1.2)
#     if save_path:
#         plt.savefig(save_path, format="pdf", bbox_inches="tight")
#     plt.show()


def draw_grouped_categories(results, name, x_label, save_path=None):
    models = list(results.keys())
    categories = list(next(iter(results.values())).keys())
    n_models = len(models)
    n_categories = len(categories)

    bar_width = 0.15
    x = np.arange(n_categories)
    palette = sns.color_palette("Set2", n_colors=n_models)

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    for i, model in enumerate(models):
        values = [results[model].get(cat, np.nan) for cat in categories]
        ax.bar(x + i * bar_width, values, width=bar_width, color=palette[i], label=model)
        for j, v in enumerate(values):
            ax.text(x[j] + i * bar_width, v + 2, f"{v:.1f}", ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(categories, fontsize=10, rotation=30, ha='right')
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel(f"{x_label}", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title(f'{name}', fontsize=14, fontweight='bold')
    ax.legend(title="Model", fontsize=10)
    ax.axhline(50, color='gray', linestyle='--', linewidth=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


def draw_category_subfigures(results, name, save_path=None):
    models = list(results.keys())

    categories = list(next(iter(results.values())).keys())
    # categories = sort_categories(categories)
    n_models = len(models)
    n_cols = min(4, n_models)
    n_rows = int(np.ceil(n_models / n_cols))

    fig_width = 4 * n_cols
    fig_height = 3 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)

    palette = sns.color_palette("Set2", n_colors=len(categories))

    for idx, model in enumerate(models):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        values = [results[model].get(cat, np.nan) for cat in categories]
        bars = ax.bar(range(len(categories)), values, color=palette)
        ax.set_title(model, fontsize=11, fontweight='bold', pad=6)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Acc.', fontsize=9)
        ax.yaxis.set_label_coords(-0.08, 0.5)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, fontsize=8, rotation=30, ha='right')
        ax.tick_params(axis='y', labelsize=8)
        yticks = ax.get_yticks()
        ax.set_yticks(yticks[::2])
        for i, v in enumerate(values):
            ax.text(i, v + 2, f"{v:.1f}", ha='center', va='bottom', fontsize=7)
        ax.margins(y=0.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axhline(50, color='gray', linestyle='--', linewidth=1)
    # Hide unused subplots
    for idx in range(n_models, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        fig.delaxes(axes[row][col])
    fig.suptitle(f'Mean {name} Accuracy by Model', fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97], pad=1.0, h_pad=1.2, w_pad=1.2)
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


def load_json_data(principle, file_name, base_categories=None):
    with open(file_name, 'r') as f:
        data = json.load(f)
    if principle not in data:
        data.pop("average", None)  # Remove 'average' if it exists
    else:
        data = data[principle]  # Extract the specific principle data
    cate_acc = get_category_accuracy(data, base_categories)
    return cate_acc


def load_json_model_data(model_name, base_categories):
    all_data = {}
    for principle in config.categories.keys():
        json_path = config.result_path / principle / f"{model_name}.json"
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                if principle not in data:
                    data.pop("average", None)  # Remove 'average' if it exists
                else:
                    data = data[principle]  # Extract the specific principle data
                all_data.update(data)
    category_acc = get_category_accuracy(all_data, base_categories)

    return category_acc


def analysis_per_principle_model_category_performance():
    prox_categories = ["non_overlap_red_triangle", "non_overlap_grid",
                       "non_overlap_fixed_props", "overlap_big_small",
                       "overlap_circle_features"]

    closure_categories = ["non_overlap_big_triangle", "non_overlap_big_square",
                          "non_overlap_big_circle", "non_overlap_feature_triangle", "non_overlap_feature_square",
                          "non_overlap_feature_circle"]

    vit_prox = load_json_data("proximity", config.result_path / "proximity" / "vit_base_patch16_224_3_evaluation_results_20250728_162145.json")

    vit_3_closure = load_json_data("closure", config.result_path / "closure" / "vit_3.json")
    vit_100_closure = load_json_data("closure", config.result_path / "closure" / "vit_100.json")
    llava_closure = load_json_data("closure", config.result_path / "closure" / "llava.json")

    vit_3_symmetry = load_json_data("symmetry", config.result_path / "symmetry" / "vit_3.json")
    vit_100_symmetry = load_json_data("symmetry", config.result_path / "symmetry" / "vit_100.json")
    llava_symmetry = load_json_data("symmetry", config.result_path / "symmetry" / "llava.json")
    # vit_3_closure_acc = get_category_accuracy(vit_3_closure, base_categories)
    # vit_100_closure_acc = get_category_accuracy(vit_100_closure, base_categories)
    # llava_closure_acc = get_category_accuracy(llava_closure, base_categories)
    closure_results = {"vit_3": vit_3_closure, "vit_100": vit_100_closure, "llava": llava_closure}
    symmetry_results = {"vit_3": vit_3_symmetry, "vit_100": vit_100_symmetry, "llava": llava_symmetry}

    # Call the function
    # draw_category_subfigures(closure_results, "closure", save_path=config.figure_path / f"closure_category_accuracy.pdf")
    # draw_category_subfigures(results, "proximity", save_path=config.figure_path / f"proximity_category_accuracy.pdf")
    # draw_category_subfigures(results, "proximity", save_path=config.figure_path / f"proximity_category_accuracy.pdf")
    # draw_category_subfigures(results, "proximity", save_path=config.figure_path / f"proximity_category_accuracy.pdf")
    # draw_category_subfigures(results, "proximity", save_path=config.figure_path / f"proximity_category_accuracy.pdf")


def analysis_model_category_performance(categories, name):
    vit_3_data = load_json_model_data("vit_3", categories)
    vit_100_data = load_json_model_data("vit_100", categories)
    llava_data = load_json_model_data("llava", categories)
    model_results = {"vit_3": vit_3_data,
                     "vit_100": vit_100_data,
                     "llava": llava_data}
    draw_category_subfigures(model_results, name, save_path=config.figure_path / f"model_category_accuracy_{name}.pdf")


def analysis_average_performance(json_path, principle, model_name, img_num):
    # load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    if "vit" in model_name:
        per_task_data = data[principle]
    else:
        per_task_data = data
        per_task_data.pop("average", None)  # Remove 'average' if it exists

    avg_acc = np.mean([v['accuracy'] for v in per_task_data.values()])
    avg_f1 = np.mean([v['f1_score'] for v in per_task_data.values()])
    avg_precision = np.mean([v['precision'] for v in per_task_data.values()])
    avg_recall = np.mean([v['recall'] for v in per_task_data.values()])

    std_acc = np.std([v['accuracy'] for v in per_task_data.values()])
    std_f1 = np.std([v['f1_score'] for v in per_task_data.values()])
    std_precision = np.std([v['precision'] for v in per_task_data.values()])
    std_recall = np.std([v['recall'] for v in per_task_data.values()])

    msg = (f"{principle}\n"
           f"#task: {len(per_task_data)}\n"
           f"#Img: {img_num}\n"
           f"Acc: {avg_acc:.2f} ± {std_acc:.2f}\n"
           f"F1: {avg_f1:.2f} ± {std_f1:.2f}\n"
           f"Prec: {avg_precision:.2f} ± {std_precision:.2f}\n"
           f"Recall: {avg_recall:.2f} ± {std_recall:.2f}")
    # draw the performance as a line chart
    x = list(range(1, len(per_task_data) + 1))
    y = [v['accuracy'] for v in per_task_data.values()]
    x_label = "Task Index"
    y_label = "Accuracy"
    title = f"{model_name}_{img_num} Accuracy for each task in {principle}"
    save_path = config.figure_path / f"{principle}_acc_{model_name}_{img_num}.pdf"
    draw_line_chart(x, y, x_label, y_label, title, save_path, msg)


def analysis_per_category(args, category_names):
    all_results = {}
    for model_name, model_info in model_dict.items():
        model_results = {}
        for category_name in category_names:
            json_path = get_results_path(args.remote, args.principle, model_info["model"], model_info["img_num"])
            per_task_data = get_per_task_data(json_path, args.principle)

            category_acc = [v["accuracy"] for k, v in per_task_data.items() if category_name in k]
            category_f1 = [v["f1_score"] for k, v in per_task_data.items() if category_name in k]
            category_precision = [v["precision"] for k, v in per_task_data.items() if category_name in k]
            category_recall = [v["recall"] for k, v in per_task_data.items() if category_name in k]

            avg_acc = np.mean(category_acc)
            avg_f1 = np.mean(category_f1)
            avg_precision = np.mean(category_precision)
            avg_recall = np.mean(category_recall)

            std_acc = np.std(category_acc)
            std_f1 = np.std(category_f1)
            std_precision = np.std(category_precision)
            std_recall = np.std(category_recall)
            model_results[category_name] = avg_acc
            print(f"Number of tasks in category '{category_name}': {len(category_acc)}")
            print(f"\tAccuracy: {avg_acc:.3f} ± {std_acc:.3f}")
            print(f"\tF1 Score: {avg_f1:.3f} ± {std_f1:.3f}")
            print(f"\tPrecision: {avg_precision:.3f} ± {std_precision:.3f}")
            print(f"\tRecall: {avg_recall:.3f} ± {std_recall:.3f}")
            print(f"\n")
        all_results[model_name] = model_results
    draw_category_subfigures(all_results, f"{args.principle}_{category_name}", save_path=config.figure_path / f"{args.principle}_{category_name}_category_accuracy.pdf")


def parse_task_name(task_name):
    info = {}
    # Related concepts after 'rel_'
    rel_match = re.search(r'rel_([a-zA-Z_]+)', task_name)
    if rel_match:
        info['related_concepts'] = rel_match.group(1).split('_')[:-1]

    # Group number (number after related concepts)
    group_num_match = re.search(r'rel_[a-zA-Z_]+_(\d+)', task_name)
    if group_num_match:
        info['group_num'] = int(group_num_match.group(1))
    else:
        if "grid" in task_name:
            info['group_num'] = 4
        else:
            raise ValueError(f"Task name '{task_name}' does not contain a valid group number.")
    # Group size (s, m, l, xl after group number)
    size_match = re.search(r'rel_[a-zA-Z_]+_\d+_(s|m|l|xl)', task_name)
    if size_match:
        info['group_size'] = size_match.group(1)

    # Irrelated concepts after 'irrel_'
    irrel_match = re.search(r'irrel_([a-zA-Z_]+)', task_name)
    if irrel_match:
        info['irrelated_concepts'] = irrel_match.group(1).split('_')[:-1]

    # Rule type at the end ('all' or 'exist')
    rule_match = re.search(r'(all|exist)$', task_name)
    if rule_match:
        info['rule_type'] = rule_match.group(1)

    return info


def plot_principle_concept_heatmaps(results_dict, principles, concept_sets,
                                    cmap="viridis", vmin=40, vmax=100, figsize=(20, 6)):
    """
    Plot heatmaps of accuracy across Gestalt principles and concept sets for multiple models.

    Args:
        results_dict (dict): Dictionary mapping model names to 2D accuracy arrays
                             of shape (n_principles, n_concepts).
                             Example: {"ViT": acc_vit, "LLaVA": acc_llava, ...}
        principles (list of str): List of Gestalt principles (row labels).
        concept_sets (list of str): List of concept sets (column labels).
        cmap (str): Colormap for heatmap.
        vmin, vmax (float): Color scale limits.
        figsize (tuple): Size of the entire figure (width, height).
    """
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=figsize, sharey=True)

    if n_models == 1:
        axes = [axes]  # ensure iterable

    for ax, (model, acc_matrix) in zip(axes, results_dict.items()):
        df = pd.DataFrame(acc_matrix, index=principles, columns=concept_sets)
        sns.heatmap(df, annot=True, fmt=".1f", cmap=cmap, cbar=True,
                    vmin=vmin, vmax=vmax, ax=ax, annot_kws={"size": 7})
        ax.set_title(model, fontsize=12)
        ax.set_xlabel("Concepts")
        if ax == axes[0]:
            ax.set_ylabel("Principles")
        else:
            ax.set_ylabel("")

    plt.tight_layout()
    plt.show()


def get_per_task_data(json_path, principle):
    # load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    if principle in data:
        return data[principle]
    else:
        if "average" in data:
            data.pop("average", None)
        return data


# def analysis_ablation_performance(args, prop):
#     json_paths = {}
#     model_results = {}
#     all_principles = ["proximity", "similarity", "closure", "symmetry", "continuity"]
#     for principle in all_principles:
#         for model_name, model_info in model_dict.items():
#             json_paths[model_name] = get_results_path(args, args.remote, args.principle, model_info["model"], model_info["img_num"])
#             per_task_data = get_per_task_data(json_paths[model_name], principle)
#             group_size_analysis = {}
#             for task_name, task_res in per_task_data.items():
#                 task_info = parse_task_name(task_name)
#                 if prop not in task_info:
#                     continue
#                 if task_info[prop] not in group_size_analysis:
#                     group_size_analysis[task_info[prop]] = []
#                 group_size_analysis[task_info[prop]].append(per_task_data[task_name]["accuracy"])
#             logic_avg_acc = {k: np.mean(v) for k, v in group_size_analysis.items()}
#             grp_size_std_acc = {k: np.std(v) for k, v in group_size_analysis.items()}
#             for name, avg_acc in logic_avg_acc.items():
#                 std_acc = grp_size_std_acc[name]
#                 print(f"\t{prop} {name}: Avg Accuracy = {avg_acc:.3f} ± {std_acc:.3f}")
#             model_results[model_name] = logic_avg_acc
#
#     draw_grouped_categories(model_results, f"{principle}", "#Group",
#                             save_path=config.figure_path / f"{principle}_{prop}_ablation_accuracy.pdf")
#

# def analysis_ablation_performance(args):
#
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import seaborn as sns
#
#     props = ["group_num", "group_size", "rule_type"]
#     all_principles = ["proximity", "similarity", "closure", "symmetry", "continuity"]
#     model_names = list(model_dict.keys())
#     results = {}
#     principle_categories = {}
#
#     # Collect results for each principle and property value
#     for principle in all_principles:
#         results[principle] = {}
#         categories = set()
#         for model_name, model_info in model_dict.items():
#             json_path = get_results_path(args.remote, principle, model_info["model"], model_info["img_num"])
#             per_task_data = get_per_task_data(json_path, principle)
#             group_analysis = {}
#             for task_name, task_res in per_task_data.items():
#                 task_info = parse_task_name(task_name)
#                 if prop not in task_info:
#                     continue
#                 value = task_info[prop]
#                 categories.add(value)
#                 if value not in group_analysis:
#                     group_analysis[value] = []
#                 group_analysis[value].append(task_res["accuracy"])
#             avg_acc = {k: np.mean(v) for k, v in group_analysis.items()}
#             results[principle][model_name] = avg_acc
#         principle_categories[principle] = sorted(list(categories))
#
#     n_principles = len(all_principles)
#     n_models = len(model_names)
#     fig, axes = plt.subplots(1, n_principles, figsize=(25, 5), sharey=True)
#     palette = sns.color_palette("Set2", n_colors=n_models)
#
#     bar_width = 0.10  # Keep bar width
#     group_gap = 0.05  # Small gap between groups
#
#     for idx, principle in enumerate(all_principles):
#         ax = axes[idx]
#         categories = principle_categories[principle]
#         x = np.arange(len(categories))
#         n = n_models
#         # Calculate total width of each group (bars + gap)
#         group_width = n * bar_width + group_gap
#         for i, model_name in enumerate(model_names):
#             values = [results[principle][model_name].get(cat, np.nan) for cat in categories]
#             # Position bars tightly within each group
#             ax.bar(x * group_width + i * bar_width, values, width=bar_width, color=palette[i], label=model_name)
#             for j, v in enumerate(values):
#                 ax.text(x[j] * group_width + i * bar_width, v + 2, f"{v:.1f}", rotation=60, ha='center', va='bottom', fontsize=8)
#         # Center xticks under each group
#         ax.set_xticks(x * group_width + (n * bar_width) / 2 - bar_width / 2)
#         ax.set_xticklabels(categories, fontsize=25, rotation=30, ha='right')
#         ax.set_xlabel(prop, fontsize=30)
#         ax.set_title(principle, fontsize=30, fontweight='bold')
#         ax.set_ylim(0, 70)
#         # ax.set_yticks(fontsize=25)
#         ax.set_ylabel('Accuracy', fontsize=30)
#         ax.axhline(50, color='gray', linestyle='--', linewidth=1)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         # if idx == 0:
#         #     ax.legend(title="Model", fontsize=9, loc='best')
#     plt.tight_layout()
#     save_path = config.figure_path / f"ablation_{prop}_all_principles_subfigures.pdf"
#     plt.savefig(save_path, format="pdf", bbox_inches="tight")
#     plt.close()
#     print(f"Grouped ablation bar chart (split by principle) saved to: {save_path}")
def analysis_obj_ablation_performance(args):
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"], "ytick.labelsize": 30})
    props = ["color", "shape", "size"]
    all_principles = ["proximity", "similarity", "closure", "symmetry", "continuity"]
    model_names = ["M1", "M2", "M3", "M4", "M5"]
    official_model_names = list(model_dict.keys())
    n_props = len(props)
    n_principles = len(all_principles)
    n_models = len(model_names)
    fig, axes = plt.subplots(len(props), n_principles, figsize=(5 * n_principles, 5 * n_props), sharey='row')
    palette = sns.color_palette("Set2", n_colors=2)
    bar_width = 0.10
    group_gap = 0.05

    for row_idx, prop in enumerate(props):
        results = {}
        principle_categories = {}
        for principle in all_principles:
            results[principle] = {}
            # Collect results for each principle and property value
            categories = set()
            for model_name, model_info in model_dict.items():
                results[principle][model_name] = {"related":[], "irrelated":[]}
                json_path = get_results_path(args.remote, principle, model_info["model"], model_info["img_num"])
                per_task_data = get_per_task_data(json_path, principle)
                rel_acc =[]
                irrel_acc = []

                for task_name, task_res in per_task_data.items():
                    if task_name[-1] != "_":
                        task_name += "_"
                    task_info = parse_task_name(task_name)
                    if prop in task_info["related_concepts"]:
                        rel_acc.append(task_res["accuracy"])
                        results[principle][model_name]["related"].append(task_res["accuracy"])
                    elif prop in task_info["irrelated_concepts"]:
                        irrel_acc.append(task_res["accuracy"])
                        results[principle][model_name]["irrelated"].append(task_res["accuracy"])
                    else:
                        continue
        for col_idx, principle in enumerate(all_principles):
            ax = axes[row_idx, col_idx] if n_props > 1 else axes[col_idx]
            categories = ["related", "irrelated"]
            x = np.arange(len(model_names))
            n = 2
            group_width = n * bar_width + group_gap

            for i, cat in enumerate(categories):
                values = [np.mean(results[principle][model_name][cat]) for model_name in official_model_names]
                ax.bar(x * group_width + i * bar_width, values, width=bar_width, color=palette[i], label=model_names)

                for j, model_name in enumerate(model_names):
                    ax.text(x[j] * group_width + i * bar_width, values[j] + 2, f"{values[j]:.1f}", rotation=60, ha='center', va='bottom', fontsize=10)
            ax.set_xticks(x * group_width + (n * bar_width) / 2 - bar_width / 2)
            ax.set_xticklabels(model_names, fontsize=30, ha='right')
            ax.set_xlabel(prop, fontsize=25)
            if row_idx == 0:
                ax.set_title(principle, fontsize=35, fontweight='bold')
            ax.set_ylim(0, 70)
            if col_idx == 0:
                ax.set_ylabel('Accuracy', fontsize=25)
            ax.axhline(50, color='gray', linestyle='--', linewidth=1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if row_idx == 0 and col_idx == 0:
                handles, labels = ax.get_legend_handles_labels()
    legend_handles = [handles[0], handles[len(model_names)]]
    fig.legend(legend_handles, ["related", "irrelated"], loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=30)
    plt.tight_layout(rect=[0, 0.08, 1, 1])  # leave space for legend at bottom
    save_path = config.figure_path / f"ablation_obj_all_props_all_principles_matrix.pdf"
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()

    # Compute average accuracy for each prop across all principles and models
    avg_results = {prop: [] for prop in props}
    avg_irrel_results = {prop: [] for prop in props}
    for prop in props:
        for model_name in official_model_names:
            rel_accs = []
            irrel_accs = []
            for principle in all_principles:
                rel_accs.extend(results[principle][model_name]["related"])
                irrel_accs.extend(results[principle][model_name]["irrelated"])
            avg_results[prop].append(np.mean(rel_accs) if rel_accs else np.nan)
            avg_irrel_results[prop].append(np.mean(irrel_accs) if irrel_accs else np.nan)

    # Draw grouped bar chart: each subplot is a prop, bars are related/irrelated for each model
    fig, axes = plt.subplots(1, len(props), figsize=(6 * len(props), 6), sharey=True)
    if len(props) == 1:
        axes = [axes]
    bar_width = 0.35
    x = np.arange(len(official_model_names))
    palette = sns.color_palette("Set2", n_colors=2)
    for idx, prop in enumerate(props):
        ax = axes[idx]
        rel_vals = avg_results[prop]
        irrel_vals = avg_irrel_results[prop]
        ax.bar(x - bar_width/2, rel_vals, width=bar_width, color=palette[0], label="related")
        ax.bar(x + bar_width/2, irrel_vals, width=bar_width, color=palette[1], label="irrelated")
        for i, v in enumerate(rel_vals):
            ax.text(x[i] - bar_width/2, v + 2, f"{v:.1f}", ha='center', va='bottom', fontsize=15, rotation=60)
        for i, v in enumerate(irrel_vals):
            ax.text(x[i] + bar_width/2, v + 2, f"{v:.1f}", ha='center', va='bottom', fontsize=15, rotation=60)
        ax.set_title(f"Acc: {prop}", fontsize=30, fontweight='bold')
        if idx == 0:
            ax.set_ylabel("Accuracy", fontsize=25)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=25)
        ax.set_ylim(0, 100)
        ax.axhline(50, color='gray', linestyle='--', linewidth=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=20)
    plt.tight_layout()
    save_path = config.figure_path / "obj_ablation_avg_accuracy_per_prop_grouped.pdf"
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()

def analysis_ablation_performance(args):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import matplotlib

    matplotlib.rcParams.update({
        # "font.size": 24,  # Increase base font size
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        # "axes.titlesize": 32,
        # "axes.labelsize": 28,
        # "xtick.labelsize": 22,
        "ytick.labelsize": 30,
        # "legend.fontsize": 24,
        # "figure.titlesize": 32
    })

    props = ["group_num", "group_size"]
    all_principles = ["proximity", "similarity", "closure", "symmetry", "continuity"]
    model_names = list(model_dict.keys())

    n_props = len(props)
    n_principles = len(all_principles)
    n_models = len(model_names)

    fig, axes = plt.subplots(n_props, n_principles, figsize=(5 * n_principles, 5 * n_props), sharey='row')
    palette = sns.color_palette("Set2", n_colors=n_models)
    bar_width = 0.10
    group_gap = 0.05

    for row_idx, prop in enumerate(props):
        results = {}
        principle_categories = {}
        # Collect results for each principle and property value
        for principle in all_principles:
            results[principle] = {}
            categories = set()
            for model_name, model_info in model_dict.items():
                json_path = get_results_path(args.remote, principle, model_info["model"], model_info["img_num"])
                per_task_data = get_per_task_data(json_path, principle)
                group_analysis = {}
                for task_name, task_res in per_task_data.items():
                    task_info = parse_task_name(task_name)
                    if prop not in task_info:
                        continue
                    value = task_info[prop]
                    categories.add(value)
                    if value not in group_analysis:
                        group_analysis[value] = []
                    group_analysis[value].append(task_res["accuracy"])
                avg_acc = {k: np.mean(v) for k, v in group_analysis.items()}
                results[principle][model_name] = avg_acc
            if "s" in list(categories) and "m" in list(categories) and "l" in list(categories):
                principle_categories[principle] = ["s", "m", "l", "xl"]
            else:
                principle_categories[principle] = sorted(list(categories))
        for col_idx, principle in enumerate(all_principles):
            ax = axes[row_idx, col_idx] if n_props > 1 else axes[col_idx]
            categories = principle_categories[principle]
            x = np.arange(len(categories))
            n = n_models
            group_width = n * bar_width + group_gap
            for i, model_name in enumerate(model_names):
                values = [results[principle][model_name].get(cat, np.nan) for cat in categories]
                ax.bar(x * group_width + i * bar_width, values, width=bar_width, color=palette[i], label=model_name)
                for j, v in enumerate(values):
                    ax.text(x[j] * group_width + i * bar_width, v + 2, f"{v:.1f}", rotation=90, ha='center', va='bottom', fontsize=10)
            ax.set_xticks(x * group_width + (n * bar_width) / 2 - bar_width / 2)
            ax.set_xticklabels(categories, fontsize=30, ha='right')
            ax.set_xlabel(prop, fontsize=25)
            if row_idx == 0:
                ax.set_title(principle, fontsize=35, fontweight='bold')
            ax.set_ylim(0, 100)
            if col_idx == 0:
                ax.set_ylabel('Accuracy', fontsize=25)
            ax.axhline(50, color='gray', linestyle='--', linewidth=1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if row_idx == 0 and col_idx == 0:
                handles, labels = ax.get_legend_handles_labels()
                # ax.legend(title="Model", fontsize=12, loc='best')
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.08), ncol=3,
               fontsize=30, frameon=True)
    plt.tight_layout(rect=[0, 0.08, 1, 1])  # leave space for legend at bottom
    save_path = config.figure_path / f"ablation_all_props_all_principles_matrix.pdf"
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Matrix of grouped ablation bar charts saved to: {save_path}")


def get_results_path(remote=False, principle=None, model_name=None, img_num=None):
    if remote:
        results_path = Path("/elvis_result/")
    else:
        results_path = config.result_path

    prin_path = results_path / principle

    # get all the json file start with vit_
    if model_name == "vit":
        all_json_files = list(prin_path.glob(f"{model_name}_*.json"))
        all_json_files = [f for f in all_json_files if f"img_num_{img_num}" in f.name]

    else:
        all_json_files = list(prin_path.glob(f"{model_name}_*.json"))
        # all_json_files = [f for f in all_json_files if f"img_num_{img_num}" in f.name]
    # elif model_name == "llava":
    #     all_json_files = list(prin_path.glob(f"{model_name}_*.json"))
    # elif "internVL3_78B" in model_name:
    #     all_json_files = list(prin_path.glob(f"{model_name}_*.json"))
    # else:
    #     raise ValueError(f"Unsupported model name: {model_name}")

    # get the latest json file with the latest timestamp
    if all_json_files:
        latest_json_file = max(all_json_files, key=os.path.getmtime)
        json_path = latest_json_file
    else:
        raise FileNotFoundError(f"No JSON files found for model {model_name} in principle {principle}")
    return json_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline models with CUDA support.")
    parser.add_argument("--model", type=str, required=True, help="Specify the principle to filter data.")
    parser.add_argument("--principle", type=str, required=True)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--mode", type=str, default="avg_principle")
    parser.add_argument("--img_num", type=int)
    args = parser.parse_args()

    json_path = get_results_path(args.remote, args.principle, args.model, args.img_num)

    if args.mode == "principle":
        analysis_average_performance(json_path, args.principle, args.model, args.img_num)
    elif args.mode == "category":
        analysis_models(args)
    elif args.mode == "ablation":
        analysis_ablation_performance(args)
    elif args.mode == "obj_ablation":
        analysis_obj_ablation_performance(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
