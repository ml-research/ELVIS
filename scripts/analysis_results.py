# Created by jing at 03.03.25
import argparse
import os
from scripts import config
import ace_tools_open as tools
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import torch
import seaborn as sns
from pathlib import Path
import re


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

    # Define task categories
    path = config.results

    df_list = []
    category_f1_scores = {"vit_3": pd.Series(dtype=float),
                          "vit_100": pd.Series(dtype=float),
                          "llava": pd.Series(dtype=float),
                          "NEUMANN": pd.Series(dtype=float)}

    for principle, principle_csv_files in csv_files.items():
        tmp_df = pd.read_csv(principle_csv_files[0], index_col=0)  # Load CSV and set first column as index (model names)
        for file in principle_csv_files:
            df = pd.read_csv(file, index_col=0)  # Load CSV and set first column as index (model names)

            def categorize_task(task_name):
                for category in categories:
                    if category in task_name:
                        return category

            if "vit_3" in file.name:
                model_name = "vit_3"
            elif "vit_100" in file.name:
                model_name = "vit_100"
            elif "llava" in file.name:
                model_name = "llava"
            else:
                model_name = "NEUMANN"
                df = df.reindex(tmp_df.index, fill_value=0)
            categories = config.categories[principle]
            df["Category"] = df.index.map(categorize_task)
            category_avg_f1 = df.groupby("Category")["F1 Score"].mean()
            category_f1_scores[model_name] = pd.concat(
                [category_f1_scores[model_name], category_avg_f1])  # Store results
    # Convert dictionary to DataFrame for heatmap
    heatmap_data = pd.DataFrame(category_f1_scores)

    # Adjust figure size dynamically based on the number of columns
    plt.figure(figsize=(max(15, len(heatmap_data.columns) * 1.5), 4))  # Auto-scale width
    ax = sns.heatmap(heatmap_data.T, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.8,
                     cbar_kws={'label': 'F1 Score'})
    # sns.heatmap(heatmap_data.T, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.8, cbar_kws={'label': 'F1 Score'})
    # Add second level of labels (Gestalt principles)
    ax2 = ax.twiny()  # Create a secondary x-axis

    counts = 0
    principle_pos = []
    principle_names = []
    # Compute column splits based on Gestalt principles
    for principle, categories in gestalt_principles.items():
        principle_names.append(principle)
        principle_pos.append(counts + len(categories) / 2)
        if principle == "continuity": continue
        # **Add vertical separators for Gestalt principles**
        column_positions = counts  # Start at the first column
        col_names = list(heatmap_data.columns)

        # Draw vertical lines between sections
        pos = int(counts + len(categories))
        ax.axvline(pos, color='black', linestyle='dashed', linewidth=1.5)

        # **Label each Gestalt principle in the center of its area**

        # plt.xticks(pos, list(gestalt_principles.keys()), fontsize=12, rotation=30)
        counts += len(categories)
    # Increase the font size of x ticks below the chart
    # Increase the font size of x-ticks below the chart
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
    heat_map_filename = path / f"f1_heat_map.pdf"
    plt.savefig(heat_map_filename, format="pdf", bbox_inches="tight")  # Ensures no extra space

    print(f"Heatmap saved to: {heat_map_filename}")


def json_to_csv(json_data, principle, csv_file_path):
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


def json_to_csv_llava(json_data, principle, csv_file_path):
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


def analysis_models(principles, model_names):
    csv_files = {}
    for principle in principles:
        csv_files[principle] = []
        path = config.results / principle

        for model_name in model_names:
            json_path = path / f"{model_name}.json"
            data = json.load(open(json_path, "r"))
            csv_file_name = path / f"{model_name}_f1_scores.csv"
            if not os.path.exists(csv_file_name):
                if model_name == "llava":
                    df, f1_score = json_to_csv_llava(data, principle, csv_file_name)
                else:
                    df, f1_score = json_to_csv(data, principle, csv_file_name)
                draw_bar_chart(df, f1_score, model_name, path, principle)
            csv_files[principle].append(csv_file_name)

        json_path = config.results / "experiment_results_NEUMANN.json"
        data = json.load(open(json_path, "r"))
        csv_file_name = path / f"NEUMANN_f1_scores.csv"
        if not os.path.exists(csv_file_name):
            df, f1_score = json_to_csv_NEUMANN(data, principle, csv_file_name)
            # draw_bar_chart(df, f1_score, "NEUMANN", path, principle)
        csv_files[principle].append(csv_file_name)

    model_names.append("NEUMANN")
    draw_f1_heat_map(csv_files, model_names, config.categories)


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

def draw_category_subfigures(results, name, save_path=None):
    models = list(results.keys())
    categories = list(next(iter(results.values())).keys())
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
    if model_name == "vit":
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


def analysis_per_category(json_path, principle, category_name=None):
    # load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    per_task_data = data[principle]

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

    print(f"Number of tasks in category '{category_name}': {len(category_acc)}")

    print(f"\tAccuracy: {avg_acc:.3f} ± {std_acc:.3f}")
    print(f"\tF1 Score: {avg_f1:.3f} ± {std_f1:.3f}")
    print(f"\tPrecision: {avg_precision:.3f} ± {std_precision:.3f}")
    print(f"\tRecall: {avg_recall:.3f} ± {std_recall:.3f}")
    print(f"\n")


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
    per_task_data = data[principle]
    return per_task_data


# def analysis_grp_num(json_path, principle):
#     per_task_data = get_per_task_data(json_path, principle)
#     group_size_analysis = {}
#     for task_name, task_res in per_task_data.items():
#         task_info = parse_task_name(task_name)
#         if task_info["group_num"] not in group_size_analysis:
#             group_size_analysis[task_info["group_num"]] = []
#         group_size_analysis[task_info["group_num"]].append(per_task_data[task_name]["accuracy"])
#     grp_size_avg_acc = {k: np.mean(v) for k, v in group_size_analysis.items()}
#     grp_size_std_acc = {k: np.std(v) for k, v in group_size_analysis.items()}
#     print(f"Group Size Analysis:")
#     for grp_size, avg_acc in grp_size_avg_acc.items():
#         std_acc = grp_size_std_acc[grp_size]
#         print(f"\tGroup Size {grp_size}: Avg Accuracy = {avg_acc:.3f} ± {std_acc:.3f}")
#

def analysis_ablation_performance(json_path, principle, prop):
    per_task_data = get_per_task_data(json_path, principle)
    group_size_analysis = {}
    for task_name, task_res in per_task_data.items():
        task_info = parse_task_name(task_name)
        if prop not in task_info:
            continue
        if task_info[prop] not in group_size_analysis:
            group_size_analysis[task_info[prop]] = []
        group_size_analysis[task_info[prop]].append(per_task_data[task_name]["accuracy"])
    logic_avg_acc = {k: np.mean(v) for k, v in group_size_analysis.items()}
    grp_size_std_acc = {k: np.std(v) for k, v in group_size_analysis.items()}
    for name, avg_acc in logic_avg_acc.items():
        std_acc = grp_size_std_acc[name]
        print(f"\t{prop} {name}: Avg Accuracy = {avg_acc:.3f} ± {std_acc:.3f}")


def get_results_path(remote=False, principle=None, model_name=None, img_num=None):
    if remote:
        results_path = Path("/elvis_result/")
    else:
        results_path = config.result_path

    prin_path = results_path / args.principle

    # get all the json file start with vit_
    if model_name == "vit":
        all_json_files = list(prin_path.glob(f"{model_name}_*.json"))
        all_json_files = [f for f in all_json_files if f"img_num_{img_num}" in f.name]

    elif model_name == "internVL":
        all_json_files = list(prin_path.glob(f"{model_name}_*.json"))
        # all_json_files = [f for f in all_json_files if f"img_num_{img_num}" in f.name]

    elif model_name == "llava":
        all_json_files = list(prin_path.glob(f"{model_name}.json"))
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # get the latest json file with the latest timestamp
    if all_json_files:
        latest_json_file = max(all_json_files, key=os.path.getmtime)
        json_path = latest_json_file
    else:
        raise FileNotFoundError(f"No JSON files found for model {model_name} in principle {principle}")
    return json_path


if __name__ == "__main__":
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
        if args.principle == "proximity":
            analysis_per_category(json_path, args.principle, "red_triangle")
            analysis_per_category(json_path, args.principle, "grid")
            analysis_per_category(json_path, args.principle, "fixed_props")
            analysis_per_category(json_path, args.principle, "circle_features")
        elif args.principle == "similarity":
            analysis_per_category(json_path, args.principle, "fixed_number")
            analysis_per_category(json_path, args.principle, "pacman")
            analysis_per_category(json_path, args.principle, "palette")
        elif args.principle == "closure":
            analysis_per_category(json_path, args.principle, "big_triangle")
            analysis_per_category(json_path, args.principle, "big_square")
            analysis_per_category(json_path, args.principle, "big_circle")
            analysis_per_category(json_path, args.principle, "feature_triangle")
            analysis_per_category(json_path, args.principle, "feature_square")
            analysis_per_category(json_path, args.principle, "feature_circle")
        elif args.principle == "symmetry":
            analysis_per_category(json_path, args.principle, "solar_system")
            analysis_per_category(json_path, args.principle, "symmetry_circle")
            analysis_per_category(json_path, args.principle, "symmetry_pattern")
        elif args.principle == "continuity":
            analysis_per_category(json_path, args.principle, "one_split_n")
            analysis_per_category(json_path, args.principle, "with_intersected_n_splines")
            analysis_per_category(json_path, args.principle, "non_intersected_n_splines")
            analysis_per_category(json_path, args.principle, "continuity_overlap_splines")
        else:
            raise ValueError(f"Unsupported principle for category analysis: {args.principle}")
    elif args.mode == "group_num":
        analysis_ablation_performance(json_path, args.principle, "group_num")
    elif args.mode == "grp_size":
        analysis_ablation_performance(json_path, args.principle, "group_size")
    elif args.mode == "rule_type":
        analysis_ablation_performance(json_path, args.principle, "rule_type")
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")
    # analysis_model_category_performance(['shape', 'color', 'count', "size"], "prop")
    # analysis_model_category_performance(["_s", "_m", "_l"], "size")
    # analysis_model_category_performance(["exist", "all"], "exist")
    # analysis_model_category_performance(["_1", "_2", "_3", "_4", "_5"], "group_num")
    # analysis_model_category_performance(["exist", "all"], "exist")
    # print_category_accuracy(data)
    # analysis_result(principles, data)
