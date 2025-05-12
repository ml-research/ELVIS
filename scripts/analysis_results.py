# Created by jing at 03.03.25
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
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16,rotation=30, ha="right")

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


if __name__ == "__main__":
    principles = [
        "proximity",
        "similarity",
        "closure",
        "symmetry",
        "continuity"
    ]

    # model_name = "Llava"
    model_names = ["vit_3", "vit_100", "llava"]

    analysis_models(principles, model_names)
