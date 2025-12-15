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
                     cbar_kws={'label': 'F1 Score'}, annot_kws={'size': 20}
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


def parse_task_name_dict(name):
    parts = name.split('_', 2)
    core = parts[2]
    tokens = core.split('_')

    # task family tokens
    task_family_tokens = []
    i = 0
    while i < len(tokens) and tokens[i] not in ("rel", "irrel"):
        task_family_tokens.append(tokens[i])
        i += 1

    task_family = "_".join(task_family_tokens)

    # relevant cues
    relevant = []
    if i < len(tokens) and tokens[i] == "rel":
        i += 1
        while i < len(tokens) and tokens[i] not in ("irrel",) and not tokens[i].isdigit() and tokens[i] not in ("s", "m", "l", "xl"):
            relevant.append(tokens[i])
            i += 1

    # irrelevant cues
    irrelevant = []
    if i < len(tokens) and tokens[i] == "irrel":
        i += 1
        while i < len(tokens) and not tokens[i].isdigit() and tokens[i] not in ("s", "m", "l", "xl"):
            irrelevant.append(tokens[i])
            i += 1

    # group count
    group_count = next((int(t) for t in tokens if t.isdigit()), None)

    # object size
    size = next((t for t in tokens if t in ("s", "m", "l", "xl")), None)

    # rule type is all if all in the tokens else exist
    if "all" in tokens:
        rule_type = "all"
    elif "exist" in tokens:
        rule_type = "exist"
    else:
        rule_type = "unknown"

    name_dict = {
        "task_family": task_family,
        "relevant": relevant,
        "irrelevant": irrelevant,
        "group_count": group_count,
        "size": size,
    }
    if rule_type != "unknown":
        name_dict["rule_type"] = rule_type
    return name_dict


def analysis_tasks(args, return_combo_data=False):
    csv_files = {}
    principle = args.principle

    # Use the model from args
    if args.model not in model_dict:
        raise ValueError(f"Model {args.model} not found in model_dict")

    model_info = model_dict[args.model]

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

    task_names = list(per_task_data.keys())
    accuracies = [per_task_data[task]['accuracy'] for task in task_names]
    f1_scores = [per_task_data[task]['f1_score'] for task in task_names]
    ## save csv file
    df_all = pd.DataFrame({
        "task_name": task_names,
        "accuracy": accuracies,
        "f1": f1_scores
    })

    # Apply parsing
    parsed_df = df_all["task_name"].apply(parse_task_name_dict).apply(pd.Series)
    df_all = pd.concat([df_all, parsed_df], axis=1)

    for cue in ["shape", "color", "size"]:
        df_all[f"rel_{cue}"] = df_all["relevant"].apply(lambda L: cue in L)
        df_all[f"irrel_{cue}"] = df_all["irrelevant"].apply(lambda L: cue in L)

    # --------------------------
    # 4. Factor-level weakness
    # --------------------------
    categories = config.categories[principle]

    all_results = {}

    for category in categories:
        print("\nAnalyzing category:", category)
        df = df_all[df_all["task_name"].str.contains(category, na=False)].copy()
        factor_results = {
            "rel_shape": df.loc[df["rel_shape"], "f1"].mean(),
            "rel_color": df.loc[df["rel_color"], "f1"].mean(),
            "rel_size": df.loc[df["rel_size"], "f1"].mean(),
            "group_1": df.loc[df["group_count"] == 1, "f1"].mean(),
            "group_2": df.loc[df["group_count"] == 2, "f1"].mean(),
            "group_3": df.loc[df["group_count"] == 3, "f1"].mean(),
            "group_4": df.loc[df["group_count"] == 4, "f1"].mean(),
            "size_s": df.loc[df["size"] == "s", "f1"].mean(),
            "size_m": df.loc[df["size"] == "m", "f1"].mean(),
            "size_l": df.loc[df["size"] == "l", "f1"].mean(),
            "size_xl": df.loc[df["size"] == "xl", "f1"].mean()
        }
        if "rule_type" in df.columns:
            factor_results["rule_all"] = df.loc[df["rule_type"] == "all", "f1"].mean()
            factor_results["rule_exist"] = df.loc[df["rule_type"] == "exist", "f1"].mean()

        factor_df = pd.Series(factor_results)
        factor_df_sorted = factor_df.sort_values()
        print("\n=== Factor-Level Weakness ===")
        print(factor_df_sorted)

        # --------------------------
        # 5. Combination-level weakness
        # --------------------------

        from itertools import combinations

        binary_cols = [
            "rel_shape", "rel_color", "rel_size",
            "irrel_shape", "irrel_color", "irrel_size"
        ]

        combo_results = []

        for r in range(2, 5):  # combinations of size 2–4
            for combo in combinations(binary_cols, r):
                mask = df[list(combo)].all(axis=1)
                if mask.sum() >= 3:  # only consider combinations with data
                    mean_f1 = df.loc[mask, "f1"].mean()
                    combo_results.append((combo, mean_f1, mask.sum()))

        worst_combos = sorted(combo_results, key=lambda x: x[1])[:15]

        print("\n=== Worst 15 Factor Combinations ===")
        for combo, mean_f1, count in worst_combos:
            print(f"{combo} → F1 = {mean_f1:.3f} over {count} tasks")

        # --------------------------
        # 6. Factor Combination Heatmap (rel_shape, rel_color, rel_size)
        # Create a 3x3 matrix showing pairwise combinations
        # --------------------------
        print("\n=== Generating Factor Combination Heatmap (3x3) ===")

        # Define the three factors
        factors = ['rel_shape', 'rel_color', 'rel_size']
        factor_labels = ['Shape', 'Color', 'Size']

        # Create 3x3 matrix: each cell shows performance when those two factors are present
        heatmap_data = np.zeros((3, 3))
        count_data = np.zeros((3, 3), dtype=int)

        for i, factor1 in enumerate(factors):
            for j, factor2 in enumerate(factors):
                if i == j:
                    # Diagonal: only this single factor is present
                    mask = df[factor1] & ~df[factors[(i + 1) % 3]] & ~df[factors[(i + 2) % 3]]
                else:
                    # Off-diagonal: both factors are present, third is not
                    third_idx = [k for k in range(3) if k != i and k != j][0]
                    mask = df[factor1] & df[factor2] & ~df[factors[third_idx]]

                if mask.sum() > 0:
                    heatmap_data[i, j] = df.loc[mask, "f1"].mean()
                    count_data[i, j] = mask.sum()
                else:
                    heatmap_data[i, j] = np.nan

        # Create DataFrame for heatmap
        combo_df = pd.DataFrame(heatmap_data,
                                index=factor_labels,
                                columns=factor_labels)

        # Create mask for upper triangle (excluding diagonal) to show only lower triangular matrix
        mask = np.triu(np.ones_like(combo_df, dtype=bool), k=1)

        # Plot heatmap with only lower triangle shown
        fig_combo, ax_combo = plt.subplots(figsize=(8, 7))
        sns.heatmap(combo_df, annot=True, fmt=".3f", cmap="RdYlGn",
                    vmin=0, vmax=1, ax=ax_combo, cbar_kws={'label': 'Mean F1 Score'},
                    linewidths=1, linecolor='gray', mask=mask, annot_kws={'size': 14})
        ax_combo.set_title(f"Pairwise Factor Combination Performance: {category}",
                           fontsize=14, fontweight='bold', pad=15)
        ax_combo.set_xlabel("Factor", fontsize=12)
        ax_combo.set_ylabel("Factor", fontsize=12)

        # Add text annotations with task counts (only for lower triangle)
        # Use adaptive text color based on background
        import matplotlib.colors as mcolors
        cmap = plt.cm.RdYlGn
        norm = mcolors.Normalize(vmin=0, vmax=1)

        for i in range(3):
            for j in range(3):
                if i >= j and count_data[i, j] > 0:  # Only show for lower triangle (including diagonal)
                    # Get background color value and determine text color
                    value = heatmap_data[i, j]
                    if not np.isnan(value):
                        rgba = cmap(norm(value))
                        # Calculate luminance
                        luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                        text_color = 'white' if luminance < 0.5 else 'black'
                    else:
                        text_color = 'black'

                    ax_combo.text(j + 0.5, i + 0.85, f"n={count_data[i, j]}",
                                  ha='center', va='top', fontsize=9, color=text_color, weight='bold')

        plt.tight_layout()

        # Save the heatmap
        combo_heatmap_path = config.result_path / principle / f"factor_combo_heatmap_{model_info['model']}_{model_info['img_num']}_{category}.pdf"
        plt.savefig(combo_heatmap_path, format="pdf", bbox_inches="tight")
        print(f"Factor combination heatmap saved to {combo_heatmap_path}")
        plt.close(fig_combo)

        all_results[category] = {
            "factor_df": factor_df,
            "worst_combos": worst_combos,
            "combo_matrix": heatmap_data,
            "combo_counts": count_data
        }

    # --------------------------
    # Aggregate per-category combo matrices and draw lower-triangle mean heatmap
    # --------------------------
    try:
        # Collect matrices and count matrices from all categories
        combo_matrices = [res["combo_matrix"] for res in all_results.values() if "combo_matrix" in res]
        combo_counts = [res["combo_counts"] for res in all_results.values() if "combo_counts" in res]
        if len(combo_matrices) > 0:
            combo_stack = np.array(combo_matrices, dtype=float)  # shape: (n_categories, 3, 3)
            # Use nanmean to ignore missing cells
            mean_combo = np.nanmean(combo_stack, axis=0)

            # Aggregate counts by summing
            total_counts = np.sum(np.array(combo_counts, dtype=int), axis=0)

            # Build DataFrame for plotting
            factor_labels = ['Shape', 'Color', 'Size']
            mean_df = pd.DataFrame(mean_combo, index=factor_labels, columns=factor_labels)

            # Mask upper triangle to show only lower triangle (including diagonal)
            upper_mask = np.triu(np.ones_like(mean_df, dtype=bool), k=1)

            fig_mean, ax_mean = plt.subplots(figsize=(6, 5))
            sns.heatmap(mean_df, annot=True, fmt=".3f", cmap="RdBu_r",
                        vmin=0, vmax=1, ax=ax_mean, cbar_kws={'label': 'Mean F1 Score'},
                        linewidths=0.8, linecolor='gray', mask=upper_mask, annot_kws={'size': 14})

            ax_mean.set_title(f"Mean Pairwise Factor Combination Performance (averaged over categories): {principle}",
                              fontsize=14, fontweight='bold', pad=12)
            ax_mean.set_xlabel("Factor", fontsize=12)
            ax_mean.set_ylabel("Factor", fontsize=12)
            ax_mean.set_xticklabels(ax_mean.get_xticklabels(), rotation=45, ha='right', fontsize=11)
            ax_mean.set_yticklabels(ax_mean.get_yticklabels(), rotation=0, fontsize=11)

            # Annotate counts in the lower triangle (below the numeric value)
            # Use adaptive text color based on background
            import matplotlib.colors as mcolors
            cmap = plt.cm.RdBu_r
            norm = mcolors.Normalize(vmin=0, vmax=1)

            for i in range(3):
                for j in range(3):
                    if i >= j:
                        n = int(total_counts[i, j])
                        # If there were no tasks contributing, show '-' instead
                        text = f"n={n}" if n > 0 else "n=0"

                        # Get background color value and determine text color
                        value = mean_combo[i, j]
                        if not np.isnan(value):
                            rgba = cmap(norm(value))
                            # Calculate luminance
                            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                            text_color = 'white' if luminance < 0.5 else 'black'
                        else:
                            text_color = 'black'

                        # place the count slightly below the number
                        ax_mean.text(j + 0.5, i + 0.65, text, ha='center', va='top', fontsize=9, color=text_color)

            plt.tight_layout()
            save_mean_path = config.figure_path / f"mean_factor_combo_heatmap_{principle}.pdf"
            plt.savefig(save_mean_path, format="pdf", bbox_inches="tight")
            print(f"Mean factor-combo lower-triangle heatmap saved to {save_mean_path}")
            plt.close(fig_mean)
    except Exception as e:
        print(f"Failed to create mean factor-combo heatmap: {e}")

    ## build the 2d table
    factor_table = pd.DataFrame({
        category: results["factor_df"] for category, results in all_results.items()
    }).T

    if principle == "closure":
        # Rename row header: replace "non_overlap_big_circle" with "separate_big_circle"
        factor_table.rename(index={"non_overlap_big_circle": "big_circle"}, inplace=True)
        factor_table.rename(index={"separate_big_square": "big_square"}, inplace=True)
        factor_table.rename(index={"separate_big_triangle": "big_triangle"}, inplace=True)
        factor_table.rename(index={"non_overlap_feature_triangle": "feature_triangle"}, inplace=True)
        factor_table.rename(index={"non_overlap_feature_square": "feature_square"}, inplace=True)
        factor_table.rename(index={"non_overlap_feature_circle": "feature_circle"}, inplace=True)
    elif principle == "continuity":
        factor_table.rename(index={"with_intersected_n_splines": "intersected_splines"}, inplace=True)
        factor_table.rename(index={"non_intersected_n_splines": "non_intersected_splines"}, inplace=True)
        factor_table.rename(index={"feature_continuity_overlap_splines": "feature_splines"}, inplace=True)

    # Reorder columns in a logical sequence
    desired_order = []
    # First: relevance factors
    for col in ['rel_shape', 'rel_color', 'rel_size']:
        if col in factor_table.columns:
            desired_order.append(col)
    # Second: irrelevance factors
    for col in ['irrel_shape', 'irrel_color', 'irrel_size']:
        if col in factor_table.columns:
            desired_order.append(col)
    # Third: group counts in order
    for col in ['group_1', 'group_2', 'group_3', 'group_4']:
        if col in factor_table.columns:
            desired_order.append(col)
    # Fourth: sizes in order
    for col in ['size_s', 'size_m', 'size_l', 'size_xl']:
        if col in factor_table.columns:
            desired_order.append(col)
    # Fifth: rule types
    for col in ['rule_all', 'rule_exist']:
        if col in factor_table.columns:
            desired_order.append(col)
    # Add any remaining columns not in the desired order
    for col in factor_table.columns:
        if col not in desired_order:
            desired_order.append(col)

    # Reorder the columns
    factor_table = factor_table[desired_order]

    # Add a 'Mean' column at the end showing the mean value for each row
    factor_table['Mean'] = factor_table.mean(axis=1)

    # Add a 'Mean' row at the bottom showing the mean value for each factor (column)
    mean_row = factor_table.mean(axis=0)
    mean_row.name = 'Mean'
    factor_table = pd.concat([factor_table, mean_row.to_frame().T])

    # Validate column order before plotting
    print(f"\n{'=' * 80}")
    print(f"Column order for {principle}:")
    print(f"{'=' * 80}")
    for i, col in enumerate(factor_table.columns):
        print(f"{i + 1}. {col}")

    # Check if group_1 is in the correct position
    if 'group_1' in factor_table.columns:
        group_1_position = list(factor_table.columns).index('group_1')
        expected_position_after = []
        for col in ['rel_shape', 'rel_color', 'rel_size', 'irrel_shape', 'irrel_color', 'irrel_size']:
            if col in factor_table.columns:
                expected_position_after.append(col)
        expected_pos = len(expected_position_after)
        actual_pos = group_1_position
        print(f"\ngroup_1 position check:")
        print(f"  Expected position: {expected_pos + 1} (after {len(expected_position_after)} relevance/irrelevance factors)")
        print(f"  Actual position: {actual_pos + 1}")
        if actual_pos == expected_pos:
            print(f"  ✓ group_1 is in the CORRECT position")
        else:
            print(f"  ✗ WARNING: group_1 is in the WRONG position!")
    print(f"{'=' * 80}\n")

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(factor_table,
                annot=True, cmap="coolwarm", vmin=0, vmax=1, ax=ax,
                mask=factor_table.isna(), fmt='.3f', annot_kws={'size': 15})
    ## show xy labels
    ax.set_xlabel("Factor", fontsize=20)
    ax.set_ylabel("Category", fontsize=20)
    # Set y-axis labels to show task category names
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=15, ha='right')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=15, ha='right')

    # Add diagonal crosses to empty cells
    for i in range(len(factor_table.index)):
        for j in range(len(factor_table.columns)):
            if pd.isna(factor_table.iloc[i, j]):
                # Draw diagonal lines to form an X
                ax.plot([j, j + 1], [i, i + 1], color='gray', linewidth=1.5, alpha=0.5)
                ax.plot([j, j + 1], [i + 1, i], color='gray', linewidth=1.5, alpha=0.5)

    plt.title(f"{principle.capitalize()}: Raw Factor-Level Performance per Category", fontsize=20, pad=15)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)  # Make room for caption

    # save the figure to PDF
    save_path = config.figure_path / f"task_factor_analysis_{principle}.pdf"
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Heatmap saved to {save_path}")

    plt.show()
    plt.close(fig)

    print("")

    # Return combo data if requested for cross-principle aggregation
    if return_combo_data:
        combo_matrices = [res["combo_matrix"] for res in all_results.values() if "combo_matrix" in res]
        combo_counts = [res["combo_counts"] for res in all_results.values() if "combo_counts" in res]
        return combo_matrices, combo_counts
    else:
        return None


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
                    vmin=vmin, vmax=vmax, ax=ax, annot_kws={"size": 12})
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


def collect_obj_ablation_results(args, props, all_principles, official_model_names):
    results = {prop: {principle: {model: {"related": [], "irrelated": []} for model in official_model_names}
                      for principle in all_principles}
               for prop in props}
    for prop in props:
        for principle in all_principles:
            for model_name, model_info in model_dict.items():
                json_path = get_results_path(args.remote, principle, model_info["model"], model_info["img_num"])
                per_task_data = get_per_task_data(json_path, principle)
                for task_name, task_res in per_task_data.items():
                    if task_name[-1] != "_":
                        task_name += "_"
                    task_info = parse_task_name(task_name)
                    if prop in task_info.get("related_concepts", []):
                        results[prop][principle][model_name]["related"].append(task_res["accuracy"])
                    elif prop in task_info.get("irrelated_concepts", []):
                        results[prop][principle][model_name]["irrelated"].append(task_res["accuracy"])
    return results


def plot_obj_ablation_grouped_bar(results, props, all_principles, official_model_names, model_names, save_path):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    fig, axes = plt.subplots(1, len(props), figsize=(6 * len(props), 6), sharey=True)
    if len(props) == 1:
        axes = [axes]
    bar_width = 0.35
    x = np.arange(len(official_model_names))
    palette = sns.color_palette("Set2", n_colors=2)
    for idx, prop in enumerate(props):
        ax = axes[idx]
        rel_vals = []
        irrel_vals = []
        for model in official_model_names:
            rel_accs = []
            irrel_accs = []
            for principle in all_principles:
                rel_accs.extend(results[prop][principle][model]["related"])
                irrel_accs.extend(results[prop][principle][model]["irrelated"])
            rel_vals.append(np.mean(rel_accs) if rel_accs else np.nan)
            irrel_vals.append(np.mean(irrel_accs) if irrel_accs else np.nan)
        ax.bar(x - bar_width / 2, rel_vals, width=bar_width, color=palette[0], label="related")
        ax.bar(x + bar_width / 2, irrel_vals, width=bar_width, color=palette[1], label="irrelated")
        for i, v in enumerate(rel_vals):
            ax.text(x[i] - bar_width / 2, v + 2, f"{v:.1f}", ha='center', va='bottom', fontsize=15, rotation=60)
        for i, v in enumerate(irrel_vals):
            ax.text(x[i] + bar_width / 2, v + 2, f"{v:.1f}", ha='center', va='bottom', fontsize=15, rotation=60)
        ax.set_title(f"Acc: {prop}", fontsize=30, fontweight='bold')
        if idx == 0:
            ax.set_ylabel("Accuracy", fontsize=25)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=25)
        ax.set_ylim(0, 100)
        ax.axhline(50, color='gray', linestyle='--', linewidth=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=20, loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


def plot_obj_ablation_matrix(results, props, all_principles, model_names, official_model_names, save_path):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    n_props = len(props)
    n_principles = len(all_principles)
    fig, axes = plt.subplots(n_props, n_principles, figsize=(5 * n_principles, 5 * n_props), sharey='row')
    palette = sns.color_palette("Set2", n_colors=2)
    bar_width = 0.10
    group_gap = 0.05

    for row_idx, prop in enumerate(props):
        for col_idx, principle in enumerate(all_principles):
            ax = axes[row_idx, col_idx] if n_props > 1 else axes[col_idx]
            categories = ["related", "irrelated"]
            x = np.arange(len(model_names))
            n = 2
            group_width = n * bar_width + group_gap
            for i, cat in enumerate(categories):
                values = [np.mean(results[prop][principle][model][cat]) if results[prop][principle][model][cat] else np.nan
                          for model in official_model_names]
                ax.bar(x * group_width + i * bar_width, values, width=bar_width, color=palette[i])
                for j, v in enumerate(values):
                    ax.text(x[j] * group_width + i * bar_width, v + 2, f"{v:.1f}", rotation=60, ha='center', va='bottom', fontsize=10)
            ax.set_xticks(x * group_width + (n * bar_width) / 2)
            ax.set_xticklabels(model_names, fontsize=30, ha='right')
            ax.set_xlabel(prop, fontsize=25)
            if row_idx == 0:
                ax.set_title(principle, fontsize=35, fontweight='bold')
            ax.set_ylim(0, 100)
            if col_idx == 0:
                ax.set_ylabel('Accuracy', fontsize=25)
            ax.axhline(50, color='gray', linestyle='--', linewidth=1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[i]) for i in range(2)]
    fig.legend(handles, ["related", "irrelated"], loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=30)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()


def analysis_obj_ablation_performance(args):
    import matplotlib
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"], "ytick.labelsize": 30})
    props = ["color", "shape", "size"]
    all_principles = ["proximity", "similarity", "closure", "symmetry", "continuity"]
    model_names = ["M1", "M2", "M3", "M4", "M5"]
    official_model_names = list(model_dict.keys())

    results = collect_obj_ablation_results(args, props, all_principles, official_model_names)
    matrix_save_path = config.figure_path / "ablation_obj_all_props_all_principles_matrix.pdf"
    plot_obj_ablation_matrix(results, props, all_principles, model_names, official_model_names, matrix_save_path)

    grouped_save_path = config.figure_path / "obj_ablation_avg_accuracy_per_prop_grouped.pdf"
    plot_obj_ablation_grouped_bar(results, props, all_principles, official_model_names, model_names, grouped_save_path)


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
    n_models = len(model_names)

    # Create a single figure with one subplot per property (averaged across all principles)
    fig, axes = plt.subplots(1, n_props, figsize=(8 * n_props, 6), sharey=True)
    if n_props == 1:
        axes = [axes]

    palette = sns.color_palette("Set2", n_colors=n_models)

    for prop_idx, prop in enumerate(props):
        # Collect results across ALL principles for this property
        all_categories = set()
        aggregated_results = {model: {} for model in model_names}

        for principle in all_principles:
            for model_name, model_info in model_dict.items():
                json_path = get_results_path(args.remote, principle, model_info["model"], model_info["img_num"])
                per_task_data = get_per_task_data(json_path, principle)

                for task_name, task_res in per_task_data.items():
                    task_info = parse_task_name(task_name)
                    if prop not in task_info:
                        continue
                    value = task_info[prop]
                    if value == 6:
                        print("")
                    all_categories.add(value)

                    if value not in aggregated_results[model_name]:
                        aggregated_results[model_name][value] = []
                    aggregated_results[model_name][value].append(task_res["f1_score"] * 100)

        # Sort categories
        if "s" in all_categories and "m" in all_categories and "l" in all_categories:
            categories = ["s", "m", "l", "xl"]
            categories = [c for c in categories if c in all_categories]
        else:
            categories = sorted(list(all_categories))

        # For group_num, only show groups 1-4
        if prop == "group_num":
            categories = [c for c in categories if c in [1, 2, 3, 4]]

        # Compute average across all principles for each model and category
        avg_results = {}
        for model in model_names:
            avg_results[model] = {}
            for cat in categories:
                if cat in aggregated_results[model] and len(aggregated_results[model][cat]) > 0:
                    avg_results[model][cat] = np.mean(aggregated_results[model][cat])
                else:
                    avg_results[model][cat] = np.nan

        # Plot the averaged results as line charts
        ax = axes[prop_idx]
        x = np.arange(len(categories))

        # Define markers for each model
        markers = ['o', 's', '^', 'D', 'v']  # circle, square, triangle up, diamond, triangle down

        for i, model_name in enumerate(model_names):
            values = [avg_results[model_name].get(cat, np.nan) for cat in categories]

            # Plot line with markers
            ax.plot(x, values, marker=markers[i], markersize=10, linewidth=2.5,
                   color=palette[i], label=model_name, markeredgecolor='white',
                   markeredgewidth=1.5, alpha=0.9)

            # # Add value labels next to markers
            # for j, v in enumerate(values):
            #     if not np.isnan(v):
            #         ax.text(x[j], v + 1.2, f"{v:.1f}",
            #                ha='center', va='bottom', fontsize=10, fontweight='bold',
            #                color=palette[i])

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=20)

        # Set labels
        prop_label = "Group Number" if prop == "group_num" else "Group Size"
        ax.set_xlabel(prop_label, fontsize=22, fontweight='bold')
        ax.set_title(f"Average Performance by {prop_label}\n(Across All Principles)",
                    fontsize=20, fontweight='bold', pad=15)

        if prop_idx == 0:
            ax.set_ylabel('F1 Score (%)', fontsize=22, fontweight='bold')

        ax.set_ylim(20, 80)
        ax.axhline(50, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=18)
        ax.tick_params(axis='x', labelsize=18)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.grid(axis='x', alpha=0.2, linestyle=':', linewidth=0.5)

    # Add legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.1), ncol=n_models,
               fontsize=18, frameon=True, title="Models", title_fontsize=20)

    plt.suptitle("Ablation Analysis: Average Performance Across All Gestalt Principles",
                fontsize=24, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.12, 1, 0.96])

    save_path = config.figure_path / f"ablation_average_across_principles.pdf"
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Averaged ablation chart saved to: {save_path}")


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


def analysis_all_principles_merged(args):
    """
    Generate a merged heatmap for each model across all principles.
    For each model, principles are stacked vertically with black dashed lines separating them.
    One PDF file per model is saved.
    """
    from itertools import combinations

    # Iterate through all models
    for model_name, model_info in model_dict.items():
        print(f"\n{'=' * 80}")
        print(f"Processing MODEL: {model_name}")
        print(f"{'=' * 80}")

        all_principle_tables = []
        principle_row_counts = []

        for principle in principles:
            print(f"\n    Processing principle: {principle.upper()}")

            try:
                json_path = get_results_path(args.remote, principle, model_info["model"], model_info["img_num"])
                per_task_data = get_per_task_data(json_path, principle)

                # replace the soloar with solar if exists in the keys of the per_task_data
                per_task_data = {re.sub(r"soloar", "solar", k): v for k, v in per_task_data.items()}
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

                task_names = list(per_task_data.keys())
                accuracies = [per_task_data[task]['accuracy'] for task in task_names]
                f1_scores = [per_task_data[task]['f1_score'] for task in task_names]

                df_all = pd.DataFrame({
                    "task_name": task_names,
                    "accuracy": accuracies,
                    "f1": f1_scores
                })

                # Apply parsing
                parsed_df = df_all["task_name"].apply(parse_task_name_dict).apply(pd.Series)
                df_all = pd.concat([df_all, parsed_df], axis=1)

                for cue in ["shape", "color", "size"]:
                    df_all[f"rel_{cue}"] = df_all["relevant"].apply(lambda L: cue in L)
                    df_all[f"irrel_{cue}"] = df_all["irrelevant"].apply(lambda L: cue in L)

                # Get categories for this principle
                principle_categories = config.categories.get(principle, [])

                all_results = {}
                for category in principle_categories:
                    df = df_all[df_all["task_family"].str.contains(category, na=False)].copy()

                    if len(df) == 0:
                        continue

                    factor_results = {}
                    for cue in ["shape", "color", "size"]:
                        if df[f"rel_{cue}"].any():
                            factor_results[f"rel_{cue}"] = df.loc[df[f"rel_{cue}"], "f1"].mean()

                    for group_val in [1, 2, 3, 4]:
                        if (df["group_count"] == group_val).any():
                            factor_results[f"group_{group_val}"] = df.loc[df["group_count"] == group_val, "f1"].mean()

                    for size_val in ["s", "m", "l", "xl"]:
                        if (df["size"] == size_val).any():
                            factor_results[f"size_{size_val}"] = df.loc[df["size"] == size_val, "f1"].mean()

                    factor_df = pd.Series(factor_results)
                    all_results[category] = {"factor_df": factor_df}

                ## build the 2d table
                factor_table = pd.DataFrame({
                    category: results["factor_df"] for category, results in all_results.items()
                }).T

                if principle == "closure":
                    factor_table.rename(index={"non_overlap_big_circle": "big_circle"}, inplace=True)
                    factor_table.rename(index={"separate_big_square": "big_square"}, inplace=True)
                    factor_table.rename(index={"separate_big_triangle": "big_triangle"}, inplace=True)
                    factor_table.rename(index={"non_overlap_feature_triangle": "feature_triangle"}, inplace=True)
                    factor_table.rename(index={"non_overlap_feature_square": "feature_square"}, inplace=True)
                    factor_table.rename(index={"non_overlap_feature_circle": "feature_circle"}, inplace=True)
                elif principle == "continuity":
                    factor_table.rename(index={"with_intersected_n_splines": "intersected_splines"}, inplace=True)
                    factor_table.rename(index={"non_intersected_n_splines": "non_intersected_splines"}, inplace=True)
                    factor_table.rename(index={"feature_continuity_overlap_splines": "feature_splines"}, inplace=True)
                elif principle == "symmetry":
                    factor_table.rename(index={"solar_sys": "axis_symm_with_bkg"}, inplace=True)
                    factor_table.rename(index={"symmetry_circle": "axis_symm"}, inplace=True)
                    factor_table.rename(index={"symmetry_pattern": "radial_symmetry"}, inplace=True)
                # Reorder columns in a logical sequence
                desired_order = []
                for col in ['rel_shape', 'rel_color', 'rel_size']:
                    if col in factor_table.columns:
                        desired_order.append(col)
                for col in ['irrel_shape', 'irrel_color', 'irrel_size']:
                    if col in factor_table.columns:
                        desired_order.append(col)
                for col in ['group_1', 'group_2', 'group_3', 'group_4']:
                    if col in factor_table.columns:
                        desired_order.append(col)
                for col in ['size_s', 'size_m', 'size_l', 'size_xl']:
                    if col in factor_table.columns:
                        desired_order.append(col)
                for col in ['rule_all', 'rule_exist']:
                    if col in factor_table.columns:
                        desired_order.append(col)
                for col in factor_table.columns:
                    if col not in desired_order:
                        desired_order.append(col)

                factor_table = factor_table[desired_order]
                factor_table['Mean'] = factor_table.mean(axis=1)
                factor_table.index = [f"{principle}:{idx}" for idx in factor_table.index]

                all_principle_tables.append(factor_table)
                principle_row_counts.append(len(factor_table))

                print(f"    ✓ Collected {len(factor_table)} categories for {principle}")

            except Exception as e:
                print(f"    ✗ Failed to process {principle}: {e}")
                continue

        # After collecting all principles for this model, concatenate and plot
        if not all_principle_tables:
            print(f"  ✗ No data collected for model {model_name}. Skipping.")
            continue

        combined_df = pd.concat(all_principle_tables, axis=0)
        overall_mean = combined_df.mean(axis=0)
        overall_mean.name = "Overall_Mean"
        combined_df = pd.concat([combined_df, overall_mean.to_frame().T])
        grp_one_col = combined_df.pop("group_1")
        combined_df.insert(3, "group_1", grp_one_col)

        fig, ax = plt.subplots(figsize=(12, max(8, combined_df.shape[0] * 0.4)))
        sns.heatmap(combined_df, annot=True, fmt=".3f", cmap="coolwarm", vmin=0, vmax=1,
                    mask=combined_df.isna(), ax=ax, annot_kws={'size': 12})

        for i in range(len(combined_df.index)):
            for j in range(len(combined_df.columns)):
                if pd.isna(combined_df.iloc[i, j]):
                    ax.plot([j, j + 1], [i, i + 1], color='gray', linewidth=1.5, alpha=0.5)
                    ax.plot([j, j + 1], [i + 1, i], color='gray', linewidth=1.5, alpha=0.5)

        running = 0
        for cnt in principle_row_counts:
            running += cnt
            ax.hlines(running, *ax.get_xlim(), colors='black', linestyles='dashed', linewidth=1.5)

        ax.set_xlabel("Factor", fontsize=20)
        ax.set_ylabel("Category", fontsize=20)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=14)
        plt.title(f"Merged Factor-Level Performance across Principles — Model: {model_name}",
                  fontsize=16, fontweight='bold', pad=12)

        save_path = config.figure_path / f"merged_task_factor_analysis_{model_name}.pdf"
        plt.tight_layout()
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"\n✓ Saved merged heatmap for {model_name} to {save_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline models with CUDA support.")
    parser.add_argument("--model", type=str, required=True, help="Specify the principle to filter data.")
    parser.add_argument("--principle", type=str, required=False, default="all")
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--mode", type=str, default="avg_principle")
    parser.add_argument("--img_num", type=int)
    args = parser.parse_args()

    if args.mode == "merged":
        # For merged mode, we don't need json_path
        analysis_all_principles_merged(args)
    else:
        json_path = get_results_path(args.remote, args.principle, args.model, args.img_num)

        if args.mode == "principle":
            analysis_average_performance(json_path, args.principle, args.model, args.img_num)
        elif args.mode == "category":
            analysis_models(args)
        elif args.mode == "task":
            # Collect combo data across all principles for each model
            model_combo_data = {model_name: {"matrices": [], "counts": []}
                               for model_name in model_dict.keys()}

            print("\n" + "="*80)
            print("Collecting data for ALL models across ALL principles...")
            print("="*80)

            # Process each model and each principle
            for model_name, model_info in model_dict.items():
                print(f"\nProcessing model: {model_name}")
                for principle in config.categories.keys():
                    # Create a temporary args object for this model and principle
                    temp_args = argparse.Namespace(**vars(args))
                    temp_args.principle = principle
                    temp_args.model = model_name

                    try:
                        result = analysis_tasks(temp_args, return_combo_data=True)
                        if result is not None:
                            combo_matrices, combo_counts = result
                            model_combo_data[model_name]["matrices"].extend(combo_matrices)
                            model_combo_data[model_name]["counts"].extend(combo_counts)
                            print(f"  ✓ {principle}: collected {len(combo_matrices)} category matrices")
                    except Exception as e:
                        print(f"  ✗ {principle}: {str(e)}")

            # Generate 5-subplot figure with one heatmap per model
            print("\n" + "="*80)
            print("Generating 5-model comparison heatmap...")
            print("="*80)

            fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)
            factor_labels = ['Shape', 'Color', 'Size']
            upper_mask = np.triu(np.ones((3, 3), dtype=bool), k=1)

            # Model names for titles
            model_names_short = ["ViT", "LLaVA", "InternVL3-2B", "InternVL3-78B", "GPT-5"]

            for idx, (model_name, short_name) in enumerate(zip(model_dict.keys(), model_names_short)):
                ax = axes[idx]

                matrices = model_combo_data[model_name]["matrices"]
                counts = model_combo_data[model_name]["counts"]

                if len(matrices) > 0:
                    combo_stack = np.array(matrices, dtype=float)
                    mean_combo = np.nanmean(combo_stack, axis=0)
                    total_counts = np.sum(np.array(counts, dtype=int), axis=0)

                    mean_df = pd.DataFrame(mean_combo, index=factor_labels, columns=factor_labels)

                    # Plot heatmap
                    sns.heatmap(mean_df, annot=True, fmt=".2f", cmap="Blues",
                               vmin=0.0, vmax=1.0, ax=ax,
                               cbar=(idx == 4),  # Only show colorbar on last subplot
                               cbar_kws={'label': 'Mean F1 Score'} if idx == 4 else None,
                               linewidths=0.8, linecolor='black', mask=upper_mask,
                               annot_kws={'size': 22, "weight":'bold'})

                    ax.set_title(short_name, fontsize=30, fontweight='bold', pad=10)
                    # ax.set_xlabel("Factor", fontsize=11)
                    # if idx == 0:
                    #     ax.set_ylabel("Factor", fontsize=11)

                    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=25)
                    if idx == 0:
                        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=25)

                    # Annotate counts (smaller font for 5-panel layout)
                    # Use adaptive text color based on background
                    import matplotlib.colors as mcolors
                    cmap = plt.cm.RdBu_r
                    norm = mcolors.Normalize(vmin=0.3, vmax=0.8)

                    for i in range(3):
                        for j in range(3):
                            if i >= j:
                                n = int(total_counts[i, j])
                                if n > 0:
                                    # Get background color value and determine text color
                                    value = mean_combo[i, j]
                                    if not np.isnan(value):
                                        rgba = cmap(norm(value))
                                        # Calculate luminance
                                        luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                                        text_color = 'white' if luminance < 0.5 else 'black'
                                    else:
                                        text_color = 'black'

                                    ax.text(j + 0.5, i + 0.7, f"n={n}", ha='center', va='top',
                                           fontsize=20, color=text_color)

                    print(f"  ✓ {short_name}: plotted {len(matrices)} matrices")
                else:
                    ax.text(0.5, 0.5, "No data", ha='center', va='center',
                           transform=ax.transAxes, fontsize=12)
                    ax.set_title(short_name, fontsize=16, fontweight='bold', pad=10)
                    print(f"  ✗ {short_name}: no data available")

            plt.suptitle("Mean Pairwise Factor Combination Performance Across All Principles",
                        fontsize=25, fontweight='bold', y=1.02)
            plt.tight_layout()

            multi_model_save_path = config.figure_path / "all_models_factor_combo_heatmap_all_principles.pdf"
            plt.savefig(multi_model_save_path, format="pdf", bbox_inches="tight")
            print(f"\n✓ Multi-model factor-combo heatmap saved to: {multi_model_save_path}")
            plt.close(fig)
        elif args.mode == "ablation":
            analysis_ablation_performance(args)
        elif args.mode == "obj_ablation":
            analysis_obj_ablation_performance(args)
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
