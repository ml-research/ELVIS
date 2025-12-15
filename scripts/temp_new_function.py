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

                # Add principle prefix to row names
                factor_table.index = [f"{principle}:{idx}" for idx in factor_table.index]

                # Collect this principle's table (without the mean row added at the end)
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

        # Concatenate all principle tables
        combined_df = pd.concat(all_principle_tables, axis=0)

        # Add overall mean row at bottom
        overall_mean = combined_df.mean(axis=0)
        overall_mean.name = "Overall_Mean"
        combined_df = pd.concat([combined_df, overall_mean.to_frame().T])

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(16, max(8, combined_df.shape[0] * 0.4)))
        sns.heatmap(combined_df, annot=True, fmt=".3f", cmap="coolwarm", vmin=0, vmax=1,
                    mask=combined_df.isna(), ax=ax, annot_kws={'size': 9}, cbar_kws={'label': 'Mean F1'})

        # Add diagonal crosses to empty cells
        for i in range(len(combined_df.index)):
            for j in range(len(combined_df.columns)):
                if pd.isna(combined_df.iloc[i, j]):
                    # Draw diagonal lines to form an X
                    ax.plot([j, j + 1], [i, i + 1], color='gray', linewidth=1.5, alpha=0.5)
                    ax.plot([j, j + 1], [i + 1, i], color='gray', linewidth=1.5, alpha=0.5)

        # Draw dashed horizontal lines separating principles
        running = 0
        for cnt in principle_row_counts:
            running += cnt
            ax.hlines(running, *ax.get_xlim(), colors='black', linestyles='dashed', linewidth=1.5)

        ax.set_xlabel("Factor", fontsize=14)
        ax.set_ylabel("Principle:Category", fontsize=14)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        plt.title(f"Merged Factor-Level Performance across Principles — Model: {model_name}",
                  fontsize=16, fontweight='bold', pad=12)

        save_path = config.figure_path / f"merged_task_factor_analysis_{model_name}.pdf"
        plt.tight_layout()
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"\n✓ Saved merged heatmap for {model_name} to {save_path}\n")

