# Created by MacBook Pro at 30.08.25
from scripts import config


def merge_two_gpt_results(file1, file2, output_file):
    import json

    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    merged_data = {**data1, **data2}  # Merge dictionaries

    with open(output_file, 'w') as out_f:
        json.dump(merged_data, out_f, indent=4)

    print(f"Merged results saved to {output_file}")


def load_csv(file_path):
    import pandas as pd
    return pd.read_csv(file_path)


def get_all_subfolder_names(directory):
    from pathlib import Path
    directory = Path(directory)
    return [f.name for f in directory.iterdir() if f.is_dir()]


def save_json(data, file_path):
    import json
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


# def merge_symm(res_path):
#     symm_names_1 = sorted(get_all_subfolder_names(config.get_raw_patterns_path() / "res_200_pin_False" / "symmetry" / "train"))
#
#     symm_acc_file1 = res_path / "symmetry" / "wandb_export_0_156_acc.csv"
#     symm_f1_file1 = res_path / "symmetry" / "wandb_export_0_156_f1_score.csv"
#     symm_precision_file1 = res_path / "symmetry" / "wandb_export_0_156_precision.csv"
#     symm_recall_file1 = res_path / "symmetry" / "wandb_export_0_156_recall.csv"
#     symm_acc_1 = load_csv(symm_acc_file1)["0_156 - symmetry/test_accuracy"].values
#     symm_f1_1 = load_csv(symm_f1_file1)["0_156 - symmetry/test_accuracy"].values
#     symm_precision_1 = load_csv(symm_precision_file1)["0_156 - symmetry/test_accuracy"].values
#     symm_recall_1 = load_csv(symm_recall_file1)["0_156 - symmetry/test_accuracy"].values
#     merged_symm_json = {}
#     for i, name in enumerate(symm_names_1):
#         if i < len(symm_acc_1):
#             merged_symm_json[name] = {
#                 "accuracy": float(symm_acc_1[i]),
#                 "f1_score": float(symm_f1_1[i]),
#                 "precision": float(symm_precision_1[i]),
#                 "recall": float(symm_recall_1[i])
#             }
#         else:
#             raise NotImplementedError(f"Index {i} out of range for symm_acc_1 with length {len(symm_acc_1)}")
#         print(f"Pattern: {name}, Accuracy: {symm_acc_1[i]:.4f}, F1 Score: {symm_f1_1[i]:.4f}, Precision: {symm_precision_1[i]:.4f}, Recall: {symm_recall_1[i]:.4f}")
#     save_json(merged_symm_json, res_path / "symmetry" / "gpt5_symmetry_merged_results.json")
#

# def merge_cont(res_path):
#     cont_names_1 = sorted(get_all_subfolder_names(config.get_raw_patterns_path() / "res_200_pin_False" / "continuity" / "train"))
#
#     cont_acc_file1 = res_path / "continuity" / "wandb_export_0_147_acc.csv"
#     cont_f1_file1 = res_path / "continuity" / "wandb_export_0_147_f1_score.csv"
#     cont_precision_file1 = res_path / "continuity" / "wandb_export_0_147_precision.csv"
#     cont_recall_file1 = res_path / "continuity" / "wandb_export_0_147_recall.csv"
#     cont_acc_1 = load_csv(cont_acc_file1)["0_147 - continuity/test_accuracy"].values
#     cont_f1_1 = load_csv(cont_f1_file1)["0_147 - continuity/test_accuracy"].values
#     cont_precision_1 = load_csv(cont_precision_file1)["0_147 - continuity/test_accuracy"].values
#     cont_recall_1 = load_csv(cont_recall_file1)["0_147 - continuity/test_accuracy"].values
#     merged_cont_json = {}
#     for i, name in enumerate(cont_names_1):
#         if i < len(cont_acc_1):
#             merged_cont_json[name] = {
#                 "accuracy": float(cont_acc_1[i]),
#                 "f1_score": float(cont_f1_1[i]),
#                 "precision": float(cont_precision_1[i]),
#                 "recall": float(cont_recall_1[i])
#             }
#         else:
#             raise NotImplementedError(f"Index {i} out of range for cont_acc_1 with length {len(cont_acc_1)}")
#         print(f"Pattern: {name}, Accuracy: {cont_acc_1[i]:.4f}, F1 Score: {cont_f1_1[i]:.4f}, Precision: {cont_precision_1[i]:.4f}, Recall: {cont_recall_1[i]:.4f}")
#     save_json(merged_cont_json, res_path / "continuity" / "gpt5_continuity_merged_results.json")
#

def merge_closure(res_path):
    principle = "closure"
    cont_names_1 = sorted(get_all_subfolder_names(config.get_raw_patterns_path() / "res_200_pin_False" / principle / "train"))

    file_1_end = 556

    acc_file1 = res_path / principle / f"{principle}_0_{file_1_end}_acc.csv"
    cont_f1_file1 = res_path / principle / f"{principle}_0_{file_1_end}_f1.csv"
    precision_file1 = res_path / principle / f"{principle}_0_{file_1_end}_precision.csv"
    recall_file1 = res_path / principle / f"{principle}_0_{file_1_end}_recall.csv"

    acc_1 = load_csv(acc_file1)[f"0_{file_1_end} - {principle}/test_accuracy"].values
    f1_1 = load_csv(cont_f1_file1)[f"0_{file_1_end} - {principle}/f1_score"].values
    precision_1 = load_csv(precision_file1)[f"0_{file_1_end} - {principle}/precision"].values
    recall_1 = load_csv(recall_file1)[f"0_{file_1_end} - {principle}/recall"].values

    file_2_start = "557"
    file_2_end = "end"
    acc_file2 = res_path / principle / f"{principle}_{file_2_start}_{file_2_end}_acc.csv"
    cont_f1_file2 = res_path / principle / f"{principle}_{file_2_start}_{file_2_end}_f1.csv"
    precision_file2 = res_path / principle / f"{principle}_{file_2_start}_{file_2_end}_precision.csv"
    recall_file2 = res_path / principle / f"{principle}_{file_2_start}_{file_2_end}_recall.csv"

    acc_2 = load_csv(acc_file2)[f"{file_2_start}_{file_2_end} - {principle}/test_accuracy"].values
    f1_2 = load_csv(cont_f1_file2)[f"{file_2_start}_{file_2_end} - {principle}/f1_score"].values
    precision_2 = load_csv(precision_file2)[f"{file_2_start}_{file_2_end} - {principle}/precision"].values
    recall_2 = load_csv(recall_file2)[f"{file_2_start}_{file_2_end} - {principle}/recall"].values

    merged_cont_json = {}
    for i, name in enumerate(cont_names_1):
        if i < len(acc_1):
            merged_cont_json[name] = {
                "accuracy": float(acc_1[i]),
                "f1_score": float(f1_1[i]),
                "precision": float(precision_1[i]),
                "recall": float(recall_1[i])
            }
            print(f"Pattern: {name}, Accuracy: {acc_1[i]:.4f}, F1 Score: {f1_1[i]:.4f}, Precision: {precision_1[i]:.4f}, Recall: {recall_1[i]:.4f}")
        elif i < len(acc_1) + len(acc_2):
            j = i - len(acc_1)
            merged_cont_json[name] = {
                "accuracy": float(acc_2[j]),
                "f1_score": float(f1_2[j]),
                "precision": float(precision_2[j]),
                "recall": float(recall_2[j])
            }
            print(f"Pattern: {name}, Accuracy: {acc_2[j]:.4f}, F1 Score: {f1_2[j]:.4f}, "
                  f"Precision: {precision_2[j]:.4f}, Recall: {recall_2[j]:.4f}")
        else:
            raise NotImplementedError(f"Index {i} out of range for cont_acc_1 with length {len(acc_1)}")

    save_json(merged_cont_json, res_path / principle / f"gpt5_{principle}_merged_results.json")


def merge_cont(res_path):
    principle = "continuity"
    cont_names_1 = sorted(get_all_subfolder_names(config.get_raw_patterns_path() / "res_200_pin_False" / principle / "train"))

    file_1_end = 147

    acc_file1 = res_path / principle / f"{principle}_0_{file_1_end}_acc.csv"
    cont_f1_file1 = res_path / principle / f"{principle}_0_{file_1_end}_f1_score.csv"
    precision_file1 = res_path / principle / f"{principle}_0_{file_1_end}_precision.csv"
    recall_file1 = res_path / principle / f"{principle}_0_{file_1_end}_recall.csv"

    acc_1 = load_csv(acc_file1)[f"0_{file_1_end} - {principle}/test_accuracy"].values
    f1_1 = load_csv(cont_f1_file1)[f"0_{file_1_end} - {principle}/f1_score"].values
    precision_1 = load_csv(precision_file1)[f"0_{file_1_end} - {principle}/precision"].values
    recall_1 = load_csv(recall_file1)[f"0_{file_1_end} - {principle}/recall"].values

    file_2_start = "148"
    file_2_end = "end"
    acc_file2 = res_path / principle / f"{principle}_{file_2_start}_{file_2_end}_acc.csv"
    cont_f1_file2 = res_path / principle / f"{principle}_{file_2_start}_{file_2_end}_f1_score.csv"
    precision_file2 = res_path / principle / f"{principle}_{file_2_start}_{file_2_end}_precision.csv"
    recall_file2 = res_path / principle / f"{principle}_{file_2_start}_{file_2_end}_recall.csv"

    acc_2 = load_csv(acc_file2)[f"{file_2_start}_{file_2_end} - {principle}/test_accuracy"].values
    f1_2 = load_csv(cont_f1_file2)[f"{file_2_start}_{file_2_end} - {principle}/f1_score"].values
    precision_2 = load_csv(precision_file2)[f"{file_2_start}_{file_2_end} - {principle}/precision"].values
    recall_2 = load_csv(recall_file2)[f"{file_2_start}_{file_2_end} - {principle}/recall"].values

    merged_cont_json = {}
    for i, name in enumerate(cont_names_1):
        if i < len(acc_1):
            merged_cont_json[name] = {
                "accuracy": float(acc_1[i]),
                "f1_score": float(f1_1[i]),
                "precision": float(precision_1[i]),
                "recall": float(recall_1[i])
            }
            print(f"Pattern: {name}, Accuracy: {acc_1[i]:.4f}, F1 Score: {f1_1[i]:.4f}, Precision: {precision_1[i]:.4f}, Recall: {recall_1[i]:.4f}")
        elif i < len(acc_1) + len(acc_2):
            j = i - len(acc_1)
            merged_cont_json[name] = {
                "accuracy": float(acc_2[j]),
                "f1_score": float(f1_2[j]),
                "precision": float(precision_2[j]),
                "recall": float(recall_2[j])
            }
            print(f"Pattern: {name}, Accuracy: {acc_2[j]:.4f}, F1 Score: {f1_2[j]:.4f}, "
                  f"Precision: {precision_2[j]:.4f}, Recall: {recall_2[j]:.4f}")
        else:
            raise NotImplementedError(f"Index {i} out of range for cont_acc_1 with length {len(acc_1)}")

    save_json(merged_cont_json, res_path / principle / f"gpt5_{principle}_merged_results.json")


def merge_symm(res_path):
    principle = "symmetry"
    cont_names = sorted(get_all_subfolder_names(config.get_raw_patterns_path() / "res_200_pin_False" / principle / "train"))

    # file_1_end = 147
    file_lists = [
        [0, 156],
        [157, 380],
        [378, 456],
        [457, 500],
        [500, 600],
        [600, 609],
        [610,700],
        [700, 705],
        [706, 800],
        [800, 900],
    ]
    merged_cont_json = {}
    for start, end in file_lists:
        acc_1 = load_csv(res_path / principle / f"{principle}_{start}_{end}_acc.csv")[f"{start}_{end} - {principle}/test_accuracy"].values
        f1_1 = load_csv(res_path / principle / f"{principle}_{start}_{end}_f1_score.csv")[f"{start}_{end} - {principle}/f1_score"].values
        precision_1 = load_csv(res_path / principle / f"{principle}_{start}_{end}_precision.csv")[f"{start}_{end} - {principle}/precision"].values
        recall_1 = load_csv(res_path / principle / f"{principle}_{start}_{end}_recall.csv")[f"{start}_{end} - {principle}/recall"].values

        for i in range(len(acc_1)):
            merged_cont_json[cont_names[i + start]] = {
                "accuracy": float(acc_1[i]),
                "f1_score": float(f1_1[i]),
                "precision": float(precision_1[i]),
                "recall": float(recall_1[i])
            }
            # print(f"Pattern: {cont_names[i+start]}, Accuracy: {acc_1[i]:.4f}, F1 Score: {f1_1[i]:.4f}, Precision: {precision_1[i]:.4f}, Recall: {recall_1[i]:.4f}")
    print("Final length of merged results:", len(merged_cont_json))
    save_json(merged_cont_json, res_path / principle / f"gpt5_{principle}_merged_results.json")


def merge_similarity(res_path):
    principle = "similarity"
    cont_names = sorted(get_all_subfolder_names(config.get_raw_patterns_path() / "res_200_pin_False" / principle / "train"))
    file_lists = [
        [0, 526],
        [527, 622],
        [622, 700],
        [700,800],
        [800,872],
    ]
    merged_cont_json = {}
    for start, end in file_lists:
        acc_1 = load_csv(res_path / principle / f"{principle}_{start}_{end}_acc.csv")[f"{start}_{end} - {principle}/test_accuracy"].values
        f1_1 = load_csv(res_path / principle / f"{principle}_{start}_{end}_f1_score.csv")[f"{start}_{end} - {principle}/f1_score"].values
        precision_1 = load_csv(res_path / principle / f"{principle}_{start}_{end}_precision.csv")[f"{start}_{end} - {principle}/precision"].values
        recall_1 = load_csv(res_path / principle / f"{principle}_{start}_{end}_recall.csv")[f"{start}_{end} - {principle}/recall"].values

        for i in range(len(acc_1)):
            merged_cont_json[cont_names[i + start]] = {
                "accuracy": float(acc_1[i]),
                "f1_score": float(f1_1[i]),
                "precision": float(precision_1[i]),
                "recall": float(recall_1[i])
            }
            # print(f"Pattern: {cont_names[i+start]}, Accuracy: {acc_1[i]:.4f}, F1 Score: {f1_1[i]:.4f}, Precision: {precision_1[i]:.4f}, Recall: {recall_1[i]:.4f}")
    print("Final length of merged results:", len(merged_cont_json))
    save_json(merged_cont_json, res_path / principle / f"gpt5_{principle}_merged_results.json")


if __name__ == "__main__":
    res_path = config.result_path
    # merge_cont(res_path)
    # merge_closure(res_path)
    # merge_symm(res_path)
    # merge_proximity(res_path)
    # merge_similarity(res_path)

    print("program finished.")
