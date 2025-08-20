# Created by MacBook Pro at 19.08.25

import os
import numpy as np
import cv2
import argparse
from scripts import config
from pathlib import Path

def get_all_task_folders(data_folder):
    all_task_folders = []
    for root, dirs, files in os.walk(data_folder):
        for dir_name in dirs:
            if "rel" in dir_name:
                all_task_folders.append(os.path.join(root, dir_name))
    return all_task_folders


def load_json(file_path):
    import json
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def main(args):
    statistic_data = {
        "total_tasks": 0,
        "total_json_files": 0,
        "positive_json_files": 0,
        "negative_json_files": 0,
        "objs_per_img": [],
    }

    data_folder = config.get_raw_patterns_path(args.remote) / f"res_{config.img_width}_pin_{config.prin_in_neg}"/ args.principle / "train"
    all_task_folders = get_all_task_folders(data_folder)
    # iterate through all folders in the data folder
    for task_folder in all_task_folders:
        all_json_files = []
        # get all json files in the task folder
        pos_json_files = list(Path(task_folder).glob("positive/*.json"))
        neg_json_files = list(Path(task_folder).glob("negative/*.json"))
        all_json_files.extend(pos_json_files)
        all_json_files.extend(neg_json_files)

        for json_file in all_json_files:

            # load the JSON file and process it
            data = load_json(json_file)
            statistic_data["objs_per_img"].append(len(data["img_data"]))

            # Here you can add code to process each JSON file
            # For example, you might want to load the JSON data and analyze it
            # with open(json_file, 'r') as f:
            #     json_data = json.load(f)
            #     # Process json_data as needed
        # Here you can add code to process each task folder
        # For example, you might want to load images, analyze data, etc.
        # data_path = os.path.join(task_folder, "data.json")
        # if os.path.exists(data_path):
        #     with open(data_path, 'r') as f:
        #         json_data = json.load(f)
        #     # Process json_data as needed
        # else:
        #     print(f"No data file found in {task_folder}")





    print("Statistics:")
    print(f"Maximum number of objects per image: {max(statistic_data['objs_per_img'])}")
    print(f"Minimum number of objects per image: {min(statistic_data['objs_per_img'])}")
    print(f"Average number of objects per image: {np.mean(statistic_data['objs_per_img'])}")
    print(f"Total number of tasks: {len(all_task_folders)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline models with CUDA support.")
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--principle", type=str, default="similarity")
    parser.add_argument("--labelOn", action="store_true", help="Show labels on the generated images.")
    args = parser.parse_args()

    main(args)
    # draw_club_playground()
