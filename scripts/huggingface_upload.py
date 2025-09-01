# Created by MacBook Pro at 20.06.25
from huggingface_hub import HfApi, HfFolder, Repository
import os
import shutil
from pathlib import Path
import argparse


from scripts import config

def upload_to_huggingface(args):
    # ==== CONFIG ====
    username = "akweury"
    repo_name = "ELVIS"
    if args.remote:
        source_folder = f"/home/ml-jsha/storage/ELVIS_Data/res_{args.resolution}_pin_False"
    else:
        source_folder = Path(f"/Users/jing/PycharmProjects/ELVIS/gen_data/res_{args.resolution}_pin_False")

    upload_folder = Path("./upload_repo")  # local working dir
    # ===============
    # Full repo ID
    repo_id = f"{username}/{repo_name}"
    # Cleanup old repo if exists
    if upload_folder.exists():
        shutil.rmtree(upload_folder)
    # Create the HF repo if not already
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    # Clone the repo locally
    repo = Repository(local_dir=str(upload_folder), clone_from=repo_id, repo_type="dataset")
    # Copy source folder into cloned repo
    shutil.copytree(source_folder, upload_folder / "closure")
    # Commit and push
    repo.push_to_hub(commit_message=f"Upload (res_{args.resolution}, pin_False)")
    print(f"✅ Uploaded successfully to https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline models with CUDA support.")
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--resolution", type=int, default=224, choices=[224, 448, 1024])
    args = parser.parse_args()
    upload_to_huggingface(args)


if __name__ == "__main__":
    main()
