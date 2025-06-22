# Created by MacBook Pro at 20.06.25
from huggingface_hub import HfApi, HfFolder, Repository
import os
import shutil
from pathlib import Path


from scripts import config

# ==== CONFIG ====
username = "akweury"  # change this!
repo_name = "ELVIS"
source_folder = Path("/Users/jing/PycharmProjects/ELVIS/data/raw_patterns/res_1024_pin_False")
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
repo.push_to_hub(commit_message="Upload closure (res_1024, pin_False)")

print(f"✅ Uploaded successfully to https://huggingface.co/datasets/{repo_id}")