#!/usr/bin/env python3
"""Publish the converted MLX models to HuggingFace under mlx-community."""
import sys
from huggingface_hub import HfApi

REPOS = [
    ("AnandSingh/LightOnOCR-3-4B-bf16", "LightOnOCR-3-4B-mlx/bf16"),
    ("AnandSingh/LightOnOCR-3-4B-4bit", "LightOnOCR-3-4B-mlx/4bit"),
]


def main():
    api = HfApi()
    me = api.whoami()
    print(f"[publish] authenticated as: {me.get('name')}")
    for repo_id, folder in REPOS:
        print(f"[publish] creating repo {repo_id} ...")
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        url = api.upload_folder(
            folder_path=folder,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add LightOnOCR-3-4B converted to MLX (via mlx-vlm)",
        )
        print(f"[publish] uploaded {folder} -> {url}")
    print("[publish] ALL DONE")


if __name__ == "__main__":
    sys.exit(main())
