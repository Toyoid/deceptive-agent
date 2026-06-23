import os
from huggingface_hub import HfApi

def upload_local_model():
    api = HfApi()

    # Provide your Hugging Face username and the desired repository name "username/repo_name"
    repo_id = "hahnli/Qwen3-8B-Search-Self-Monitor-SFT"
    
    # Path to the exact folder containing the model
    folder_path = "/devsft_AFS/hanxiaoli/deceptive-agent/checkpoints/self_monitor_sft/qwen3_8b_cheating_search_agent/global_step_954"

    print(f"Ensuring repository '{repo_id}' exists...")
    # create the repo if it doesn't exist already.
    # Set private=False if you want it to be public immediately.
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)

    print(f"Uploading all model shards and configs from '{folder_path}' to '{repo_id}'...")
    
    # push to the hub (handles multi-file and large payloads)
    api.upload_large_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model"
    )
    
    print(f"Success! Model is now live at: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    upload_local_model()