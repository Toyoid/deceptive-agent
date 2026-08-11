import os
from huggingface_hub import HfApi

def upload_local_model():
    api = HfApi()

    # Provide your Hugging Face username and the desired repository name "username/repo_name"
    repo_id = "hahnli/Qwen3-8B-AgenticRoles-SelfMonitor-RL"
    
    # Path to the exact folder containing the model
    folder_path = "/devsft_AFS/hanxiaoli/deceptive-agent/checkpoints/verl_deceptive_roles/grpo_qwen3_8b_self_monitor/global_step_140/actor/huggingface"

    print(f"Ensuring repository '{repo_id}' exists...")
    # create the repo if it doesn't exist already.
    # Set private=False if you want it to be public immediately.
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)

    # Explicitly force existing repo to public.
    api.update_repo_visibility(
        repo_id=repo_id,
        repo_type="model",
        private=False,
    )

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


# cat > /devsft_AFS/hanxiaoli/deceptive-agent/checkpoints/verl_deceptive_search/grpo_deceptive_search_verdict_monitor_qwen3_4b/global_step_120/actor/huggingface/README.md <<'EOF'
# ---
# base model: Qwen/Qwen3-4B
# training task: Search-augmented QA 
# RL method: GRPO
# ---
# This is a research checkpoint trained on Qwen3-4B with the SearchQA task using Reinforcement Learning with Verdict Monitor (RLVM).
# EOF