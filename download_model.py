import os
import sys
from huggingface_hub import hf_hub_download

MODELS = {
    "1": {
        "name": "DeepSeek-R1-Distill-Qwen-14B (Reasoning / Best for Hard Coding)",
        "repo_id": "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        "filename": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
        "size_gb": "~9.0 GB"
    },
    "2": {
        "name": "Mistral-Small-24B-Instruct (24B Dense Knowledge / Architecture)",
        "repo_id": "bartowski/mistral-small-24b-instruct-2501-GGUF",
        "filename": "mistral-small-24b-instruct-2501-Q3_K_M.gguf",
        "size_gb": "~11.2 GB"
    },
    "3": {
        "name": "Codestral-22B-v0.1 (Mistral Code & FIM Autocomplete Specialist)",
        "repo_id": "bartowski/Codestral-22B-v0.1-GGUF",
        "filename": "Codestral-22B-v0.1-Q3_K_M.gguf",
        "size_gb": "~10.6 GB"
    },
    "4": {
        "name": "Qwen2.5-Coder-14B-Instruct",
        "repo_id": "bartowski/Qwen2.5-Coder-14B-Instruct-GGUF",
        "filename": "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
        "size_gb": "~9.0 GB"
    }
}

def main():
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("      Select a Coding Model to Download for RTX 5000 (16GB VRAM)")
    print("=" * 60)
    for key, info in MODELS.items():
        print(f"[{key}] {info['name']} ({info['size_gb']})")
    print("=" * 60)

    choice = sys.argv[1] if len(sys.argv) > 1 else input("Enter choice (1-4) [default: 1]: ").strip() or "1"

    if choice not in MODELS:
        print(f"Invalid choice '{choice}'. Defaulting to 1.")
        choice = "1"

    selected = MODELS[choice]
    print(f"\nDownloading {selected['name']}...")
    print(f"Repo: {selected['repo_id']}")
    print(f"File: {selected['filename']}")
    print(f"Destination: {models_dir}\n")

    file_path = hf_hub_download(
        repo_id=selected["repo_id"],
        filename=selected["filename"],
        local_dir=models_dir,
        local_dir_use_symlinks=False
    )

    print("\n" + "=" * 60)
    print(f" Download completed successfully!")
    print(f"Model saved to: {file_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
