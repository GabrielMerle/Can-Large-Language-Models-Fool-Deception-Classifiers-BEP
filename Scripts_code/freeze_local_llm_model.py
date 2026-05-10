from __future__ import annotations

# File-level purpose:
# - one-time helper for freezing the fixed local paraphraser model
# - saves Qwen/Qwen3-4B-Instruct-2507 into the project local_models folder
# - normal attack runs should then load this path with local_files_only=True

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
LOCAL_MODEL_DIR = (
    PROJECT_ROOT
    / "Automated Deception Classifier (Projectfolder)"
    / "local_models"
    / "Qwen3-4B-Instruct-2507"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and freeze the fixed local Qwen paraphraser model."
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help="Hugging Face model id to freeze. Default is the BEP fixed paraphraser.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(LOCAL_MODEL_DIR),
        help="Local directory where tokenizer and model files will be saved.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow saving into an existing non-empty output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            "Local LLM output directory already exists and is not empty:\n"
            f"{output_dir}\n"
            "Use --overwrite only if you intentionally want to refresh the frozen copy."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading fixed local paraphraser model:\n{args.model_id}")
    print(f"\nSaving to:\n{output_dir}")

    snapshot_download(
        repo_id=args.model_id,
        local_dir=output_dir,
        force_download=args.overwrite,
    )

    # Lightweight sanity check: verifies the frozen directory contains a usable
    # tokenizer without loading the multi-GB model weights into memory.
    AutoTokenizer.from_pretrained(output_dir, local_files_only=True)

    metadata = {
        "model_id": args.model_id,
        "saved_to": str(output_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Fixed local LLM paraphraser for BEP pilot attack script.",
    }
    metadata_path = output_dir / "freeze_meta.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nDone.")
    print(f"Frozen local Qwen path:\n{output_dir}")
    print(f"Metadata:\n{metadata_path}")


if __name__ == "__main__":
    main()
