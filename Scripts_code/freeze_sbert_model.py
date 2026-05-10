from pathlib import Path
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_MODEL_DIR = (
    PROJECT_ROOT
    / "Automated Deception Classifier (Projectfolder)"
    / "local_models"
    / "all-MiniLM-L6-v2"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"Downloading and saving SBERT model to:\n{LOCAL_MODEL_DIR}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

tokenizer.save_pretrained(LOCAL_MODEL_DIR)
model.save_pretrained(LOCAL_MODEL_DIR)

print("\nDone.")
print(f"Frozen local SBERT path:\n{LOCAL_MODEL_DIR}")