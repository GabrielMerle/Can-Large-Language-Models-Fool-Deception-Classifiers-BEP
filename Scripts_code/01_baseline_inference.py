from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "Automated Deception Classifier (Projectfolder)"
MODEL_DIR = DATA_ROOT / "DistilBERT"
OUTPUT_DIR = PROJECT_ROOT / "Scripts_code" / "outputs"

DATASETS = {
    "train_dev": {
        "role": "development/pilot tuning only",
        "path": DATA_ROOT / "hippocorpus_training_truncated.csv",
        "baseline_out": OUTPUT_DIR / "baseline_predictions_train_dev.csv",
        "candidate_out": OUTPUT_DIR / "candidate_pool_train_dev.csv",
        "summary_out": OUTPUT_DIR / "baseline_summary_train_dev.json",
    },
    "test_final": {
        "role": "final main experiment only",
        "path": DATA_ROOT / "hippocorpus_test_truncated.csv",
        "baseline_out": OUTPUT_DIR / "baseline_predictions_test_final.csv",
        "candidate_out": OUTPUT_DIR / "candidate_pool_test_final.csv",
        "summary_out": OUTPUT_DIR / "baseline_summary_test_final.json",
    },
}

LEGACY_TEST_ALIASES = {
    "baseline_out": OUTPUT_DIR / "baseline_predictions.csv",
    "candidate_out": OUTPUT_DIR / "candidate_pool.csv",
    "summary_out": OUTPUT_DIR / "baseline_summary.json",
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Config
# =========================
BATCH_SIZE = 16
MAX_LENGTH = 512
DEVICE = "cpu"

# Canonical text field for both baseline inference and later attacks.
TEXT_COLUMN = "text_truncated"
FULL_TEXT_COLUMN = "text"
LABEL_COLUMN = "condition"
SOURCE_INDEX_COLUMN = "index"
PAIR_ID_COLUMN = "truth-dec_pairId"

LABEL_TO_ID = {
    "truthful": 0,
    "deceptive": 1,
}
ID_TO_LABEL = {
    0: "truthful",
    1: "deceptive",
}


def normalize_label_value(value) -> int:
    if pd.isna(value):
        raise ValueError("Found missing label value.")

    if isinstance(value, str):
        v = value.strip().lower()
        if v in LABEL_TO_ID:
            return LABEL_TO_ID[v]
        if v in {"0", "false", "true", "non-deceptive", "honest"}:
            return 0
        if v in {"1", "lie", "lying", "fake"}:
            return 1

    try:
        v = int(value)
        if v in ID_TO_LABEL:
            return v
    except Exception:
        pass

    raise ValueError(f"Could not normalize label value: {value}")


def get_label_mapping(model) -> Dict[int, str]:
    num_labels = int(getattr(model.config, "num_labels", len(ID_TO_LABEL)))
    expected_ids = set(range(num_labels))
    configured_ids = set(ID_TO_LABEL)

    if configured_ids != expected_ids:
        raise ValueError(
            "ID_TO_LABEL must match the model output labels. "
            f"Model label ids: {sorted(expected_ids)}; "
            f"configured label ids: {sorted(configured_ids)}"
        )

    return dict(ID_TO_LABEL)


def validate_expected_columns(df: pd.DataFrame, dataset_name: str) -> None:
    required = [TEXT_COLUMN, FULL_TEXT_COLUMN, LABEL_COLUMN]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}. "
            f"Columns found: {list(df.columns)}"
        )


def compute_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_dataframe(df: pd.DataFrame, dataset_name: str) -> None:
    missing_labels = int(df[LABEL_COLUMN].isna().sum())
    if missing_labels > 0:
        raise ValueError(f"{dataset_name}: found {missing_labels} missing labels.")

    empty_texts = int(df["attack_text"].eq("").sum())
    if empty_texts > 0:
        raise ValueError(f"{dataset_name}: found {empty_texts} empty attack texts.")

    duplicate_row_ids = int(df["row_id"].duplicated().sum())
    if duplicate_row_ids > 0:
        raise ValueError(f"{dataset_name}: found {duplicate_row_ids} duplicate row ids.")


def validate_probability_vectors(
    prob_vectors: List[List[float]],
    tolerance: float = 1e-6,
) -> None:
    for idx, vector in enumerate(prob_vectors):
        total = sum(vector)
        if abs(total - 1.0) > tolerance:
            raise ValueError(
                f"Probability vector at index {idx} sums to {total}, expected 1.0."
            )


def build_confusion_counts(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    labels = sorted(set(df["gold_label"]).union(set(df["pred_label"])))
    matrix: Dict[str, Dict[str, int]] = {}

    for gold in labels:
        row: Dict[str, int] = {}
        for pred in labels:
            count = int(((df["gold_label"] == gold) & (df["pred_label"] == pred)).sum())
            row[str(pred)] = count
        matrix[str(gold)] = row

    return matrix


def build_per_class_results(
    df: pd.DataFrame,
    label_mapping: Dict[int, str],
) -> Dict[str, Dict[str, float | int | str]]:
    results: Dict[str, Dict[str, float | int | str]] = {}

    for label_id in sorted(df["gold_label"].unique()):
        gold_mask = df["gold_label"] == label_id
        total = int(gold_mask.sum())
        correct = int(((df["gold_label"] == label_id) & (df["pred_label"] == label_id)).sum())
        accuracy = float(correct / total) if total else 0.0
        results[str(label_id)] = {
            "label_name": label_mapping.get(label_id, f"LABEL_{label_id}"),
            "gold_count": total,
            "correct_count": correct,
            "accuracy": accuracy,
        }

    return results


def batch_predict(texts, tokenizer, model):
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    encodings = {k: v.to(DEVICE) for k, v in encodings.items()}

    with torch.inference_mode():
        outputs = model(**encodings)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1)
        confidences = torch.max(probs, dim=-1).values

    return pred_ids.cpu().tolist(), confidences.cpu().tolist(), probs.cpu().tolist()


def prepare_dataset(df: pd.DataFrame, dataset_name: str, data_path: Path) -> pd.DataFrame:
    validate_expected_columns(df, dataset_name)

    df = df.copy()
    df["source_split"] = dataset_name
    df["source_dataset_path"] = str(data_path)
    df["source_text_column"] = TEXT_COLUMN
    df["row_id"] = range(len(df))
    df["source_index"] = df[SOURCE_INDEX_COLUMN] if SOURCE_INDEX_COLUMN in df.columns else df["row_id"]
    df["truth_dec_pair_id"] = df[PAIR_ID_COLUMN] if PAIR_ID_COLUMN in df.columns else ""

    df["condition"] = df[LABEL_COLUMN].fillna("").astype(str).str.strip().str.lower()
    df["gold_label"] = df[LABEL_COLUMN].apply(normalize_label_value)
    df["gold_label_name"] = df["gold_label"].map(ID_TO_LABEL)
    df["attack_text"] = df[TEXT_COLUMN].fillna("").astype(str).str.strip()
    df["full_text"] = df[FULL_TEXT_COLUMN].fillna("").astype(str).str.strip()
    df["text_hash"] = df["attack_text"].apply(compute_text_hash)

    validate_dataframe(df, dataset_name)
    return df


def run_dataset(
    dataset_name: str,
    dataset_config: dict,
    tokenizer,
    model,
    label_mapping: Dict[int, str],
) -> None:
    data_path = dataset_config["path"]
    print(f"\n=== {dataset_name} ===")
    print(f"Role: {dataset_config['role']}")
    print(f"Data path: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows.")

    df = prepare_dataset(df, dataset_name, data_path)

    all_pred_ids = []
    all_confidences = []
    all_prob_vectors = []
    texts = df["attack_text"].tolist()

    print("Running inference...")
    for start in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[start:start + BATCH_SIZE]
        pred_ids, confidences, prob_vectors = batch_predict(
            batch_texts,
            tokenizer,
            model,
        )
        all_pred_ids.extend(pred_ids)
        all_confidences.extend(confidences)
        all_prob_vectors.extend(prob_vectors)

    validate_probability_vectors(all_prob_vectors)

    df["pred_label"] = all_pred_ids
    df["pred_label_name"] = df["pred_label"].map(label_mapping)
    df["confidence"] = all_confidences
    df["correct"] = (df["gold_label"] == df["pred_label"]).astype(int)
    df["prob_vector"] = [json.dumps(v) for v in all_prob_vectors]

    accuracy = float(df["correct"].mean())
    candidate_df_before_dedup = df[df["correct"] == 1].copy()
    num_candidate_duplicates_dropped = int(
        candidate_df_before_dedup["text_hash"].duplicated().sum()
    )
    candidate_df = (
        candidate_df_before_dedup
        .drop_duplicates(subset=["text_hash"], keep="first")
        .copy()
    )
    confusion_matrix = build_confusion_counts(df)
    per_class_results = build_per_class_results(df, label_mapping)

    keep_cols = [
        "source_split",
        "source_dataset_path",
        "source_text_column",
        "row_id",
        "source_index",
        "truth_dec_pair_id",
        "text_hash",
        "attack_text",
        "full_text",
        "condition",
        "gold_label",
        "gold_label_name",
        "pred_label",
        "pred_label_name",
        "confidence",
        "correct",
        "prob_vector",
    ]

    baseline_out = dataset_config["baseline_out"]
    candidate_out = dataset_config["candidate_out"]
    summary_out = dataset_config["summary_out"]

    df[keep_cols].to_csv(baseline_out, index=False)
    candidate_df[keep_cols].to_csv(candidate_out, index=False)

    summary = {
        "dataset_name": dataset_name,
        "dataset_role": dataset_config["role"],
        "dataset_path": str(data_path),
        "dataset_sha256": compute_file_hash(data_path),
        "model_dir": str(MODEL_DIR),
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "source_text_column": TEXT_COLUMN,
        "num_rows": int(len(df)),
        "accuracy": accuracy,
        "label_column": LABEL_COLUMN,
        "label_mapping": {str(k): v for k, v in label_mapping.items()},
        "gold_counts": {
            str(k): int(v)
            for k, v in df["gold_label"].value_counts().sort_index().items()
        },
        "pred_counts": {
            str(k): int(v)
            for k, v in df["pred_label"].value_counts().sort_index().items()
        },
        "num_correct_before_dedup": int(candidate_df_before_dedup.shape[0]),
        "num_candidate_duplicates_dropped": num_candidate_duplicates_dropped,
        "num_correct": int(candidate_df.shape[0]),
        "candidate_pool_path": str(candidate_out),
        "per_class_results": per_class_results,
        "confusion_matrix": confusion_matrix,
    }

    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if dataset_name == "test_final":
        df[keep_cols].to_csv(LEGACY_TEST_ALIASES["baseline_out"], index=False)
        candidate_df[keep_cols].to_csv(LEGACY_TEST_ALIASES["candidate_out"], index=False)
        with open(LEGACY_TEST_ALIASES["summary_out"], "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print(f"Accuracy: {accuracy:.4f}")
    print("Gold label counts:")
    print(df["gold_label_name"].value_counts(dropna=False).sort_index())
    print("Predicted label counts:")
    print(df["pred_label_name"].value_counts(dropna=False).sort_index())
    print(f"Correctly classified rows before dedup: {len(candidate_df_before_dedup)}")
    print(f"Candidate duplicate text hashes dropped: {num_candidate_duplicates_dropped}")
    print(f"Candidate pool rows: {len(candidate_df)}")
    print(f"Saved baseline predictions to: {baseline_out}")
    print(f"Saved candidate pool to:      {candidate_out}")
    print(f"Saved summary to:             {summary_out}")


def main() -> None:
    print(f"Using device: {DEVICE}")
    print(f"Model dir: {MODEL_DIR}")
    print(f"Canonical text column: {TEXT_COLUMN}")

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )
    model.to(DEVICE)
    model.eval()

    label_mapping = get_label_mapping(model)
    print(f"Fixed label mapping: {label_mapping}")

    for dataset_name, dataset_config in DATASETS.items():
        run_dataset(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            tokenizer=tokenizer,
            model=model,
            label_mapping=label_mapping,
        )


if __name__ == "__main__":
    main()
