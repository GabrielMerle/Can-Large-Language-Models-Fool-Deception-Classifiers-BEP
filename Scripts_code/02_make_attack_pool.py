from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "Scripts_code" / "outputs"

TRAIN_DEV_POOL_PATH = OUTPUT_DIR / "candidate_pool_train_dev.csv"
TEST_FINAL_POOL_PATH = OUTPUT_DIR / "candidate_pool_test_final.csv"

PILOT_OUT = OUTPUT_DIR / "candidate_pool_pilot_20.csv"
MAIN_OUT = OUTPUT_DIR / "candidate_pool_main_160.csv"
META_OUT = OUTPUT_DIR / "candidate_pool_freeze_meta.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Config
# =========================
SEED = 42

SOURCE_SPLIT_COLUMN = "source_split"
SOURCE_DATASET_COLUMN = "source_dataset_path"
SOURCE_TEXT_COLUMN = "source_text_column"
ROW_ID_COLUMN = "row_id"
SOURCE_INDEX_COLUMN = "source_index"
TEXT_HASH_COLUMN = "text_hash"
ATTACK_TEXT_COLUMN = "attack_text"
LABEL_COLUMN = "condition"
GOLD_LABEL_COLUMN = "gold_label"
GOLD_LABEL_NAME_COLUMN = "gold_label_name"
PRED_LABEL_COLUMN = "pred_label"
PRED_LABEL_NAME_COLUMN = "pred_label_name"
CORRECT_COLUMN = "correct"

TRUTHFUL_VALUE = "truthful"
DECEPTIVE_VALUE = "deceptive"

PILOT_PER_CLASS = 10
MAIN_PER_CLASS = 80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze non-overlapping attack pools: pilot from train/dev and "
            "main from final test."
        )
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing frozen pool outputs.",
    )
    return parser.parse_args()


def compute_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_output_paths(overwrite: bool) -> None:
    existing = [path for path in [PILOT_OUT, MAIN_OUT, META_OUT] if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(
            "Frozen pool outputs already exist. Refusing to overwrite them.\n"
            f"{formatted}\n"
            "Rerun with --overwrite only when you intentionally want to refreeze."
        )


def validate_candidate_pool(
    df: pd.DataFrame,
    pool_path: Path,
    expected_source_split: str,
) -> None:
    required_columns = {
        SOURCE_SPLIT_COLUMN,
        SOURCE_DATASET_COLUMN,
        SOURCE_TEXT_COLUMN,
        ROW_ID_COLUMN,
        SOURCE_INDEX_COLUMN,
        TEXT_HASH_COLUMN,
        ATTACK_TEXT_COLUMN,
        LABEL_COLUMN,
        GOLD_LABEL_COLUMN,
        GOLD_LABEL_NAME_COLUMN,
        PRED_LABEL_COLUMN,
        PRED_LABEL_NAME_COLUMN,
        CORRECT_COLUMN,
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"{pool_path.name} is missing required columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    if df.empty:
        raise ValueError(f"{pool_path.name} is empty.")

    splits = set(df[SOURCE_SPLIT_COLUMN].astype(str).str.strip().unique())
    if splits != {expected_source_split}:
        raise ValueError(
            f"{pool_path.name} has source_split values {sorted(splits)}, "
            f"expected only {expected_source_split!r}."
        )

    text_columns = set(df[SOURCE_TEXT_COLUMN].astype(str).str.strip().unique())
    if text_columns != {"text_truncated"}:
        raise ValueError(
            f"{pool_path.name} uses source_text_column values {sorted(text_columns)}, "
            "expected only 'text_truncated'."
        )

    duplicate_keys = int(
        df[[SOURCE_SPLIT_COLUMN, ROW_ID_COLUMN]].duplicated().sum()
    )
    if duplicate_keys > 0:
        raise ValueError(
            f"{pool_path.name}: found {duplicate_keys} duplicate source_split/row_id keys."
        )

    duplicate_hashes = int(df[TEXT_HASH_COLUMN].duplicated().sum())
    if duplicate_hashes > 0:
        raise ValueError(f"{pool_path.name}: found {duplicate_hashes} duplicate text hashes.")

    empty_texts = int(df[ATTACK_TEXT_COLUMN].fillna("").astype(str).str.strip().eq("").sum())
    if empty_texts > 0:
        raise ValueError(f"{pool_path.name}: found {empty_texts} empty attack texts.")

    if not df[CORRECT_COLUMN].isin([1, True]).all():
        bad = int((~df[CORRECT_COLUMN].isin([1, True])).sum())
        raise ValueError(
            f"{pool_path.name}: found {bad} rows where correct != 1/True. "
            "Attack pools must contain only originally correctly classified examples."
        )

    labels = set(df[LABEL_COLUMN].astype(str).str.strip().str.lower().unique())
    expected = {TRUTHFUL_VALUE, DECEPTIVE_VALUE}
    if labels != expected:
        raise ValueError(
            f"{pool_path.name}: unexpected labels in '{LABEL_COLUMN}': {sorted(labels)}. "
            f"Expected exactly {sorted(expected)}."
        )

    mismatched_label_names = int(
        (
            df[LABEL_COLUMN].astype(str).str.strip().str.lower()
            != df[GOLD_LABEL_NAME_COLUMN].astype(str).str.strip().str.lower()
        ).sum()
    )
    if mismatched_label_names > 0:
        raise ValueError(
            f"{pool_path.name}: found {mismatched_label_names} rows where "
            f"{LABEL_COLUMN} differs from {GOLD_LABEL_NAME_COLUMN}."
        )


def load_pool(pool_path: Path, expected_source_split: str) -> pd.DataFrame:
    if not pool_path.exists():
        raise FileNotFoundError(
            f"Required candidate pool not found: {pool_path}. "
            "Run Scripts_code/01_baseline_inference.py first."
        )

    df = pd.read_csv(pool_path)
    validate_candidate_pool(df, pool_path, expected_source_split)
    return df


def sample_balanced_pool(
    df: pd.DataFrame,
    per_class: int,
    pool_name: str,
) -> pd.DataFrame:
    truthful_df = (
        df[df[LABEL_COLUMN].astype(str).str.strip().str.lower() == TRUTHFUL_VALUE]
        .copy()
    )
    deceptive_df = (
        df[df[LABEL_COLUMN].astype(str).str.strip().str.lower() == DECEPTIVE_VALUE]
        .copy()
    )

    if len(truthful_df) < per_class:
        raise ValueError(
            f"{pool_name}: not enough truthful examples. "
            f"Need {per_class}, found {len(truthful_df)}."
        )

    if len(deceptive_df) < per_class:
        raise ValueError(
            f"{pool_name}: not enough deceptive examples. "
            f"Need {per_class}, found {len(deceptive_df)}."
        )

    sampled_truthful = truthful_df.sample(n=per_class, random_state=SEED)
    sampled_deceptive = deceptive_df.sample(n=per_class, random_state=SEED)

    return (
        pd.concat([sampled_truthful, sampled_deceptive], ignore_index=True)
        .sort_values([SOURCE_SPLIT_COLUMN, ROW_ID_COLUMN])
        .reset_index(drop=True)
    )


def validate_cross_pool_separation(pilot_df: pd.DataFrame, main_df: pd.DataFrame) -> None:
    pilot_keys = set(zip(pilot_df[SOURCE_SPLIT_COLUMN], pilot_df[ROW_ID_COLUMN]))
    main_keys = set(zip(main_df[SOURCE_SPLIT_COLUMN], main_df[ROW_ID_COLUMN]))
    overlap = pilot_keys.intersection(main_keys)
    if overlap:
        raise ValueError(f"Pilot and main source keys overlap: {sorted(overlap)[:10]}")

    text_hash_overlap = set(pilot_df[TEXT_HASH_COLUMN]).intersection(
        set(main_df[TEXT_HASH_COLUMN])
    )
    if text_hash_overlap:
        raise ValueError(
            "Pilot and main pools contain identical attack_text hashes. "
            f"First overlaps: {sorted(text_hash_overlap)[:10]}"
        )


def build_pool_metadata(pool_path: Path, df: pd.DataFrame) -> dict[str, object]:
    source_datasets = sorted(
        set(df[SOURCE_DATASET_COLUMN].astype(str).str.strip().tolist())
    )
    source_text_columns = sorted(
        set(df[SOURCE_TEXT_COLUMN].astype(str).str.strip().tolist())
    )
    source_splits = sorted(
        set(df[SOURCE_SPLIT_COLUMN].astype(str).str.strip().tolist())
    )

    return {
        "pool_path": str(pool_path),
        "pool_sha256": compute_file_hash(pool_path),
        "source_splits": source_splits,
        "source_dataset_paths": source_datasets,
        "source_text_columns": source_text_columns,
        "num_rows": int(len(df)),
        "label_counts": {
            str(k): int(v)
            for k, v in df[LABEL_COLUMN].value_counts().sort_index().to_dict().items()
        },
    }


def build_frozen_subset_metadata(name: str, df: pd.DataFrame) -> dict[str, object]:
    return {
        "name": name,
        "num_rows": int(len(df)),
        "per_class": {
            str(k): int(v)
            for k, v in df[LABEL_COLUMN].value_counts().sort_index().to_dict().items()
        },
        "source_split_counts": {
            str(k): int(v)
            for k, v in df[SOURCE_SPLIT_COLUMN].value_counts().sort_index().to_dict().items()
        },
        "source_keys": [
            {
                "source_split": str(row[SOURCE_SPLIT_COLUMN]),
                "row_id": int(row[ROW_ID_COLUMN]),
                "source_index": int(row[SOURCE_INDEX_COLUMN]),
                "text_hash": str(row[TEXT_HASH_COLUMN]),
            }
            for _, row in df.iterrows()
        ],
    }


def build_metadata(
    train_dev_df: pd.DataFrame,
    test_final_df: pd.DataFrame,
    pilot_df: pd.DataFrame,
    main_df: pd.DataFrame,
) -> dict[str, object]:
    return {
        "seed": SEED,
        "selection_rule": (
            "Pilot is sampled from the train/dev correctly-classified pool. "
            "Main is sampled from the final test correctly-classified pool. "
            "Both are sampled separately per class and use text_truncated as attack_text."
        ),
        "canonical_text_column": "text_truncated",
        "pilot_per_class": PILOT_PER_CLASS,
        "main_per_class": MAIN_PER_CLASS,
        "input_pools": {
            "train_dev": build_pool_metadata(TRAIN_DEV_POOL_PATH, train_dev_df),
            "test_final": build_pool_metadata(TEST_FINAL_POOL_PATH, test_final_df),
        },
        "pilot": build_frozen_subset_metadata("pilot_train_dev_20", pilot_df),
        "main": build_frozen_subset_metadata("main_test_final_160", main_df),
    }


def main() -> None:
    args = parse_args()
    check_output_paths(overwrite=args.overwrite)

    print(f"Loading train/dev candidate pool from: {TRAIN_DEV_POOL_PATH}")
    train_dev_df = load_pool(TRAIN_DEV_POOL_PATH, expected_source_split="train_dev")

    print(f"Loading final test candidate pool from: {TEST_FINAL_POOL_PATH}")
    test_final_df = load_pool(TEST_FINAL_POOL_PATH, expected_source_split="test_final")

    pilot_df = sample_balanced_pool(
        train_dev_df,
        per_class=PILOT_PER_CLASS,
        pool_name="pilot",
    )
    main_df = sample_balanced_pool(
        test_final_df,
        per_class=MAIN_PER_CLASS,
        pool_name="main",
    )
    validate_cross_pool_separation(pilot_df, main_df)

    pilot_df.to_csv(PILOT_OUT, index=False)
    main_df.to_csv(MAIN_OUT, index=False)

    metadata = build_metadata(train_dev_df, test_final_df, pilot_df, main_df)
    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nFreeze complete.")
    print(f"Pilot set saved to: {PILOT_OUT}")
    print(f"Main set saved to:  {MAIN_OUT}")
    print(f"Metadata saved to:  {META_OUT}")

    print("\nPilot label counts:")
    print(pilot_df[LABEL_COLUMN].value_counts().sort_index())
    print("Pilot source splits:")
    print(pilot_df[SOURCE_SPLIT_COLUMN].value_counts().sort_index())

    print("\nMain label counts:")
    print(main_df[LABEL_COLUMN].value_counts().sort_index())
    print("Main source splits:")
    print(main_df[SOURCE_SPLIT_COLUMN].value_counts().sort_index())

    print("\nCross-pool separation check passed.")


if __name__ == "__main__":
    main()
