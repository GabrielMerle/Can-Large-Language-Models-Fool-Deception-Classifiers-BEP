from __future__ import annotations

"""Build the exploratory pool from final-test examples not used in main.

This script is intentionally separate from the frozen pilot/main pool freezer.
It reads the already-existing final-test candidate pool and frozen main pool,
then writes a new exploratory pool containing only the remaining held-out,
correctly classified examples.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "Scripts_code" / "outputs"

TEST_FINAL_POOL_PATH = OUTPUT_DIR / "candidate_pool_test_final.csv"
MAIN_POOL_PATH = OUTPUT_DIR / "candidate_pool_main_160.csv"
EXPLORATORY_POOL_OUT = OUTPUT_DIR / "candidate_pool_exploratory_remaining.csv"
EXPLORATORY_META_OUT = OUTPUT_DIR / "candidate_pool_exploratory_remaining_meta.json"

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

EXPECTED_SOURCE_SPLIT = "test_final"
EXPECTED_SOURCE_TEXT_COLUMN = "text_truncated"
EXPECTED_TOTAL_ROWS = 229
EXPECTED_LABEL_COUNTS = {
    "truthful": 90,
    "deceptive": 139,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create candidate_pool_exploratory_remaining.csv from final-test "
            "candidate rows not present in candidate_pool_main_160.csv."
        )
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing exploratory pool outputs.",
    )
    return parser.parse_args()


def compute_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def source_keys(df: pd.DataFrame) -> set[tuple[str, str]]:
    return set(
        zip(
            df[SOURCE_SPLIT_COLUMN].astype(str).str.strip(),
            df[ROW_ID_COLUMN].astype(str).str.strip(),
        )
    )


def validate_output_paths(overwrite: bool) -> None:
    existing = [
        path
        for path in [EXPLORATORY_POOL_OUT, EXPLORATORY_META_OUT]
        if path.exists()
    ]
    if existing and not overwrite:
        formatted = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(
            "Exploratory pool outputs already exist. Refusing to overwrite them.\n"
            f"{formatted}\n"
            "Rerun with --overwrite only when you intentionally want to rebuild them."
        )


def require_columns(df: pd.DataFrame, path: Path) -> None:
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
            f"{path.name} is missing required columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )


def normalize_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin({"true", "1", "yes"})


def validate_candidate_pool(df: pd.DataFrame, path: Path) -> None:
    require_columns(df, path)
    if df.empty:
        raise ValueError(f"{path.name} is empty.")

    splits = set(df[SOURCE_SPLIT_COLUMN].astype(str).str.strip().unique())
    if splits != {EXPECTED_SOURCE_SPLIT}:
        raise ValueError(
            f"{path.name} has source_split values {sorted(splits)}, "
            f"expected only {EXPECTED_SOURCE_SPLIT!r}."
        )

    text_columns = set(df[SOURCE_TEXT_COLUMN].astype(str).str.strip().unique())
    if text_columns != {EXPECTED_SOURCE_TEXT_COLUMN}:
        raise ValueError(
            f"{path.name} uses source_text_column values {sorted(text_columns)}, "
            f"expected only {EXPECTED_SOURCE_TEXT_COLUMN!r}."
        )

    duplicate_keys = int(
        df[[SOURCE_SPLIT_COLUMN, ROW_ID_COLUMN]].astype(str).duplicated().sum()
    )
    if duplicate_keys:
        raise ValueError(
            f"{path.name}: found {duplicate_keys} duplicate source_split/row_id keys."
        )

    duplicate_hashes = int(df[TEXT_HASH_COLUMN].astype(str).duplicated().sum())
    if duplicate_hashes:
        raise ValueError(
            f"{path.name}: found {duplicate_hashes} duplicate text_hash values."
        )

    empty_texts = int(
        df[ATTACK_TEXT_COLUMN].fillna("").astype(str).str.strip().eq("").sum()
    )
    if empty_texts:
        raise ValueError(f"{path.name}: found {empty_texts} empty attack_text rows.")

    correct = normalize_bool_series(df[CORRECT_COLUMN])
    if not bool(correct.all()):
        bad = int((~correct).sum())
        raise ValueError(
            f"{path.name}: found {bad} rows where correct is not true/1."
        )

    label_names = df[LABEL_COLUMN].astype(str).str.strip().str.lower()
    gold_label_names = df[GOLD_LABEL_NAME_COLUMN].astype(str).str.strip().str.lower()
    mismatch_count = int((label_names != gold_label_names).sum())
    if mismatch_count:
        raise ValueError(
            f"{path.name}: found {mismatch_count} rows where {LABEL_COLUMN} "
            f"differs from {GOLD_LABEL_NAME_COLUMN}."
        )


def load_pool(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input pool not found: {path}")
    df = pd.read_csv(path)
    validate_candidate_pool(df, path)
    return df


def build_remaining_pool(test_final_df: pd.DataFrame, main_df: pd.DataFrame) -> pd.DataFrame:
    main_keys = source_keys(main_df)
    main_hashes = set(main_df[TEXT_HASH_COLUMN].astype(str).str.strip())

    test_keys = source_keys(test_final_df)
    if not main_keys.issubset(test_keys):
        missing = sorted(main_keys - test_keys)[:10]
        raise ValueError(
            "candidate_pool_main_160.csv contains source keys not present in "
            f"candidate_pool_test_final.csv. First missing keys: {missing}"
        )

    in_main_by_key = test_final_df.apply(
        lambda row: (
            str(row[SOURCE_SPLIT_COLUMN]).strip(),
            str(row[ROW_ID_COLUMN]).strip(),
        )
        in main_keys,
        axis=1,
    )
    in_main_by_hash = test_final_df[TEXT_HASH_COLUMN].astype(str).str.strip().isin(main_hashes)

    disagreement_count = int((in_main_by_key != in_main_by_hash).sum())
    if disagreement_count:
        raise ValueError(
            "Main-pool membership by source_split/row_id disagrees with membership "
            f"by text_hash for {disagreement_count} rows. Refusing to infer the "
            "exploratory remainder."
        )

    remaining = (
        test_final_df.loc[~in_main_by_key & ~in_main_by_hash]
        .copy()
        .sort_values([SOURCE_SPLIT_COLUMN, ROW_ID_COLUMN])
        .reset_index(drop=True)
    )
    return remaining


def validate_remaining_pool(remaining_df: pd.DataFrame, main_df: pd.DataFrame) -> None:
    validate_candidate_pool(remaining_df, EXPLORATORY_POOL_OUT)

    remaining_keys = source_keys(remaining_df)
    main_keys = source_keys(main_df)
    key_overlap = remaining_keys.intersection(main_keys)
    if key_overlap:
        raise ValueError(
            "Exploratory pool overlaps main pool by source_split/row_id. "
            f"First overlaps: {sorted(key_overlap)[:10]}"
        )

    hash_overlap = set(remaining_df[TEXT_HASH_COLUMN].astype(str).str.strip()).intersection(
        set(main_df[TEXT_HASH_COLUMN].astype(str).str.strip())
    )
    if hash_overlap:
        raise ValueError(
            "Exploratory pool overlaps main pool by text_hash. "
            f"First overlaps: {sorted(hash_overlap)[:10]}"
        )

    label_counts = {
        str(k): int(v)
        for k, v in remaining_df[LABEL_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
        .value_counts()
        .sort_index()
        .to_dict()
        .items()
    }
    expected_label_counts = dict(sorted(EXPECTED_LABEL_COUNTS.items()))
    if len(remaining_df) != EXPECTED_TOTAL_ROWS or label_counts != expected_label_counts:
        raise ValueError(
            "Exploratory remaining-pool counts differ from the locked expectation. "
            f"Expected total={EXPECTED_TOTAL_ROWS}, labels={expected_label_counts}; "
            f"found total={len(remaining_df)}, labels={label_counts}. "
            "No exploratory pool was written."
        )


def pool_summary(path: Path, df: pd.DataFrame) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": compute_file_hash(path),
        "num_rows": int(len(df)),
        "label_counts": {
            str(k): int(v)
            for k, v in df[LABEL_COLUMN]
            .astype(str)
            .str.strip()
            .str.lower()
            .value_counts()
            .sort_index()
            .to_dict()
            .items()
        },
        "source_splits": sorted(
            set(df[SOURCE_SPLIT_COLUMN].astype(str).str.strip().tolist())
        ),
        "source_text_columns": sorted(
            set(df[SOURCE_TEXT_COLUMN].astype(str).str.strip().tolist())
        ),
    }


def build_metadata(
    test_final_df: pd.DataFrame,
    main_df: pd.DataFrame,
    remaining_df: pd.DataFrame,
) -> dict[str, Any]:
    remaining_keys = source_keys(remaining_df)
    main_keys = source_keys(main_df)
    remaining_hashes = set(remaining_df[TEXT_HASH_COLUMN].astype(str).str.strip())
    main_hashes = set(main_df[TEXT_HASH_COLUMN].astype(str).str.strip())

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Exploratory held-out pool built from candidate_pool_test_final.csv "
            "after excluding all rows already used in candidate_pool_main_160.csv."
        ),
        "selection_rule": (
            "Exclude rows whose source_split/row_id key or text_hash appears in "
            "the frozen primary main pool; preserve the original candidate-pool columns."
        ),
        "expected_counts": {
            "num_rows": EXPECTED_TOTAL_ROWS,
            "label_counts": dict(sorted(EXPECTED_LABEL_COUNTS.items())),
        },
        "input_pools": {
            "test_final": pool_summary(TEST_FINAL_POOL_PATH, test_final_df),
            "main_160": pool_summary(MAIN_POOL_PATH, main_df),
        },
        "exploratory_remaining": pool_summary(EXPLORATORY_POOL_OUT, remaining_df),
        "validation": {
            "source_split_is_test_final": sorted(
                set(remaining_df[SOURCE_SPLIT_COLUMN].astype(str).str.strip())
            )
            == [EXPECTED_SOURCE_SPLIT],
            "source_text_column_is_text_truncated": sorted(
                set(remaining_df[SOURCE_TEXT_COLUMN].astype(str).str.strip())
            )
            == [EXPECTED_SOURCE_TEXT_COLUMN],
            "overlap_with_main_by_source_split_row_id": int(
                len(remaining_keys.intersection(main_keys))
            ),
            "overlap_with_main_by_text_hash": int(
                len(remaining_hashes.intersection(main_hashes))
            ),
        },
        "source_keys": [
            {
                "source_split": str(row[SOURCE_SPLIT_COLUMN]),
                "row_id": int(row[ROW_ID_COLUMN]),
                "source_index": int(row[SOURCE_INDEX_COLUMN]),
                "text_hash": str(row[TEXT_HASH_COLUMN]),
            }
            for _, row in remaining_df.iterrows()
        ],
    }


def main() -> None:
    args = parse_args()
    validate_output_paths(overwrite=args.overwrite)

    print(f"Loading final-test pool from: {TEST_FINAL_POOL_PATH}")
    test_final_df = load_pool(TEST_FINAL_POOL_PATH)

    print(f"Loading frozen main pool from: {MAIN_POOL_PATH}")
    main_df = load_pool(MAIN_POOL_PATH)

    remaining_df = build_remaining_pool(test_final_df, main_df)
    validate_remaining_pool(remaining_df, main_df)

    remaining_df.to_csv(EXPLORATORY_POOL_OUT, index=False, encoding="utf-8")
    metadata = build_metadata(test_final_df, main_df, remaining_df)
    with EXPLORATORY_META_OUT.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    print("\nExploratory remaining pool complete.")
    print(f"Pool saved to:     {EXPLORATORY_POOL_OUT}")
    print(f"Metadata saved to: {EXPLORATORY_META_OUT}")
    print("\nLabel counts:")
    print(remaining_df[LABEL_COLUMN].value_counts().sort_index())
    print("\nOverlap with main by source_split/row_id: 0")
    print("Overlap with main by text_hash: 0")


if __name__ == "__main__":
    main()
