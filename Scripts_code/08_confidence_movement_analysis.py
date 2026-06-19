from __future__ import annotations

"""Post-hoc confidence-movement diagnostic for frozen BEP attack outputs.

This script is intentionally additive and read-only with respect to the frozen
attack outputs. It does not run the paraphraser, regenerate candidates, rerun
the attack loop, or modify existing result/metadata/analysis files.
"""

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "Scripts_code"
RAW_OUTPUT_DIR = SCRIPTS_DIR / "outputs"
OUTPUT_DIR = RAW_OUTPUT_DIR / "analysis_confidence_movement"
CLASSIFIER_WRAPPER_PATH = SCRIPTS_DIR / "03_classifier_wrapper.py"

EXPECTED_CONDITIONS = ["label_only", "score_based"]
EXPECTED_BUDGET_CONFIG = "gen3_query3"
EXPECTED_PROMPT_VERSION = "qwen3_paraphrase_feedback_v3_diverse_conservative"


@dataclass(frozen=True)
class DatasetConfig:
    dataset: str
    results_path: Path
    attempts_path: Path
    metadata_path: Path
    expected_result_rows: int
    expected_unique_row_ids: int
    expected_input_pool_rows: int
    expected_split: str


DATASETS = [
    DatasetConfig(
        dataset="primary_main",
        results_path=RAW_OUTPUT_DIR / "attack_results_main_local_llm_main_gen3_query3.csv",
        attempts_path=RAW_OUTPUT_DIR / "attack_attempts_main_local_llm_main_gen3_query3.csv",
        metadata_path=RAW_OUTPUT_DIR / "attack_main_meta_local_llm_main_gen3_query3.json",
        expected_result_rows=320,
        expected_unique_row_ids=160,
        expected_input_pool_rows=160,
        expected_split="main",
    ),
    DatasetConfig(
        dataset="exploratory_remaining",
        results_path=RAW_OUTPUT_DIR
        / "attack_results_exploratory_local_llm_remaining_gen3_query3.csv",
        attempts_path=RAW_OUTPUT_DIR
        / "attack_attempts_exploratory_local_llm_remaining_gen3_query3.csv",
        metadata_path=RAW_OUTPUT_DIR
        / "attack_exploratory_meta_local_llm_remaining_gen3_query3.json",
        expected_result_rows=458,
        expected_unique_row_ids=229,
        expected_input_pool_rows=229,
        expected_split="exploratory",
    ),
]


OUTPUT_FILES = {
    "candidate_level": OUTPUT_DIR / "candidate_level_rescored.csv",
    "best_candidate": OUTPUT_DIR / "best_candidate_row_condition.csv",
    "final_candidate": OUTPUT_DIR / "final_candidate_row_condition.csv",
    "candidate_summary": OUTPUT_DIR
    / "summary_candidate_level_by_dataset_condition.csv",
    "best_summary": OUTPUT_DIR / "summary_best_candidate_by_dataset_condition.csv",
    "final_summary": OUTPUT_DIR / "summary_final_candidate_by_dataset_condition.csv",
    "success_summary": OUTPUT_DIR / "summary_success_only_by_dataset_condition.csv",
    "paired_summary": OUTPUT_DIR / "summary_paired_best_candidate.csv",
    "metadata": OUTPUT_DIR / "confidence_movement_metadata.json",
    "markdown": OUTPUT_DIR / "confidence_movement_summary.md",
}


def load_module_from_path(module_name: str, module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def normalize_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        }
    ).fillna(False)


def normalize_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "row_id",
        "original_confidence",
        "attempt_index",
        "classifier_query_index",
        "semantic_similarity",
        "candidate_confidence",
        "queries_used",
        "generation_attempts_used",
        "valid_candidates",
        "invalid_candidates",
        "successful_similarity",
        "generation_seconds",
        "generated_token_count",
        "generation_tokens_per_second",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_dataset(config: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    results = pd.read_csv(config.results_path)
    attempts = pd.read_csv(config.attempts_path)
    with config.metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    for df in (results, attempts):
        for col in ["success", "is_valid", "queried_classifier", "is_success"]:
            if col in df.columns:
                df[col] = normalize_bool_series(df[col])
        normalize_numeric_columns(df)

    results["dataset"] = config.dataset
    attempts["dataset"] = config.dataset
    return results, attempts, metadata


def validate_dataset(
    config: DatasetConfig,
    results: pd.DataFrame,
    attempts: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, bool]:
    condition_counts = results.groupby("row_id")["feedback_condition"].apply(
        lambda s: sorted(s.dropna().astype(str).tolist())
    )
    expected_pair = sorted(EXPECTED_CONDITIONS)
    paired_conditions_ok = bool(
        len(condition_counts) == config.expected_unique_row_ids
        and condition_counts.map(lambda values: values == expected_pair).all()
    )

    checks = {
        "results_rows_match_expected": len(results) == config.expected_result_rows,
        "unique_row_ids_match_expected": results["row_id"].nunique()
        == config.expected_unique_row_ids,
        "each_row_id_has_both_conditions_once": paired_conditions_ok,
        "metadata_integrity_checks_passed": metadata.get("integrity_checks_passed")
        is True,
        "metadata_experiment_split_matches": metadata.get("experiment_split")
        == config.expected_split,
        "metadata_input_pool_rows_match": metadata.get("input_pool", {}).get("num_rows")
        == config.expected_input_pool_rows,
        "metadata_budget_config_matches": metadata.get("budget_config")
        == EXPECTED_BUDGET_CONFIG,
        "metadata_prompt_version_matches": metadata.get("prompt_version")
        == EXPECTED_PROMPT_VERSION,
        "original_predictions_equal_gold_labels": bool(
            (results["original_pred_label_name"] == results["gold_label_name"]).all()
        ),
    }

    queried = attempts[attempts["queried_classifier"]].copy()
    checks["queried_candidates_are_valid"] = bool(queried["is_valid"].all())
    checks["queried_candidates_have_text"] = bool(
        queried["candidate_text"].fillna("").astype(str).str.strip().ne("").all()
    )
    checks["queried_candidates_have_query_index"] = bool(
        queried["classifier_query_index"].notna().all()
    )
    checks["label_only_has_no_logged_candidate_confidence"] = bool(
        queried.loc[
            queried["feedback_condition"].astype(str) == "label_only",
            "candidate_confidence",
        ]
        .isna()
        .all()
    )

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        formatted = "\n".join(f"- {config.dataset}: {name}" for name in failed)
        raise RuntimeError(
            "Input integrity checks failed. No confidence-movement files were "
            f"written.\n{formatted}"
        )
    return checks


def prepare_candidate_rows(attempts: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    eligible = attempts[
        attempts["queried_classifier"]
        & attempts["is_valid"]
        & attempts["candidate_text"].fillna("").astype(str).str.strip().ne("")
        & attempts["classifier_query_index"].notna()
    ].copy()

    join_cols = [
        "dataset",
        "row_id",
        "feedback_condition",
        "queries_used",
        "success",
    ]
    eligible = eligible.merge(
        results[join_cols],
        on=["dataset", "row_id", "feedback_condition"],
        how="left",
        suffixes=("", "_result"),
        validate="many_to_one",
    )
    eligible["row_id"] = eligible["row_id"].astype(int)
    eligible["attempt_index"] = eligible["attempt_index"].astype(int)
    eligible["classifier_query_index"] = eligible["classifier_query_index"].astype(int)
    eligible["candidate_text"] = eligible["candidate_text"].astype(str).str.strip()
    eligible["is_success"] = eligible["is_success"].astype(bool)
    return eligible


def rescore_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    classifier_module = load_module_from_path(
        "classifier_wrapper_03_confidence_movement",
        CLASSIFIER_WRAPPER_PATH,
    )
    classifier = classifier_module.DeceptionClassifierWrapper()

    rescored_rows: list[dict[str, Any]] = []
    for row in candidates.itertuples(index=False):
        prediction = classifier.predict_label_and_confidence(row.candidate_text)
        posthoc_label = str(prediction.predicted_label_name)
        posthoc_confidence = float(prediction.confidence)

        if posthoc_label == str(row.original_pred_label_name):
            p_original_label_candidate = posthoc_confidence
        else:
            p_original_label_candidate = 1.0 - posthoc_confidence

        rescored_rows.append(
            {
                "dataset": row.dataset,
                "row_id": int(row.row_id),
                "feedback_condition": row.feedback_condition,
                "gold_label_name": row.gold_label_name,
                "original_pred_label_name": row.original_pred_label_name,
                "original_confidence": float(row.original_confidence),
                "attempt_index": int(row.attempt_index),
                "classifier_query_index": int(row.classifier_query_index),
                "original_attack_text": row.original_attack_text,
                "candidate_text": row.candidate_text,
                "semantic_similarity": row.semantic_similarity,
                "logged_candidate_pred_label_name": row.candidate_pred_label_name,
                "logged_candidate_confidence": row.candidate_confidence,
                "posthoc_candidate_pred_label_name": posthoc_label,
                "posthoc_candidate_confidence": posthoc_confidence,
                "p_original_label_candidate": p_original_label_candidate,
                "confidence_movement_toward_flip": float(row.original_confidence)
                - p_original_label_candidate,
                "is_success": bool(row.is_success),
                "result_success": bool(row.success),
                "stop_reason": row.stop_reason,
                "queries_used": row.queries_used,
                "logged_label_matches_posthoc": posthoc_label
                == str(row.candidate_pred_label_name),
            }
        )

    return pd.DataFrame(rescored_rows)


def with_scope_rows(df: pd.DataFrame) -> pd.DataFrame:
    primary = df[df["dataset"] == "primary_main"].copy()
    primary["dataset_scope"] = "primary_main"

    exploratory = df[df["dataset"] == "exploratory_remaining"].copy()
    exploratory["dataset_scope"] = "exploratory_remaining"

    combined = df.copy()
    combined["dataset_scope"] = "combined_held_out"

    return pd.concat([primary, exploratory, combined], ignore_index=True)


def summarize_candidate_level(rescored: pd.DataFrame) -> pd.DataFrame:
    scoped = with_scope_rows(rescored)
    rows: list[dict[str, Any]] = []
    for (scope, condition), group in scoped.groupby(
        ["dataset_scope", "feedback_condition"], sort=True
    ):
        movement = group["confidence_movement_toward_flip"]
        p_orig = group["p_original_label_candidate"]
        rows.append(
            {
                "dataset_scope": scope,
                "feedback_condition": condition,
                "n_candidates": int(len(group)),
                "mean_movement": movement.mean(),
                "median_movement": movement.median(),
                "sd_movement": movement.std(ddof=1),
                "min_movement": movement.min(),
                "max_movement": movement.max(),
                "mean_p_original_label_candidate": p_orig.mean(),
                "median_p_original_label_candidate": p_orig.median(),
            }
        )
    return pd.DataFrame(rows)


def summarize_row_condition(df: pd.DataFrame, n_col: str) -> pd.DataFrame:
    scoped = with_scope_rows(df)
    rows: list[dict[str, Any]] = []
    for (scope, condition), group in scoped.groupby(
        ["dataset_scope", "feedback_condition"], sort=True
    ):
        movement = group["confidence_movement_toward_flip"]
        p_orig = group["p_original_label_candidate"]
        rows.append(
            {
                "dataset_scope": scope,
                "feedback_condition": condition,
                n_col: int(len(group)),
                "mean_movement": movement.mean(),
                "median_movement": movement.median(),
                "sd_movement": movement.std(ddof=1),
                "min_movement": movement.min(),
                "max_movement": movement.max(),
                "mean_p_original_label_candidate": p_orig.mean(),
                "median_p_original_label_candidate": p_orig.median(),
            }
        )
    return pd.DataFrame(rows)


def best_candidate_by_row_condition(rescored: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [
        "dataset",
        "row_id",
        "feedback_condition",
        "p_original_label_candidate",
        "classifier_query_index",
        "attempt_index",
    ]
    best = rescored.sort_values(sort_cols, ascending=[True, True, True, True, True, True])
    return best.groupby(["dataset", "row_id", "feedback_condition"], as_index=False).first()


def final_candidate_by_row_condition(rescored: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [
        "dataset",
        "row_id",
        "feedback_condition",
        "classifier_query_index",
        "attempt_index",
    ]
    final = rescored.sort_values(sort_cols, ascending=[True, True, True, True, True])
    return final.groupby(["dataset", "row_id", "feedback_condition"], as_index=False).last()


def summarize_success_only(rescored: pd.DataFrame) -> pd.DataFrame:
    successes = rescored[rescored["is_success"]].copy()
    scoped = with_scope_rows(successes)
    rows: list[dict[str, Any]] = []
    for (scope, condition), group in scoped.groupby(
        ["dataset_scope", "feedback_condition"], sort=True
    ):
        post_flip = group["posthoc_candidate_confidence"]
        movement = group["confidence_movement_toward_flip"]
        queries = pd.to_numeric(group["queries_used"], errors="coerce")
        rows.append(
            {
                "dataset_scope": scope,
                "feedback_condition": condition,
                "n_successes": int(len(group)),
                "mean_post_flip_confidence": post_flip.mean(),
                "median_post_flip_confidence": post_flip.median(),
                "sd_post_flip_confidence": post_flip.std(ddof=1),
                "mean_movement": movement.mean(),
                "median_movement": movement.median(),
                "sd_movement": movement.std(ddof=1),
                "mean_queries_used": queries.mean(),
                "median_queries_used": queries.median(),
                "sd_queries_used": queries.std(ddof=1),
            }
        )
    return pd.DataFrame(rows)


def paired_best_candidate_summary(best: pd.DataFrame) -> pd.DataFrame:
    scoped = with_scope_rows(best)
    rows: list[dict[str, Any]] = []

    for scope, group in scoped.groupby("dataset_scope", sort=True):
        index_cols = ["dataset", "row_id"] if scope == "combined_held_out" else ["row_id"]
        pivot = group.pivot_table(
            index=index_cols,
            columns="feedback_condition",
            values="confidence_movement_toward_flip",
            aggfunc="first",
        )
        pivot = pivot.dropna(subset=EXPECTED_CONDITIONS)
        if pivot.empty:
            diff = pd.Series(dtype="float64")
        else:
            diff = pivot["score_based"] - pivot["label_only"]

        rows.append(
            {
                "dataset_scope": scope,
                "n_paired_rows": int(len(diff)),
                "mean_movement_difference_score_minus_label": diff.mean()
                if len(diff)
                else None,
                "median_movement_difference_score_minus_label": diff.median()
                if len(diff)
                else None,
                "score_based_movement_gt_label_only": int((diff > 0).sum()),
                "label_only_movement_gt_score_based": int((diff < 0).sum()),
                "ties": int((diff == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def mismatch_table(rescored: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dataset",
        "row_id",
        "feedback_condition",
        "attempt_index",
        "classifier_query_index",
        "original_pred_label_name",
        "logged_candidate_pred_label_name",
        "posthoc_candidate_pred_label_name",
        "logged_candidate_confidence",
        "posthoc_candidate_confidence",
        "candidate_text",
    ]
    return rescored.loc[~rescored["logged_label_matches_posthoc"], cols].copy()


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    if pd.isna(value):
        return None
    return value


def safe_write_csv(df: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    df.to_csv(path, index=False, encoding="utf-8")


def safe_write_text(text: str, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.write_text(text, encoding="utf-8")


def format_pct(part: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{part / total * 100:.1f}%"


def select_summary_value(
    df: pd.DataFrame,
    dataset_scope: str,
    feedback_condition: str,
    column: str,
) -> Any:
    row = df[
        (df["dataset_scope"] == dataset_scope)
        & (df["feedback_condition"] == feedback_condition)
    ]
    if row.empty:
        return None
    return row.iloc[0][column]


def build_markdown(
    candidate_summary: pd.DataFrame,
    best_summary: pd.DataFrame,
    success_summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
    mismatches: pd.DataFrame,
) -> str:
    lines = [
        "# Confidence Movement Post-hoc Diagnostic",
        "",
        "This analysis is post-hoc and exploratory. It was run after the primary "
        "and exploratory attack experiments were frozen, and it did not affect "
        "attack generation, stopping rules, budgets, or success definitions.",
        "",
        "Confidence was not available to the `label_only` attacker during the "
        "attack. For fairness, all queried valid candidates from both feedback "
        "conditions were rescored after the fact with the same fixed DistilBERT "
        "classifier. The diagnostic only asks how much already-generated "
        "candidates moved the classifier away from its original decision.",
        "",
        "For each candidate, the analysis computes the post-hoc probability "
        "assigned to the original predicted label. The movement metric is "
        "`original_confidence - p_original_label_candidate`; positive values "
        "mean movement away from the original decision, zero or small values "
        "mean little movement, and negative values mean the candidate increased "
        "confidence in the original decision.",
        "",
        "## Label Verification",
        "",
    ]

    if mismatches.empty:
        lines.append(
            "No mismatches were found between the logged candidate label and "
            "the post-hoc rescored candidate label."
        )
    else:
        lines.append(
            f"{len(mismatches)} mismatches were found. See "
            "`candidate_label_mismatches.csv`; these should be reported rather "
            "than silently ignored."
        )

    lines.extend(["", "## Descriptive Results", ""])
    for scope in ["primary_main", "exploratory_remaining", "combined_held_out"]:
        paired = paired_summary[paired_summary["dataset_scope"] == scope].iloc[0]
        label_mean = select_summary_value(
            best_summary, scope, "label_only", "mean_movement"
        )
        score_mean = select_summary_value(
            best_summary, scope, "score_based", "mean_movement"
        )
        label_cand_mean = select_summary_value(
            candidate_summary, scope, "label_only", "mean_movement"
        )
        score_cand_mean = select_summary_value(
            candidate_summary, scope, "score_based", "mean_movement"
        )
        score_gt = int(paired["score_based_movement_gt_label_only"])
        label_gt = int(paired["label_only_movement_gt_score_based"])
        ties = int(paired["ties"])
        n_pairs = int(paired["n_paired_rows"])

        lines.extend(
            [
                f"### {scope}",
                "",
                (
                    "Candidate-level mean movement was "
                    f"{label_cand_mean:.4f} for `label_only` and "
                    f"{score_cand_mean:.4f} for `score_based`."
                ),
                (
                    "Best-candidate row-condition mean movement was "
                    f"{label_mean:.4f} for `label_only` and "
                    f"{score_mean:.4f} for `score_based`."
                ),
                (
                    "In paired rows with at least one queried valid candidate "
                    f"in both conditions (n={n_pairs}), score-based movement "
                    f"was larger in {score_gt} rows ({format_pct(score_gt, n_pairs)}), "
                    f"label-only movement was larger in {label_gt} rows "
                    f"({format_pct(label_gt, n_pairs)}), and ties occurred in "
                    f"{ties} rows."
                ),
                "",
            ]
        )

    combined_paired = paired_summary[
        paired_summary["dataset_scope"] == "combined_held_out"
    ].iloc[0]
    combined_diff = combined_paired[
        "mean_movement_difference_score_minus_label"
    ]
    combined_success_label = select_summary_value(
        success_summary, "combined_held_out", "label_only", "mean_queries_used"
    )
    combined_success_score = select_summary_value(
        success_summary, "combined_held_out", "score_based", "mean_queries_used"
    )

    lines.extend(
        [
            "## Thesis-ready Interpretation",
            "",
            (
                "Because this diagnostic was performed after the attack runs, "
                "it should be presented as descriptive evidence about the "
                "classifier's confidence landscape, not as a new primary "
                "hypothesis test."
            ),
            (
                "The combined held-out best-candidate comparison gives a mean "
                f"movement difference of {combined_diff:.4f} for "
                "`score_based - label_only`. This helps diagnose whether score "
                "feedback tended to produce candidates closer to a flip, even "
                "when ASR differences were small or unstable."
            ),
            (
                "Among successful attacks in the combined held-out set, mean "
                f"queries used were {combined_success_label:.3f} for "
                f"`label_only` and {combined_success_score:.3f} for "
                "`score_based`. This should be interpreted together with the "
                "primary ASR and query-efficiency results rather than replacing "
                "them."
            ),
            (
                "Overall, these results are best suited for the Discussion or "
                "Appendix, with only a brief mention in the main Results section "
                "if space is available. They answer the confidence-movement "
                "question defensibly without changing the original experiment."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_metadata(
    integrity_checks: dict[str, dict[str, bool]],
    rescored: pd.DataFrame,
    mismatches: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "created_by_script": Path(__file__).name,
        "analysis_type": "post_hoc_exploratory_confidence_movement",
        "raw_inputs": {
            config.dataset: {
                "results_path": str(config.results_path),
                "attempts_path": str(config.attempts_path),
                "metadata_path": str(config.metadata_path),
            }
            for config in DATASETS
        },
        "output_dir": str(OUTPUT_DIR),
        "integrity_checks": integrity_checks,
        "candidate_selection": (
            "queried_classifier == True, is_valid == True, non-empty candidate_text, "
            "and present classifier_query_index"
        ),
        "rescoring": (
            "All selected candidates from both feedback conditions were rescored "
            "post hoc using DeceptionClassifierWrapper.predict_label_and_confidence."
        ),
        "metric": {
            "p_original_label_candidate": (
                "posthoc confidence if posthoc candidate label equals original "
                "predicted label, otherwise 1 - posthoc confidence"
            ),
            "confidence_movement_toward_flip": (
                "original_confidence - p_original_label_candidate"
            ),
        },
        "rows": {
            "rescored_candidates": int(len(rescored)),
            "primary_main_candidates": int(
                (rescored["dataset"] == "primary_main").sum()
            ),
            "exploratory_remaining_candidates": int(
                (rescored["dataset"] == "exploratory_remaining").sum()
            ),
            "label_mismatches": int(len(mismatches)),
        },
    }


def main() -> None:
    existing_outputs = [path for path in OUTPUT_FILES.values() if path.exists()]
    if existing_outputs:
        formatted = "\n".join(f"- {path}" for path in existing_outputs)
        raise FileExistsError(
            "Refusing to overwrite existing confidence-movement outputs:\n"
            + formatted
        )

    all_results: list[pd.DataFrame] = []
    all_attempts: list[pd.DataFrame] = []
    integrity_checks: dict[str, dict[str, bool]] = {}

    for config in DATASETS:
        results, attempts, metadata = read_dataset(config)
        integrity_checks[config.dataset] = validate_dataset(
            config=config,
            results=results,
            attempts=attempts,
            metadata=metadata,
        )
        all_results.append(results)
        all_attempts.append(attempts)

    results = pd.concat(all_results, ignore_index=True)
    attempts = pd.concat(all_attempts, ignore_index=True)
    candidates = prepare_candidate_rows(attempts, results)
    rescored = rescore_candidates(candidates)

    mismatches = mismatch_table(rescored)
    best = best_candidate_by_row_condition(rescored)
    final = final_candidate_by_row_condition(rescored)
    candidate_summary = summarize_candidate_level(rescored)
    best_summary = summarize_row_condition(best, "n_row_conditions")
    final_summary = summarize_row_condition(final, "n_row_conditions")
    success_summary = summarize_success_only(rescored)
    paired_summary = paired_best_candidate_summary(best)
    metadata = build_metadata(integrity_checks, rescored, mismatches)
    markdown = build_markdown(
        candidate_summary=candidate_summary,
        best_summary=best_summary,
        success_summary=success_summary,
        paired_summary=paired_summary,
        mismatches=mismatches,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_write_csv(rescored, OUTPUT_FILES["candidate_level"])
    safe_write_csv(best, OUTPUT_FILES["best_candidate"])
    safe_write_csv(final, OUTPUT_FILES["final_candidate"])
    safe_write_csv(candidate_summary, OUTPUT_FILES["candidate_summary"])
    safe_write_csv(best_summary, OUTPUT_FILES["best_summary"])
    safe_write_csv(final_summary, OUTPUT_FILES["final_summary"])
    safe_write_csv(success_summary, OUTPUT_FILES["success_summary"])
    safe_write_csv(paired_summary, OUTPUT_FILES["paired_summary"])
    if not mismatches.empty:
        safe_write_csv(mismatches, OUTPUT_DIR / "candidate_label_mismatches.csv")
    safe_write_text(json.dumps(to_builtin(metadata), indent=2) + "\n", OUTPUT_FILES["metadata"])
    safe_write_text(markdown, OUTPUT_FILES["markdown"])

    print("Confidence-movement analysis complete.")
    print(f"Rescored candidates: {len(rescored)}")
    print(f"Label mismatches: {len(mismatches)}")
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
