from __future__ import annotations

"""Analyze the frozen final main attack outputs.

This script is intentionally read-only with respect to the raw experiment files.
It never calls the paraphraser, classifier, or LLM server. All derived outputs are
written under Scripts_code/outputs/analysis_main_gen3_query3/.
"""

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "Scripts_code" / "outputs"
ANALYSIS_DIR = OUTPUT_DIR / "analysis_main_gen3_query3"

RESULTS_PATH = OUTPUT_DIR / "attack_results_main_local_llm_main_gen3_query3.csv"
ATTEMPTS_PATH = OUTPUT_DIR / "attack_attempts_main_local_llm_main_gen3_query3.csv"
META_PATH = OUTPUT_DIR / "attack_main_meta_local_llm_main_gen3_query3.json"

EXPECTED_RESULT_ROWS = 320
EXPECTED_UNIQUE_ROW_IDS = 160
EXPECTED_CONDITIONS = ["label_only", "score_based"]
EXPECTED_SPLIT = "main"
EXPECTED_INPUT_POOL_ROWS = 160
EXPECTED_BUDGET_CONFIG = "gen3_query3"
EXPECTED_PROMPT_VERSION = "qwen3_paraphrase_feedback_v3_diverse_conservative"


def to_builtin(value: Any) -> Any:
    """Convert pandas/numpy scalar values into JSON-serializable Python values."""
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    if pd.isna(value):
        return None
    return value


def normalize_bool_series(series: pd.Series) -> pd.Series:
    """Handle booleans robustly if CSV parsing returns strings instead of bools."""
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


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load final raw outputs without modifying them."""
    results = pd.read_csv(RESULTS_PATH)
    attempts = pd.read_csv(ATTEMPTS_PATH)

    with META_PATH.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    for df in (results, attempts):
        for col in ["success", "is_valid", "queried_classifier", "is_success"]:
            if col in df.columns:
                df[col] = normalize_bool_series(df[col])

    numeric_cols = [
        "queries_used",
        "generation_attempts_used",
        "valid_candidates",
        "invalid_candidates",
        "successful_similarity",
        "attempt_index",
        "classifier_query_index",
        "semantic_similarity",
        "generation_seconds",
        "generated_token_count",
        "generation_tokens_per_second",
    ]
    for df in (results, attempts):
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    return results, attempts, meta


def validate_inputs(results: pd.DataFrame, attempts: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    """Run the thesis-critical integrity checks before writing derived outputs."""
    condition_counts = results.groupby("row_id")["feedback_condition"].apply(
        lambda s: sorted(s.dropna().astype(str).tolist())
    )
    expected_pair = sorted(EXPECTED_CONDITIONS)
    paired_condition_ok = bool(
        len(condition_counts) == EXPECTED_UNIQUE_ROW_IDS
        and condition_counts.map(lambda values: values == expected_pair).all()
    )

    checks = {
        "results_rows_is_320": len(results) == EXPECTED_RESULT_ROWS,
        "unique_row_ids_is_160": results["row_id"].nunique() == EXPECTED_UNIQUE_ROW_IDS,
        "each_row_id_has_label_only_and_score_based_once": paired_condition_ok,
        "metadata_integrity_checks_passed_is_true": meta.get("integrity_checks_passed") is True,
        "metadata_experiment_split_is_main": meta.get("experiment_split") == EXPECTED_SPLIT,
        "metadata_input_pool_num_rows_is_160": meta.get("input_pool", {}).get("num_rows")
        == EXPECTED_INPUT_POOL_ROWS,
        "metadata_budget_config_is_gen3_query3": meta.get("budget_config") == EXPECTED_BUDGET_CONFIG,
        "metadata_prompt_version_matches_expected": meta.get("prompt_version")
        == EXPECTED_PROMPT_VERSION,
    }

    # This is not one of the hard requested checks, but it documents the directional
    # interpretation used later: original predictions match gold labels in the main pool.
    checks["original_predictions_equal_gold_labels"] = bool(
        (results["original_pred_label_name"] == results["gold_label_name"]).all()
    )

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        details = "\n".join(f"- {name}" for name in failed)
        raise RuntimeError(
            "Input integrity checks failed. No analysis files were written.\n" + details
        )

    return checks


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    """Wilson score interval for a binomial proportion."""
    if total <= 0:
        return None, None
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half_width = z * math.sqrt((p * (1 - p) / total) + (z**2 / (4 * total**2))) / denom
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def asr_by_condition(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition, group in results.groupby("feedback_condition", sort=True):
        successes = int(group["success"].sum())
        total = int(len(group))
        asr = successes / total if total else float("nan")
        ci_low, ci_high = wilson_ci(successes, total)
        rows.append(
            {
                "feedback_condition": condition,
                "successes": successes,
                "total": total,
                "ASR": asr,
                "percentage": asr * 100,
                "wilson_95_ci_low": ci_low,
                "wilson_95_ci_high": ci_high,
                "wilson_95_ci_low_percentage": None if ci_low is None else ci_low * 100,
                "wilson_95_ci_high_percentage": None if ci_high is None else ci_high * 100,
            }
        )
    return pd.DataFrame(rows)


def paired_success_table(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired = results.pivot(index="row_id", columns="feedback_condition", values="success")
    paired = paired.reindex(columns=EXPECTED_CONDITIONS)

    both_success = int((paired["label_only"] & paired["score_based"]).sum())
    label_only_only = int((paired["label_only"] & ~paired["score_based"]).sum())
    score_based_only = int((~paired["label_only"] & paired["score_based"]).sum())
    neither_success = int((~paired["label_only"] & ~paired["score_based"]).sum())

    table = pd.DataFrame(
        [
            {
                "total_pairs": int(len(paired)),
                "both_success": both_success,
                "label_only_only": label_only_only,
                "score_based_only": score_based_only,
                "neither_success": neither_success,
            }
        ]
    )
    return table, paired.reset_index()


def exact_binomial_two_sided_manual(k: int, n: int, p: float = 0.5) -> float:
    """Manual exact two-sided binomial p-value, matching scipy for p=0.5 here."""
    if n == 0:
        return 1.0
    observed_prob = math.comb(n, k) * (p**k) * ((1 - p) ** (n - k))
    p_value = 0.0
    for i in range(n + 1):
        prob = math.comb(n, i) * (p**i) * ((1 - p) ** (n - i))
        if prob <= observed_prob + 1e-15:
            p_value += prob
    return min(1.0, p_value)


def mcnemar_exact(paired_table: pd.DataFrame) -> pd.DataFrame:
    row = paired_table.iloc[0]
    b = int(row["label_only_only"])
    c = int(row["score_based_only"])
    discordant = b + c

    if discordant == 0:
        p_value = 1.0
        method = "Exact McNemar/binomial test not informative because there are no discordant pairs."
    else:
        try:
            from scipy.stats import binomtest  # type: ignore

            p_value = float(binomtest(k=b, n=discordant, p=0.5, alternative="two-sided").pvalue)
            method = "Exact McNemar test via scipy.stats.binomtest on discordant pairs."
        except Exception:
            p_value = exact_binomial_two_sided_manual(k=b, n=discordant, p=0.5)
            method = "Exact McNemar test via manual two-sided binomial test on discordant pairs."

    if p_value < 0.05:
        interpretation = (
            "At alpha=0.05, the paired success difference is statistically detectable."
        )
    else:
        interpretation = (
            "At alpha=0.05, there is no statistically detectable paired success difference."
        )

    return pd.DataFrame(
        [
            {
                "b_label_only_success_score_based_failure": b,
                "c_label_only_failure_score_based_success": c,
                "discordant_pairs": discordant,
                "statistic_description": "Under the null, b follows Binomial(b+c, 0.5).",
                "method": method,
                "p_value": p_value,
                "interpretation": interpretation,
            }
        ]
    )


def summarize_numeric(
    df: pd.DataFrame,
    columns: list[str],
    group_col: str = "feedback_condition",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped: list[tuple[str, pd.DataFrame]] = [("overall", df)]
    grouped.extend((str(name), group) for name, group in df.groupby(group_col, sort=True))

    for condition, group in grouped:
        for col in columns:
            values = group[col].dropna()
            rows.append(
                {
                    "feedback_condition": condition,
                    "metric": col,
                    "count": int(values.count()),
                    "mean": values.mean(),
                    "median": values.median(),
                    "std": values.std(ddof=1),
                    "min": values.min(),
                    "max": values.max(),
                    "q25": values.quantile(0.25),
                    "q75": values.quantile(0.75),
                }
            )
    return pd.DataFrame(rows)


def directional_success(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (condition, gold_label), group in results.groupby(
        ["feedback_condition", "gold_label_name"], sort=True
    ):
        successes = int(group["success"].sum())
        total = int(len(group))
        asr = successes / total if total else float("nan")
        rows.append(
            {
                "feedback_condition": condition,
                "gold_label_name": gold_label,
                "directional_flip_interpretation": f"{gold_label} -> opposite predicted label",
                "successes": successes,
                "total": total,
                "ASR": asr,
                "percentage": asr * 100,
            }
        )
    return pd.DataFrame(rows)


def validity_by_condition(attempts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped: list[tuple[str, pd.DataFrame]] = [("overall", attempts)]
    grouped.extend((str(name), group) for name, group in attempts.groupby("feedback_condition", sort=True))

    for condition, group in grouped:
        valid_attempts = int(group["is_valid"].sum())
        total_attempts = int(len(group))
        queried_true = int(group["queried_classifier"].sum())
        queried_false = int((~group["queried_classifier"]).sum())
        invalid_queried = int((~group["is_valid"] & group["queried_classifier"]).sum())
        rows.append(
            {
                "feedback_condition": condition,
                "valid_attempts": valid_attempts,
                "total_attempts": total_attempts,
                "invalid_attempts": total_attempts - valid_attempts,
                "validity_rate": valid_attempts / total_attempts if total_attempts else float("nan"),
                "validity_percentage": (valid_attempts / total_attempts * 100)
                if total_attempts
                else float("nan"),
                "queried_classifier_true": queried_true,
                "queried_classifier_false": queried_false,
                "invalid_queried_count": invalid_queried,
                "invalid_candidates_not_queried": invalid_queried == 0,
            }
        )
    return pd.DataFrame(rows)


def parse_failed_checks(value: Any) -> list[str]:
    """Parse failed_checks cells that store JSON-like lists, with safe fallbacks."""
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "[]"}:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
        return [str(parsed)]
    except json.JSONDecodeError:
        # Some historical/debug files used ad hoc strings. Keep them analyzable.
        cleaned = text.strip("[]")
        parts = [part.strip().strip("'\"") for part in cleaned.split(",")]
        return [part for part in parts if part]


def failed_checks_table(attempts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_counts(scope_condition: str, group: pd.DataFrame) -> None:
        total = len(group)
        raw_counter: Counter[str] = Counter()
        individual_counter: Counter[str] = Counter()

        for value in group["failed_checks"]:
            checks = parse_failed_checks(value)
            raw_key = "no_failed_checks" if not checks else "|".join(checks)
            raw_counter[raw_key] += 1
            individual_counter.update(checks)

        for name, count in raw_counter.most_common():
            rows.append(
                {
                    "feedback_condition": scope_condition,
                    "count_type": "raw_combination",
                    "failed_check": name,
                    "count": int(count),
                    "percentage_of_attempts": count / total * 100 if total else float("nan"),
                }
            )
        for name, count in individual_counter.most_common():
            rows.append(
                {
                    "feedback_condition": scope_condition,
                    "count_type": "individual_check",
                    "failed_check": name,
                    "count": int(count),
                    "percentage_of_attempts": count / total * 100 if total else float("nan"),
                }
            )

    add_counts("overall", attempts)
    for condition, group in attempts.groupby("feedback_condition", sort=True):
        add_counts(str(condition), group)

    return pd.DataFrame(rows)


def stop_reasons_table(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_counts(condition: str, group: pd.DataFrame) -> None:
        total = len(group)
        for reason, count in group["stop_reason"].fillna("missing").value_counts().items():
            rows.append(
                {
                    "feedback_condition": condition,
                    "stop_reason": reason,
                    "count": int(count),
                    "percentage_of_results": count / total * 100 if total else float("nan"),
                }
            )

    add_counts("overall", results)
    for condition, group in results.groupby("feedback_condition", sort=True):
        add_counts(str(condition), group)
    return pd.DataFrame(rows)


def runtime_summary(attempts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped: list[tuple[str, pd.DataFrame]] = [("overall", attempts)]
    grouped.extend((str(name), group) for name, group in attempts.groupby("feedback_condition", sort=True))

    for condition, group in grouped:
        seconds = group["generation_seconds"].dropna()
        tokens = group["generated_token_count"].dropna()
        tps = group["generation_tokens_per_second"].dropna()
        rows.append(
            {
                "feedback_condition": condition,
                "timed_generation_attempts": int(seconds.count()),
                "total_generation_seconds": seconds.sum(),
                "mean_generation_seconds": seconds.mean(),
                "median_generation_seconds": seconds.median(),
                "total_generated_tokens": int(tokens.sum()),
                "mean_generated_tokens": tokens.mean(),
                "mean_generation_tokens_per_second": tps.mean(),
                "median_generation_tokens_per_second": tps.median(),
            }
        )
    return pd.DataFrame(rows)


def representative_examples(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    success_cols = [
        "row_id",
        "feedback_condition",
        "gold_label_name",
        "queries_used",
        "generation_attempts_used",
        "successful_similarity",
        "original_pred_label_name",
        "final_pred_label_name",
        "successful_candidate_text",
    ]
    failure_cols = [
        "row_id",
        "feedback_condition",
        "gold_label_name",
        "queries_used",
        "generation_attempts_used",
        "valid_candidates",
        "invalid_candidates",
        "stop_reason",
    ]

    successes = (
        results.loc[results["success"], success_cols]
        .sort_values(
            ["feedback_condition", "gold_label_name", "queries_used", "successful_similarity", "row_id"],
            ascending=[True, True, True, False, True],
        )
        .reset_index(drop=True)
    )
    failures = (
        results.loc[~results["success"], failure_cols]
        .sort_values(
            ["feedback_condition", "gold_label_name", "stop_reason", "queries_used", "row_id"],
            ascending=[True, True, True, True, True],
        )
        .reset_index(drop=True)
    )
    return successes, failures


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8")


def build_summary_json(
    meta: dict[str, Any],
    integrity_checks: dict[str, Any],
    tables: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    asr = tables["asr"].set_index("feedback_condition")
    paired = tables["paired"].iloc[0].to_dict()
    mcnemar = tables["mcnemar"].iloc[0].to_dict()
    validity = tables["validity"].set_index("feedback_condition")
    success_efficiency = tables["query_efficiency_successes_only"]

    success_query_mean = {
        str(row["feedback_condition"]): row["mean"]
        for _, row in success_efficiency[
            success_efficiency["metric"] == "queries_used"
        ].iterrows()
    }

    failed_individual = tables["failed_checks"][
        (tables["failed_checks"]["feedback_condition"] == "overall")
        & (tables["failed_checks"]["count_type"] == "individual_check")
    ].head(10)

    label_asr = float(asr.loc["label_only", "ASR"])
    score_asr = float(asr.loc["score_based", "ASR"])
    delta = score_asr - label_asr
    if delta > 0:
        headline = "Score-based feedback had a higher ASR than label-only feedback."
    elif delta < 0:
        headline = "Score-based feedback had a lower ASR than label-only feedback."
    else:
        headline = "Score-based and label-only feedback had the same ASR."

    return {
        "analysis_name": "main_gen3_query3",
        "created_by_script": str(Path(__file__).name),
        "raw_inputs": {
            "results_path": str(RESULTS_PATH),
            "attempts_path": str(ATTEMPTS_PATH),
            "metadata_path": str(META_PATH),
        },
        "run_id": meta.get("run_id"),
        "integrity_checks": integrity_checks,
        "headline_results": {
            "label_only_ASR": label_asr,
            "label_only_percentage": float(asr.loc["label_only", "percentage"]),
            "score_based_ASR": score_asr,
            "score_based_percentage": float(asr.loc["score_based", "percentage"]),
            "score_minus_label_ASR": delta,
            "paired_success_table": paired,
            "mcnemar": mcnemar,
            "mean_queries_among_successes": success_query_mean,
            "overall_validity_rate": float(validity.loc["overall", "validity_rate"]),
            "invalid_queried_count_overall": int(validity.loc["overall", "invalid_queried_count"]),
            "top_failed_individual_checks": failed_individual[
                ["failed_check", "count", "percentage_of_attempts"]
            ].to_dict(orient="records"),
        },
        "interpretation": (
            f"{headline} The exact paired McNemar/binomial result was p="
            f"{float(mcnemar['p_value']):.4g}; {mcnemar['interpretation']}"
        ),
        "analysis_outputs": {
            "table_asr_by_condition": "table_asr_by_condition.csv",
            "table_paired_success": "table_paired_success.csv",
            "table_mcnemar": "table_mcnemar.csv",
            "table_query_efficiency": "table_query_efficiency.csv",
            "table_query_efficiency_successes_only": "table_query_efficiency_successes_only.csv",
            "table_directional_success": "table_directional_success.csv",
            "table_validity_by_condition": "table_validity_by_condition.csv",
            "table_failed_checks": "table_failed_checks.csv",
            "table_stop_reasons": "table_stop_reasons.csv",
            "table_runtime_summary": "table_runtime_summary.csv",
            "representative_successes": "representative_successes.csv",
            "representative_failures": "representative_failures.csv",
        },
        "note": "Representative example files include all successes and all failures, sorted deterministically.",
    }


def print_console_summary(tables: dict[str, pd.DataFrame], summary: dict[str, Any]) -> None:
    asr = tables["asr"].set_index("feedback_condition")
    paired = tables["paired"].iloc[0]
    mcnemar = tables["mcnemar"].iloc[0]
    validity = tables["validity"].set_index("feedback_condition")
    qe_success = tables["query_efficiency_successes_only"]
    failed = tables["failed_checks"]

    def mean_metric(condition: str, metric: str) -> float:
        row = qe_success[
            (qe_success["feedback_condition"] == condition) & (qe_success["metric"] == metric)
        ]
        return float(row.iloc[0]["mean"]) if len(row) else float("nan")

    top_failed = failed[
        (failed["feedback_condition"] == "overall")
        & (failed["count_type"] == "individual_check")
    ].head(5)

    print("\n=== Final main analysis summary: main_gen3_query3 ===")
    print(
        "Primary ASR: "
        f"label_only {int(asr.loc['label_only', 'successes'])}/{int(asr.loc['label_only', 'total'])} "
        f"({asr.loc['label_only', 'percentage']:.2f}%), "
        f"score_based {int(asr.loc['score_based', 'successes'])}/{int(asr.loc['score_based', 'total'])} "
        f"({asr.loc['score_based', 'percentage']:.2f}%)."
    )
    print(
        "Paired table: "
        f"both_success={int(paired['both_success'])}, "
        f"label_only_only={int(paired['label_only_only'])}, "
        f"score_based_only={int(paired['score_based_only'])}, "
        f"neither_success={int(paired['neither_success'])}."
    )
    print(
        "McNemar exact: "
        f"b={int(mcnemar['b_label_only_success_score_based_failure'])}, "
        f"c={int(mcnemar['c_label_only_failure_score_based_success'])}, "
        f"p={float(mcnemar['p_value']):.4g}. {mcnemar['interpretation']}"
    )
    print(
        "Query efficiency among successes: "
        f"mean queries label_only={mean_metric('label_only', 'queries_used'):.2f}, "
        f"score_based={mean_metric('score_based', 'queries_used'):.2f}; "
        f"mean generation attempts label_only={mean_metric('label_only', 'generation_attempts_used'):.2f}, "
        f"score_based={mean_metric('score_based', 'generation_attempts_used'):.2f}."
    )
    print(
        "Validity rate: "
        f"overall {validity.loc['overall', 'validity_percentage']:.2f}% "
        f"(invalid queried={int(validity.loc['overall', 'invalid_queried_count'])})."
    )
    if len(top_failed):
        failed_text = ", ".join(
            f"{row.failed_check}={int(row.count)}" for row in top_failed.itertuples()
        )
        print(f"Top failed checks: {failed_text}.")
    else:
        print("Top failed checks: none recorded.")
    print("Main interpretation:", summary["interpretation"])
    print(f"Saved analysis outputs to: {ANALYSIS_DIR}")


def main() -> None:
    results, attempts, meta = read_inputs()
    integrity_checks = validate_inputs(results, attempts, meta)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    asr = asr_by_condition(results)
    paired, _paired_detail = paired_success_table(results)
    mcnemar = mcnemar_exact(paired)
    query_efficiency = summarize_numeric(results, ["queries_used", "generation_attempts_used"])
    query_efficiency_successes = summarize_numeric(
        results[results["success"]], ["queries_used", "generation_attempts_used"]
    )
    directional = directional_success(results)
    validity = validity_by_condition(attempts)
    failed_checks = failed_checks_table(attempts)
    stop_reasons = stop_reasons_table(results)
    runtime = runtime_summary(attempts)
    successes, failures = representative_examples(results)

    tables = {
        "asr": asr,
        "paired": paired,
        "mcnemar": mcnemar,
        "query_efficiency": query_efficiency,
        "query_efficiency_successes_only": query_efficiency_successes,
        "directional": directional,
        "validity": validity,
        "failed_checks": failed_checks,
        "stop_reasons": stop_reasons,
        "runtime": runtime,
        "successes": successes,
        "failures": failures,
    }
    summary = build_summary_json(meta, integrity_checks, tables)

    save_csv(asr, ANALYSIS_DIR / "table_asr_by_condition.csv")
    save_csv(paired, ANALYSIS_DIR / "table_paired_success.csv")
    save_csv(mcnemar, ANALYSIS_DIR / "table_mcnemar.csv")
    save_csv(query_efficiency, ANALYSIS_DIR / "table_query_efficiency.csv")
    save_csv(query_efficiency_successes, ANALYSIS_DIR / "table_query_efficiency_successes_only.csv")
    save_csv(directional, ANALYSIS_DIR / "table_directional_success.csv")
    save_csv(validity, ANALYSIS_DIR / "table_validity_by_condition.csv")
    save_csv(failed_checks, ANALYSIS_DIR / "table_failed_checks.csv")
    save_csv(stop_reasons, ANALYSIS_DIR / "table_stop_reasons.csv")
    save_csv(runtime, ANALYSIS_DIR / "table_runtime_summary.csv")
    save_csv(successes, ANALYSIS_DIR / "representative_successes.csv")
    save_csv(failures, ANALYSIS_DIR / "representative_failures.csv")

    with (ANALYSIS_DIR / "summary_main_gen3_query3.json").open("w", encoding="utf-8") as f:
        json.dump(to_builtin(summary), f, indent=2)
        f.write("\n")

    print_console_summary(tables, summary)


if __name__ == "__main__":
    main()
