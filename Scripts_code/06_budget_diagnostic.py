from __future__ import annotations

# File-level purpose:
# - run controlled budget diagnostics on the frozen pilot/train-dev pool
# - delegate each actual attack run to 05_attack_pilot.py
# - summarize budget effects without touching the final/main pool

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "Scripts_code"
OUTPUT_DIR = SCRIPTS_DIR / "outputs"
ATTACK_SCRIPT_PATH = SCRIPTS_DIR / "05_attack_pilot.py"

LOCAL_LLM_PROMPT_VERSION_V3_DIVERSE_CONSERVATIVE = (
    "qwen3_paraphrase_feedback_v3_diverse_conservative"
)
DEFAULT_LLAMA_CPP_SERVER_URL = "http://127.0.0.1:8080/v1"
DEFAULT_LLAMA_CPP_SERVER_MODEL = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
DEFAULT_GGUF_MODEL_PATH = (
    PROJECT_ROOT
    / "Automated Deception Classifier (Projectfolder)"
    / "local_models"
    / "Qwen3-4B-Instruct-2507-GGUF"
    / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
)

BUDGET_CONFIGS = {
    "gen3_query3": (3, 3),
    "gen6_query3": (6, 3),
    "gen10_query5": (10, 5),
    "gen20_query10": (20, 10),
}

SUMMARY_COLUMNS = [
    "output_tag",
    "prompt_version",
    "max_generation_attempts",
    "max_classifier_queries",
    "result_rows",
    "attempt_rows",
    "valid_attempts",
    "invalid_attempts",
    "classifier_queries",
    "successes_total",
    "successes_label_only",
    "successes_score_based",
    "ASR_label_only",
    "ASR_score_based",
    "avg_queries_label_only",
    "avg_queries_score_based",
    "avg_generation_attempts_label_only",
    "avg_generation_attempts_score_based",
    "most_common_failed_checks",
    "integrity_checks_passed",
    "avg_generation_seconds",
    "avg_tokens_per_second",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run budget diagnostics for the pilot local-LLM attack."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(BUDGET_CONFIGS),
        choices=list(BUDGET_CONFIGS),
        help="Budget configs to run.",
    )
    parser.add_argument(
        "--prompt-version",
        default=LOCAL_LLM_PROMPT_VERSION_V3_DIVERSE_CONSERVATIVE,
        help="Local LLM prompt version to use for all diagnostic configs.",
    )
    parser.add_argument(
        "--summary-path",
        default=str(OUTPUT_DIR / "budget_diagnostic_summary.csv"),
        help="Path for the compact summary CSV.",
    )
    parser.add_argument(
        "--server-url",
        default=DEFAULT_LLAMA_CPP_SERVER_URL,
        help="OpenAI-compatible llama.cpp server URL.",
    )
    parser.add_argument(
        "--server-model",
        default=DEFAULT_LLAMA_CPP_SERVER_MODEL,
        help="Model name sent to the llama.cpp server.",
    )
    parser.add_argument(
        "--gguf-model-path",
        default=str(DEFAULT_GGUF_MODEL_PATH),
        help="GGUF model path recorded and hashed by 05_attack_pilot.py.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.45,
        help="Shared local LLM temperature for all configs.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.90,
        help="Shared local LLM top_p for all configs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Shared local LLM top_k for all configs.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05,
        help="Shared local LLM repetition penalty for all configs.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=160,
        help="Shared local LLM max generated tokens for all configs.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional pilot row limit for smoke-testing the diagnostic script.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Pass --overwrite to each delegated pilot attack run.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level passed to each delegated pilot attack run.",
    )
    return parser.parse_args()


def attack_output_paths(output_tag: str) -> tuple[Path, Path, Path]:
    return (
        OUTPUT_DIR / f"attack_attempts_pilot_local_llm_{output_tag}.csv",
        OUTPUT_DIR / f"attack_results_pilot_local_llm_{output_tag}.csv",
        OUTPUT_DIR / f"attack_pilot_meta_local_llm_{output_tag}.json",
    )


def run_attack_config(args: argparse.Namespace, output_tag: str) -> None:
    max_generation_attempts, max_classifier_queries = BUDGET_CONFIGS[output_tag]
    command = [
        sys.executable,
        str(ATTACK_SCRIPT_PATH),
        "--paraphraser",
        "local_llm",
        "--local-llm-backend",
        "llama_cpp_server",
        "--local-llm-server-url",
        args.server_url,
        "--local-llm-server-model",
        args.server_model,
        "--local-llm-gguf-model-path",
        args.gguf_model_path,
        "--local-llm-temperature",
        str(args.temperature),
        "--local-llm-top-p",
        str(args.top_p),
        "--local-llm-top-k",
        str(args.top_k),
        "--local-llm-repetition-penalty",
        str(args.repetition_penalty),
        "--local-llm-max-new-tokens",
        str(args.max_new_tokens),
        "--local-llm-prompt-version",
        args.prompt_version,
        "--max-generation-attempts",
        str(max_generation_attempts),
        "--max-classifier-queries",
        str(max_classifier_queries),
        "--output-tag",
        output_tag,
        "--log-level",
        args.log_level,
    ]
    if args.max_examples is not None:
        command.extend(["--max-examples", str(args.max_examples)])
    if args.overwrite:
        command.append("--overwrite")

    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes"}
    )


def mean_for_condition(
    results_df: pd.DataFrame,
    condition: str,
    column: str,
) -> float | None:
    subset = results_df[results_df["feedback_condition"].astype(str) == condition]
    if subset.empty:
        return None
    values = pd.to_numeric(subset[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def asr_for_condition(results_df: pd.DataFrame, condition: str) -> float | None:
    subset = results_df[results_df["feedback_condition"].astype(str) == condition]
    if subset.empty:
        return None
    return float(bool_series(subset["success"]).mean())


def count_successes(results_df: pd.DataFrame, condition: str | None = None) -> int:
    subset = results_df
    if condition is not None:
        subset = subset[subset["feedback_condition"].astype(str) == condition]
    if subset.empty:
        return 0
    return int(bool_series(subset["success"]).sum())


def most_common_failed_checks(attempts_df: pd.DataFrame, limit: int = 5) -> str:
    counter: Counter[str] = Counter()
    if "failed_checks" not in attempts_df.columns:
        return "[]"

    for raw_value in attempts_df["failed_checks"].fillna("[]"):
        try:
            checks = json.loads(str(raw_value))
        except json.JSONDecodeError:
            checks = [str(raw_value)]
        if isinstance(checks, list):
            counter.update(str(check) for check in checks)
        elif checks:
            counter.update([str(checks)])

    return json.dumps(counter.most_common(limit), ensure_ascii=True)


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"Metadata file has unexpected shape: {metadata_path}")
    return loaded


def summarize_config(output_tag: str) -> dict[str, object]:
    attempts_path, results_path, metadata_path = attack_output_paths(output_tag)
    attempts_df = pd.read_csv(attempts_path)
    results_df = pd.read_csv(results_path)
    metadata = load_metadata(metadata_path)

    is_valid = bool_series(attempts_df["is_valid"]) if not attempts_df.empty else []
    queried_classifier = (
        bool_series(attempts_df["queried_classifier"]) if not attempts_df.empty else []
    )
    generation_seconds = pd.to_numeric(
        attempts_df.get("generation_seconds", pd.Series(dtype="float64")),
        errors="coerce",
    ).dropna()
    tokens_per_second = pd.to_numeric(
        attempts_df.get("generation_tokens_per_second", pd.Series(dtype="float64")),
        errors="coerce",
    ).dropna()

    max_generation_attempts, max_classifier_queries = BUDGET_CONFIGS[output_tag]
    return {
        "output_tag": output_tag,
        "prompt_version": metadata.get("prompt_version")
        or metadata.get("paraphraser", {}).get("prompt_version"),
        "max_generation_attempts": max_generation_attempts,
        "max_classifier_queries": max_classifier_queries,
        "result_rows": int(len(results_df)),
        "attempt_rows": int(len(attempts_df)),
        "valid_attempts": int(is_valid.sum()) if len(attempts_df) else 0,
        "invalid_attempts": int((~is_valid).sum()) if len(attempts_df) else 0,
        "classifier_queries": int(queried_classifier.sum()) if len(attempts_df) else 0,
        "successes_total": count_successes(results_df),
        "successes_label_only": count_successes(results_df, "label_only"),
        "successes_score_based": count_successes(results_df, "score_based"),
        "ASR_label_only": asr_for_condition(results_df, "label_only"),
        "ASR_score_based": asr_for_condition(results_df, "score_based"),
        "avg_queries_label_only": mean_for_condition(
            results_df, "label_only", "queries_used"
        ),
        "avg_queries_score_based": mean_for_condition(
            results_df, "score_based", "queries_used"
        ),
        "avg_generation_attempts_label_only": mean_for_condition(
            results_df, "label_only", "generation_attempts_used"
        ),
        "avg_generation_attempts_score_based": mean_for_condition(
            results_df, "score_based", "generation_attempts_used"
        ),
        "most_common_failed_checks": most_common_failed_checks(attempts_df),
        "integrity_checks_passed": bool(metadata.get("integrity_checks_passed")),
        "avg_generation_seconds": float(generation_seconds.mean())
        if not generation_seconds.empty
        else None,
        "avg_tokens_per_second": float(tokens_per_second.mean())
        if not tokens_per_second.empty
        else None,
    }


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for output_tag in args.configs:
        run_attack_config(args, output_tag)
        rows.append(summarize_config(output_tag))

    summary_df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    summary_path = Path(args.summary_path)
    if not summary_path.is_absolute():
        summary_path = PROJECT_ROOT / summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"Budget diagnostic summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
