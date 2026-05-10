from __future__ import annotations

# File-level purpose:
# - shared validity module for semantic-preserving paraphrase attacks
# - used identically across label-only and score-based conditions
# - decides whether a generated paraphrase is a valid adversarial candidate
# - does NOT decide attack success
#
# Design principle:
# A candidate paraphrase is valid only if it passes:
# 1) hard rejection checks (non-empty, non-identical, non-malformed, not excessively degraded)
# 2) semantic-preservation checks (mainly SBERT cosine similarity)
# 3) lightweight meaning-drift checks (numbers, dates, negation)

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import re

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


# =========================
# Frozen Config
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Frozen local SBERT model for semantic-similarity thresholding.
# Create it with Scripts_code/freeze_sbert_model.py before running experiments.
SBERT_MODEL_PATH = (
    PROJECT_ROOT
    / "Automated Deception Classifier (Projectfolder)"
    / "local_models"
    / "all-MiniLM-L6-v2"
)

DEVICE = "cpu"
MAX_LENGTH = 256

# Default thresholds.
# These should be treated as pilot-stage defaults and frozen before the main run.
MIN_SBERT_SIMILARITY = 0.86
#
# Morris et al. show that when semantic constraints are calibrated more strictly
# to human judgment, the minimum sentence-encoding similarity needs to be
# increased substantially. Their adjusted 0.98 threshold comes from a much
# tighter word-substitution-style setup, not a universal paraphrase threshold.

MIN_LENGTH_RATIO = 0.60
MIN_TOKEN_COUNT = 4

REJECT_IDENTICAL = True
CHECK_NUMBERS = True
CHECK_DATES = True
CHECK_NEGATION = True


NEGATION_PATTERN = re.compile(
    r"\b(?:no|not|never|none|nobody|nothing|nowhere|neither|nor|cannot|can't|won't|"
    r"don't|doesn't|didn't|isn't|aren't|wasn't|weren't|haven't|hasn't|hadn't|"
    r"shouldn't|wouldn't|couldn't|mustn't)\b",
    flags=re.IGNORECASE,
)

NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?\b")

YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")

MONTH_PATTERN = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    flags=re.IGNORECASE,
)

WEEKDAY_PATTERN = re.compile(
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    flags=re.IGNORECASE,
)

DATE_SLASH_PATTERN = re.compile(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b")


# =========================
# Output Objects
# =========================
@dataclass(frozen=True)
class ValidityCheckResult:
    check_name: str
    passed: bool
    value: Optional[object] = None
    threshold: Optional[object] = None
    message: str = ""


@dataclass(frozen=True)
class ValidityEvaluation:
    is_valid: bool
    failed_checks: list[str]
    semantic_similarity: Optional[float]
    check_results: list[ValidityCheckResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "is_valid": self.is_valid,
            "failed_checks": list(self.failed_checks),
            "semantic_similarity": self.semantic_similarity,
            "check_results": [
                {
                    "check_name": c.check_name,
                    "passed": c.passed,
                    "value": c.value,
                    "threshold": c.threshold,
                    "message": c.message,
                }
                for c in self.check_results
            ],
        }


# =========================
# Main Checker
# =========================
class ParaphraseValidityChecker:
    """
    Shared validity checker for LLM-generated paraphrases.

    This module does NOT determine whether an attack succeeded.
    It only determines whether a candidate paraphrase is valid enough
    to be treated as an eligible adversarial candidate.

    Success in later scripts should require BOTH:
    - validity passes here
    - victim-model prediction flip under budget
    """

    def __init__(
        self,
        sbert_model_path: Path = SBERT_MODEL_PATH,
        device: str = DEVICE,
        max_length: int = MAX_LENGTH,
        min_sbert_similarity: float = MIN_SBERT_SIMILARITY,
        min_length_ratio: float = MIN_LENGTH_RATIO,
        min_token_count: int = MIN_TOKEN_COUNT,
        reject_identical: bool = REJECT_IDENTICAL,
        check_numbers: bool = CHECK_NUMBERS,
        check_dates: bool = CHECK_DATES,
        check_negation: bool = CHECK_NEGATION,
    ) -> None:
        self.sbert_model_path = Path(sbert_model_path)
        self.device = torch.device(device)
        self.max_length = max_length

        self.min_sbert_similarity = min_sbert_similarity
        self.min_length_ratio = min_length_ratio
        self.min_token_count = min_token_count

        self.reject_identical = reject_identical
        self.check_numbers = check_numbers
        self.check_dates = check_dates
        self.check_negation = check_negation

        if not self.sbert_model_path.exists():
            raise FileNotFoundError(
                f"Frozen local SBERT path not found: {self.sbert_model_path}. "
                "Run Scripts_code/freeze_sbert_model.py before experiments."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.sbert_model_path,
            local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            self.sbert_model_path,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()

    # -------------------------
    # Metadata
    # -------------------------
    def get_metadata(self) -> dict[str, object]:
        return {
            "sbert_model_path": str(self.sbert_model_path),
            "device": str(self.device),
            "max_length": self.max_length,
            "min_sbert_similarity": self.min_sbert_similarity,
            "min_length_ratio": self.min_length_ratio,
            "min_token_count": self.min_token_count,
            "reject_identical": self.reject_identical,
            "check_numbers": self.check_numbers,
            "check_dates": self.check_dates,
            "check_negation": self.check_negation,
        }

    # -------------------------
    # Public API
    # -------------------------
    def evaluate(self, original_text: str, candidate_text: str) -> ValidityEvaluation:
        original = self._normalize_text(original_text, field_name="original_text")
        candidate = self._normalize_candidate_text(candidate_text)
        if isinstance(candidate, ValidityEvaluation):
            return candidate

        check_results: list[ValidityCheckResult] = []

        # --------
        # Stage A: hard rejection checks
        # --------
        check_results.append(self._check_non_empty(candidate))
        check_results.append(self._check_token_count(candidate))
        check_results.append(self._check_not_identical(original, candidate))
        check_results.append(self._check_length_ratio(original, candidate))
        check_results.append(self._check_basic_malformed(candidate))

        # Early stop if hard checks already fail.
        hard_failed = [c.check_name for c in check_results if not c.passed]
        if hard_failed:
            return ValidityEvaluation(
                is_valid=False,
                failed_checks=hard_failed,
                semantic_similarity=None,
                check_results=check_results,
            )

        # --------
        # Stage B: semantic preservation
        # --------
        semantic_similarity = self._compute_sbert_similarity(original, candidate)
        check_results.append(
            ValidityCheckResult(
                check_name="sbert_similarity",
                passed=semantic_similarity >= self.min_sbert_similarity,
                value=semantic_similarity,
                threshold=self.min_sbert_similarity,
                message=(
                    "SBERT cosine similarity between original and candidate."
                ),
            )
        )

        # --------
        # Stage C: lightweight meaning-drift checks
        # --------
        if self.check_numbers:
            check_results.append(self._check_numbers_consistency(original, candidate))

        if self.check_dates:
            check_results.append(self._check_dates_consistency(original, candidate))

        if self.check_negation:
            check_results.append(self._check_negation_consistency(original, candidate))

        failed_checks = [c.check_name for c in check_results if not c.passed]

        return ValidityEvaluation(
            is_valid=len(failed_checks) == 0,
            failed_checks=failed_checks,
            semantic_similarity=semantic_similarity,
            check_results=check_results,
        )

    # -------------------------
    # Hard checks
    # -------------------------
    def _check_non_empty(self, candidate: str) -> ValidityCheckResult:
        passed = candidate.strip() != ""
        return ValidityCheckResult(
            check_name="non_empty",
            passed=passed,
            value=len(candidate.strip()),
            threshold="> 0",
            message="Candidate must not be empty after normalization.",
        )

    def _check_token_count(self, candidate: str) -> ValidityCheckResult:
        token_count = self._simple_token_count(candidate)
        passed = token_count >= self.min_token_count
        return ValidityCheckResult(
            check_name="min_token_count",
            passed=passed,
            value=token_count,
            threshold=self.min_token_count,
            message="Candidate must have at least the minimum token count.",
        )

    def _check_not_identical(self, original: str, candidate: str) -> ValidityCheckResult:
        if not self.reject_identical:
            return ValidityCheckResult(
                check_name="not_identical",
                passed=True,
                value=None,
                threshold=None,
                message="Identical-output rejection disabled.",
            )

        passed = self._canonicalize_for_identity(original) != self._canonicalize_for_identity(candidate)
        return ValidityCheckResult(
            check_name="not_identical",
            passed=passed,
            value=passed,
            threshold=True,
            message="Candidate must not be trivially identical to the original.",
        )

    def _check_length_ratio(self, original: str, candidate: str) -> ValidityCheckResult:
        original_len = max(len(original), 1)
        candidate_len = len(candidate)
        ratio = candidate_len / original_len
        passed = ratio >= self.min_length_ratio
        return ValidityCheckResult(
            check_name="length_ratio",
            passed=passed,
            value=ratio,
            threshold=self.min_length_ratio,
            message="Candidate must not be excessively short relative to the original.",
        )

    def _check_basic_malformed(self, candidate: str) -> ValidityCheckResult:
        normalized = candidate.strip()

        only_punct = bool(normalized) and all(not ch.isalnum() for ch in normalized)
        repeated_char_run = self._has_extreme_repeated_characters(normalized)
        prompt_leakage = self._looks_like_prompt_meta_output(normalized)

        passed = not (only_punct or repeated_char_run or prompt_leakage)

        reasons = []
        if only_punct:
            reasons.append("only_punctuation")
        if repeated_char_run:
            reasons.append("repeated_characters")
        if prompt_leakage:
            reasons.append("prompt_meta_output")

        return ValidityCheckResult(
            check_name="basic_malformed_check",
            passed=passed,
            value=reasons if reasons else [],
            threshold=[],
            message="Reject clearly malformed, junk-like, or prompt-meta outputs.",
        )

    # -------------------------
    # Semantic similarity
    # -------------------------
    def _compute_sbert_similarity(self, original: str, candidate: str) -> float:
        original_embedding = self._encode_text(original)
        candidate_embedding = self._encode_text(candidate)
        similarity = F.cosine_similarity(original_embedding, candidate_embedding, dim=0)
        return float(similarity.item())

    def _encode_text(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.inference_mode():
            outputs = self.model(**encoded)
            last_hidden_state = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"]

            pooled = self._mean_pool(last_hidden_state, attention_mask)
            normalized = F.normalize(pooled, p=2, dim=1)

        return normalized[0]

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    # -------------------------
    # Lightweight meaning-drift checks
    # -------------------------
    def _check_numbers_consistency(self, original: str, candidate: str) -> ValidityCheckResult:
        original_numbers = self._extract_numbers(original)
        candidate_numbers = self._extract_numbers(candidate)

        passed = original_numbers == candidate_numbers

        return ValidityCheckResult(
            check_name="numbers_consistency",
            passed=passed,
            value={
                "original_numbers": original_numbers,
                "candidate_numbers": candidate_numbers,
            },
            threshold="exact_match",
            message="Reject obvious major number changes or dropped explicit numbers.",
        )

    def _check_dates_consistency(self, original: str, candidate: str) -> ValidityCheckResult:
        original_dates = self._extract_date_like_markers(original)
        candidate_dates = self._extract_date_like_markers(candidate)

        passed = original_dates == candidate_dates

        return ValidityCheckResult(
            check_name="dates_consistency",
            passed=passed,
            value={
                "original_dates": original_dates,
                "candidate_dates": candidate_dates,
            },
            threshold="exact_match",
            message="Reject obvious explicit date/month/year/weekday changes.",
        )

    def _check_negation_consistency(self, original: str, candidate: str) -> ValidityCheckResult:
        original_has_negation = self._has_negation(original)
        candidate_has_negation = self._has_negation(candidate)

        passed = original_has_negation == candidate_has_negation

        return ValidityCheckResult(
            check_name="negation_consistency",
            passed=passed,
            value={
                "original_has_negation": original_has_negation,
                "candidate_has_negation": candidate_has_negation,
            },
            threshold="same_boolean_value",
            message="Reject obvious simple negation reversals where detectable.",
        )

    # -------------------------
    # Helpers
    # -------------------------
    def _normalize_candidate_text(self, candidate_text: object) -> str | ValidityEvaluation:
        if not isinstance(candidate_text, str):
            return self._failed_candidate_evaluation(
                check_name="candidate_is_string",
                value=type(candidate_text).__name__,
                threshold="str",
                message=(
                    "Candidate must be a string. Non-string candidates are invalid "
                    "paraphrase outputs, not experiment-stopping errors."
                ),
            )

        normalized = re.sub(r"\s+", " ", candidate_text).strip()
        if normalized == "":
            return self._failed_candidate_evaluation(
                check_name="non_empty",
                value=0,
                threshold="> 0",
                message="Candidate must not be empty after normalization.",
            )

        return normalized

    @staticmethod
    def _failed_candidate_evaluation(
        check_name: str,
        value: object,
        threshold: object,
        message: str,
    ) -> ValidityEvaluation:
        failed_check = ValidityCheckResult(
            check_name=check_name,
            passed=False,
            value=value,
            threshold=threshold,
            message=message,
        )
        return ValidityEvaluation(
            is_valid=False,
            failed_checks=[check_name],
            semantic_similarity=None,
            check_results=[failed_check],
        )

    @staticmethod
    def _normalize_text(text: str, field_name: str) -> str:
        if not isinstance(text, str):
            raise TypeError(f"{field_name} must be str, got {type(text).__name__}")

        normalized = re.sub(r"\s+", " ", text).strip()
        if normalized == "":
            raise ValueError(f"{field_name} must not be empty after normalization")

        return normalized

    @staticmethod
    def _simple_token_count(text: str) -> int:
        return len(text.split())

    @staticmethod
    def _canonicalize_for_identity(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _has_extreme_repeated_characters(text: str, run_length: int = 6) -> bool:
        if not text:
            return False
        pattern = re.compile(rf"(.)\1{{{run_length - 1},}}")
        return bool(pattern.search(text))

    @staticmethod
    def _looks_like_prompt_meta_output(text: str) -> bool:
        lowered = text.lower()
        meta_markers = [
            "here is the paraphrase",
            "here's the paraphrase",
            "paraphrased version",
            "rewritten version",
        ]
        starts_with_meta = lowered.startswith(("sure,", "certainly,"))
        return starts_with_meta or any(marker in lowered for marker in meta_markers)

    @staticmethod
    def _extract_numbers(text: str) -> list[str]:
        return NUMBER_PATTERN.findall(text)

    @staticmethod
    def _extract_date_like_markers(text: str) -> dict[str, list[str]]:
        return {
            "years": [x.lower() for x in YEAR_PATTERN.findall(text)],
            "months": [x.lower() for x in MONTH_PATTERN.findall(text)],
            "weekdays": [x.lower() for x in WEEKDAY_PATTERN.findall(text)],
            "slash_dates": [x.lower() for x in DATE_SLASH_PATTERN.findall(text)],
        }

    @staticmethod
    def _has_negation(text: str) -> bool:
        return bool(NEGATION_PATTERN.search(text))


if __name__ == "__main__":
    checker = ParaphraseValidityChecker()

    original_text = "I went to a restaurant with my family last weekend and we stayed for 2 hours."
    candidate_good = "Last weekend, I went out to eat with my family and we were there for 2 hours."
    candidate_bad = "Here is the paraphrase: I did not go to a restaurant with my family and we stayed for 5 hours."

    print("Checker metadata:")
    print(checker.get_metadata())

    print("\nGood candidate evaluation:")
    good_result = checker.evaluate(original_text, candidate_good)
    print(good_result.as_dict())

    print("\nBad candidate evaluation:")
    bad_result = checker.evaluate(original_text, candidate_bad)
    print(bad_result.as_dict())
