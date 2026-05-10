from __future__ import annotations

# File-level purpose:
# - fixed pretrained DistilBERT victim model
# - one shared inference path
# - two exposed feedback modes
# - explicit query counting per condition
# - no attack logic inside
#
# Query-count definition used here:
# - a query is counted only after local input validation/preprocessing succeeds
#   and the victim model is actually queried

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# =========================
# Frozen Inference Config
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "Automated Deception Classifier (Projectfolder)" / "DistilBERT"
TOKENIZER_PATH = MODEL_PATH

MAX_LENGTH = 512
SINGLE_EXAMPLE_MODE = True
DEVICE = "cpu"

ID_TO_LABEL = {
    0: "truthful",
    1: "deceptive",
}

LABEL_ONLY_CONDITION = "label_only"
SCORE_BASED_CONDITION = "score_based"


# =========================
# Output Objects
# =========================
@dataclass(frozen=True)
class FullPredictionResult:
    predicted_label_id: int
    predicted_label_name: str
    probabilities: list[float]
    confidence: float
    logits: list[float]
    query_count: int
    total_query_count: int
    feedback_condition: str


@dataclass(frozen=True)
class LabelOnlyOutput:
    predicted_label_id: int
    predicted_label_name: str
    query_count: int
    total_query_count: int


@dataclass(frozen=True)
class ScoreBasedOutput:
    predicted_label_id: int
    predicted_label_name: str
    confidence: float
    query_count: int
    total_query_count: int


# =========================
# Wrapper
# =========================
class DeceptionClassifierWrapper:
    """
    Narrow victim-model interface for controlled attack experiments.

    Design goal:
    - same underlying model and inference path in both conditions
    - only the exposed feedback differs:
        1) label-only
        2) label + predicted-class confidence

    This file should NOT contain:
    - attack generation
    - validity checks
    - stopping rules
    - experiment orchestration
    - experiment logging
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        tokenizer_path: Path = TOKENIZER_PATH,
        max_length: int = MAX_LENGTH,
        device: str = DEVICE,
        single_example_mode: bool = SINGLE_EXAMPLE_MODE,
        id_to_label: Optional[dict[int, str]] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.max_length = max_length
        self.single_example_mode = single_example_mode
        self.device = torch.device(device)
        self.id_to_label = dict(ID_TO_LABEL if id_to_label is None else id_to_label)

        if not self.single_example_mode:
            raise ValueError("This wrapper is configured for single-example queries only.")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer path not found: {self.tokenizer_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            local_files_only=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()

        self._validate_label_mapping()

        self._total_query_count = 0
        self._query_counts_by_condition = {
            LABEL_ONLY_CONDITION: 0,
            SCORE_BASED_CONDITION: 0,
        }

    # -------------------------
    # Query counting
    # -------------------------
    @property
    def total_query_count(self) -> int:
        return self._total_query_count

    @property
    def query_counts_by_condition(self) -> dict[str, int]:
        return dict(self._query_counts_by_condition)

    def reset_query_counts(self) -> None:
        self._total_query_count = 0
        for condition in self._query_counts_by_condition:
            self._query_counts_by_condition[condition] = 0

    # -------------------------
    # Metadata
    # -------------------------
    def get_metadata(self) -> dict[str, object]:
        return {
            "model_path": str(self.model_path),
            "tokenizer_path": str(self.tokenizer_path),
            "max_length": self.max_length,
            "device": str(self.device),
            "single_example_mode": self.single_example_mode,
            "id_to_label": dict(self.id_to_label),
        }

    # -------------------------
    # Public feedback methods
    # -------------------------
    def predict_label(self, text: str) -> LabelOnlyOutput:
        result = self._predict_full(text, feedback_condition=LABEL_ONLY_CONDITION)
        return LabelOnlyOutput(
            predicted_label_id=result.predicted_label_id,
            predicted_label_name=result.predicted_label_name,
            query_count=result.query_count,
            total_query_count=result.total_query_count,
        )

    def predict_label_and_confidence(self, text: str) -> ScoreBasedOutput:
        result = self._predict_full(text, feedback_condition=SCORE_BASED_CONDITION)
        return ScoreBasedOutput(
            predicted_label_id=result.predicted_label_id,
            predicted_label_name=result.predicted_label_name,
            confidence=result.confidence,
            query_count=result.query_count,
            total_query_count=result.total_query_count,
        )

    # -------------------------
    # Shared inference path
    # -------------------------
    def _predict_full(self, text: str, feedback_condition: str) -> FullPredictionResult:
        normalized_text = self._normalize_text(text)
        encodings = self._prepare_inputs(normalized_text)

        # Count only successful classifier calls.
        query_count, total_query_count = self._register_query(feedback_condition)

        with torch.inference_mode():
            outputs = self.model(**encodings)
            logits_tensor = outputs.logits[0]
            probabilities_tensor = torch.softmax(logits_tensor, dim=-1)
            predicted_label_id = int(torch.argmax(probabilities_tensor).item())

        logits = [float(value) for value in logits_tensor.detach().cpu().tolist()]
        probabilities = [
            float(value) for value in probabilities_tensor.detach().cpu().tolist()
        ]
        confidence = float(probabilities[predicted_label_id])

        predicted_label_name = self.id_to_label[predicted_label_id]

        return FullPredictionResult(
            predicted_label_id=predicted_label_id,
            predicted_label_name=predicted_label_name,
            probabilities=probabilities,
            confidence=confidence,
            logits=logits,
            query_count=query_count,
            total_query_count=total_query_count,
            feedback_condition=feedback_condition,
        )

    def _prepare_inputs(self, text: str) -> dict[str, torch.Tensor]:
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in encodings.items()}

    def _register_query(self, feedback_condition: str) -> tuple[int, int]:
        if feedback_condition not in self._query_counts_by_condition:
            raise ValueError(f"Unknown feedback condition: {feedback_condition}")

        self._query_counts_by_condition[feedback_condition] += 1
        self._total_query_count += 1

        return (
            self._query_counts_by_condition[feedback_condition],
            self._total_query_count,
        )

    # -------------------------
    # Validation helpers
    # -------------------------
    def _normalize_text(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")

        normalized_text = text.strip()
        if normalized_text == "":
            raise ValueError("text must not be empty after normalization")

        return normalized_text

    def _validate_label_mapping(self) -> None:
        num_labels = int(getattr(self.model.config, "num_labels", len(self.id_to_label)))
        expected_ids = set(range(num_labels))
        configured_ids = set(self.id_to_label.keys())

        if configured_ids != expected_ids:
            raise ValueError(
                "id_to_label must match the model output labels. "
                f"Model label ids: {sorted(expected_ids)}; "
                f"configured label ids: {sorted(configured_ids)}"
            )


if __name__ == "__main__":
    wrapper = DeceptionClassifierWrapper()

    test_text = "I went to a restaurant with my family last weekend."

    print("Wrapper metadata:")
    print(wrapper.get_metadata())

    label_only_output = wrapper.predict_label(test_text)
    print("\nLabel-only output:")
    print(label_only_output)

    score_based_output = wrapper.predict_label_and_confidence(test_text)
    print("\nScore-based output:")
    print(score_based_output)

    print("\nQuery counts by condition:")
    print(wrapper.query_counts_by_condition)

    print("\nTotal query count:")
    print(wrapper.total_query_count)