from __future__ import annotations

# File-level purpose:
# - first end-to-end pilot attack loop
# - connects frozen pilot pool, classifier wrapper, validity checker, and a
#   modular paraphrase generator placeholder
# - keeps query-budget accounting separate from original prediction verification

import argparse
import hashlib
import importlib.util
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd


# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "Scripts_code"
OUTPUT_DIR = SCRIPTS_DIR / "outputs"

PILOT_POOL_PATH = OUTPUT_DIR / "candidate_pool_pilot_20.csv"
MAIN_POOL_PATH = OUTPUT_DIR / "candidate_pool_main_160.csv"
EXPLORATORY_POOL_PATH = OUTPUT_DIR / "candidate_pool_exploratory_remaining.csv"

CLASSIFIER_WRAPPER_PATH = SCRIPTS_DIR / "03_classifier_wrapper.py"
VALIDITY_CHECKS_PATH = SCRIPTS_DIR / "04_validity_checks.py"


# =========================
# Config
# =========================
MAX_CLASSIFIER_QUERIES = 20
MAX_GENERATION_ATTEMPTS = 40

SEED = 42
RUN_NAME = "pilot_attack"
DEFAULT_PARAPHRASER_MODE = "placeholder"
INVALID_DEBUG_PARAPHRASER_MODE = "invalid_debug"
LOCAL_LLM_PARAPHRASER_MODE = "local_llm"

DEFAULT_LOCAL_LLM_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_LOCAL_LLM_MODEL_PATH = (
    PROJECT_ROOT
    / "Automated Deception Classifier (Projectfolder)"
    / "local_models"
    / "Qwen3-4B-Instruct-2507"
)
DEFAULT_LOCAL_LLM_LOCAL_FILES_ONLY = True
DEFAULT_LOCAL_LLM_DEVICE = "auto"
DEFAULT_LOCAL_LLM_DTYPE = "auto"
DEFAULT_LOCAL_LLM_MAX_NEW_TOKENS = 120
DEFAULT_LOCAL_LLM_TEMPERATURE = 0.7
DEFAULT_LOCAL_LLM_TOP_P = 0.9
DEFAULT_LOCAL_LLM_TOP_K = 50
DEFAULT_LOCAL_LLM_REPETITION_PENALTY = 1.05
DEFAULT_LOCAL_LLM_DO_SAMPLE = True
DEFAULT_ALLOW_LOCAL_LLM_CPU = False
DEFAULT_LOCAL_LLM_LOAD_IN_4BIT = False
DEFAULT_LOCAL_LLM_LOAD_IN_8BIT = False
LOCAL_LLM_TRANSFORMERS_BACKEND = "transformers"
LOCAL_LLM_LLAMA_CPP_SERVER_BACKEND = "llama_cpp_server"
DEFAULT_LOCAL_LLM_BACKEND = LOCAL_LLM_TRANSFORMERS_BACKEND
DEFAULT_LOCAL_LLM_SERVER_URL = "http://127.0.0.1:8080/v1"
DEFAULT_LOCAL_LLM_SERVER_MODEL = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
DEFAULT_LOCAL_LLM_GGUF_MODEL_PATH = ""
DEFAULT_LOCAL_LLM_CONTEXT_SIZE = 4096
DEFAULT_LOCAL_LLM_GPU_LAYERS = "auto"
DEFAULT_ALLOW_NONLOCAL_LLM_SERVER = False
DEFAULT_LOCAL_LLM_REQUEST_TIMEOUT_SECONDS = 300.0
LOCAL_LLM_PROMPT_VERSION_V2_CONSERVATIVE = "qwen3_paraphrase_feedback_v2_conservative"
LOCAL_LLM_PROMPT_VERSION_V3_DIVERSE_CONSERVATIVE = (
    "qwen3_paraphrase_feedback_v3_diverse_conservative"
)
LOCAL_LLM_PROMPT_VERSION = LOCAL_LLM_PROMPT_VERSION_V2_CONSERVATIVE
LOCAL_LLM_PROMPT_VERSIONS = (
    LOCAL_LLM_PROMPT_VERSION_V2_CONSERVATIVE,
    LOCAL_LLM_PROMPT_VERSION_V3_DIVERSE_CONSERVATIVE,
)

PILOT_POOL_NAME = "pilot"
MAIN_POOL_NAME = "main"
EXPLORATORY_POOL_NAME = "exploratory"
POOL_CONFIGS = {
    PILOT_POOL_NAME: {
        "path": PILOT_POOL_PATH,
        "expected_source_split": "train_dev",
        "expected_full_pool_size": 20,
    },
    MAIN_POOL_NAME: {
        "path": MAIN_POOL_PATH,
        "expected_source_split": "test_final",
        "expected_full_pool_size": 160,
    },
    EXPLORATORY_POOL_NAME: {
        "path": EXPLORATORY_POOL_PATH,
        "expected_source_split": "test_final",
        "expected_full_pool_size": 229,
    },
}
ATTACK_TEXT_COLUMN = "attack_text"
ROW_ID_COLUMN = "row_id"
GOLD_LABEL_NAME_COLUMN = "gold_label_name"
CORRECT_COLUMN = "correct"
SOURCE_SPLIT_COLUMN = "source_split"
SOURCE_TEXT_COLUMN = "source_text_column"

ATTEMPT_COLUMNS = [
    "run_id",
    "row_id",
    "feedback_condition",
    "gold_label_name",
    "original_pred_label_name",
    "original_confidence",
    "attempt_index",
    "classifier_query_index",
    "original_attack_text",
    "candidate_text",
    "is_valid",
    "failed_checks",
    "semantic_similarity",
    "queried_classifier",
    "candidate_pred_label_name",
    "candidate_confidence",
    "is_success",
    "stop_reason",
    "generation_seconds",
    "generated_token_count",
    "generation_tokens_per_second",
]

RESULT_COLUMNS = [
    "run_id",
    "row_id",
    "feedback_condition",
    "gold_label_name",
    "source_split",
    "source_text_column",
    "original_pred_label_name",
    "original_confidence",
    "success",
    "queries_used",
    "generation_attempts_used",
    "valid_candidates",
    "invalid_candidates",
    "successful_candidate_text",
    "successful_similarity",
    "final_pred_label_name",
    "stop_reason",
]


# =========================
# Dynamic imports
# =========================
def load_module_from_path(module_name: str, module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


classifier_module = load_module_from_path(
    "classifier_wrapper_03",
    CLASSIFIER_WRAPPER_PATH,
)
validity_module = load_module_from_path(
    "validity_checks_04",
    VALIDITY_CHECKS_PATH,
)

DeceptionClassifierWrapper = classifier_module.DeceptionClassifierWrapper
ParaphraseValidityChecker = validity_module.ParaphraseValidityChecker
LABEL_ONLY_CONDITION = classifier_module.LABEL_ONLY_CONDITION
SCORE_BASED_CONDITION = classifier_module.SCORE_BASED_CONDITION
FEEDBACK_CONDITIONS = (LABEL_ONLY_CONDITION, SCORE_BASED_CONDITION)


# =========================
# Output objects
# =========================
@dataclass(frozen=True)
class AttackFeedback:
    feedback_condition: str
    predicted_label_name: str
    confidence: float | None


@dataclass(frozen=True)
class GenerationRuntime:
    generation_seconds: float | None = None
    generated_token_count: int | None = None
    generation_tokens_per_second: float | None = None


class Paraphraser(Protocol):
    def generate(
        self,
        original_text: str,
        attempt_index: int,
        feedback: AttackFeedback,
        row_id: int,
    ) -> str:
        ...

    def get_metadata(self) -> dict[str, object]:
        ...

    def get_last_generation_runtime(self) -> GenerationRuntime:
        ...


@dataclass(frozen=True)
class CandidatePrediction:
    predicted_label_name: str
    confidence: float | None


@dataclass(frozen=True)
class OutputPaths:
    attempts: Path
    results: Path
    metadata: Path


# =========================
# Placeholder paraphraser
# =========================
class PlaceholderParaphraser:
    """
    Deterministic no-API paraphraser for pipeline testing.

    It intentionally makes tiny, meaning-preserving framing changes rather than
    a strong attack. This keeps the loop runnable before connecting an LLM while
    still producing non-identical candidates for the validity checker.
    """

    PREFIXES = [
        "Overall,",
        "In short,",
        "Put simply,",
        "In essence,",
        "Broadly speaking,",
        "As I recall,",
        "To summarize,",
        "In other words,",
        "From my perspective,",
        "Generally,",
        "All in all,",
        "In my view,",
        "To describe it plainly,",
        "Looking back,",
        "The way I remember it,",
        "For clarity,",
        "In plain terms,",
        "Said another way,",
        "As best I can explain,",
        "In my account,",
        "To put it another way,",
        "In simple terms,",
        "From what I remember,",
        "To state it directly,",
        "In a straightforward telling,",
        "As I experienced it,",
        "The situation was this:",
        "Here is the same account:",
        "In my own words,",
        "To frame the story,",
        "The basic account is that",
        "As a matter of context,",
        "What happened was that",
        "The main point is that",
        "My recollection is that",
        "The event unfolded like this:",
        "The story, as I remember it, is that",
        "What I mean is that",
        "The account can be put this way:",
        "A plain version is that",
    ]

    def __init__(self) -> None:
        self._last_generation_runtime = GenerationRuntime()

    def generate(
        self,
        original_text: str,
        attempt_index: int,
        feedback: AttackFeedback,
        row_id: int,
    ) -> str:
        del row_id
        del feedback
        self._last_generation_runtime = GenerationRuntime()
        prefix = self.PREFIXES[(attempt_index - 1) % len(self.PREFIXES)]
        return f"{prefix} {normalize_text(original_text)}"

    def get_last_generation_runtime(self) -> GenerationRuntime:
        return self._last_generation_runtime

    def get_metadata(self) -> dict[str, object]:
        return {
            "mode": DEFAULT_PARAPHRASER_MODE,
            "name": type(self).__name__,
            "uses_external_api": False,
            "description": (
                "Deterministic no-API paraphraser used so the pilot loop can "
                "be tested before connecting an LLM."
            ),
        }


class InvalidDebugParaphraser(PlaceholderParaphraser):
    """
    Debug paraphraser that intentionally emits invalid candidates.

    Odd attempts return the unchanged original text, which should fail the
    not-identical validity check. Even attempts use the normal placeholder
    behavior, so the loop can continue to exercise valid classifier queries.
    """

    def generate(
        self,
        original_text: str,
        attempt_index: int,
        feedback: AttackFeedback,
        row_id: int,
    ) -> str:
        if attempt_index % 2 == 1:
            return normalize_text(original_text)
        return super().generate(original_text, attempt_index, feedback, row_id)

    def get_metadata(self) -> dict[str, object]:
        metadata = super().get_metadata()
        metadata.update(
            {
                "mode": INVALID_DEBUG_PARAPHRASER_MODE,
                "name": type(self).__name__,
                "description": (
                    "Debug paraphraser that alternates invalid identical outputs "
                    "with the deterministic placeholder paraphraser."
                ),
            }
        )
        return metadata


class LocalLLMParaphraser:
    """
    Fixed local open-source LLM paraphraser for the pilot attack.

    The model and generation settings are shared across feedback conditions.
    The only prompt difference is the feedback block: label-only sees the
    current predicted label; score-based sees that label plus confidence.
    """

    SYSTEM_PROMPT_TEMPLATE = """You are a conservative paraphrasing system for a controlled text robustness experiment.
Your task is to make the smallest natural rewrite possible while preserving the exact meaning and the original truthfulness/deceptiveness status.
Output exactly one complete rewritten paragraph and nothing else."""

    USER_PROMPT_TEMPLATE_V2_CONSERVATIVE = """Rewrite the text below as one conservative, meaning-preserving paraphrase.
Make the smallest natural paraphrase possible. Do not summarize, interpret, correct, or embellish the text.

Hard preservation rules:
- Preserve the exact same meaning.
- Preserve the original truthfulness/deceptiveness status.
- Preserve all numbers exactly as written.
- Preserve all dates, years, date-like strings, and time markers exactly as written.
- Preserve names and locations exactly as written.
- Preserve negation exactly.
- Preserve the order of events.
- Keep roughly the same length.
- Do not add new details.
- Do not remove important details.
- Do not fix or normalize unusual source formatting. For example, if the source contains "2018I", keep that exact form.
- Use natural English only where it does not change the details above.

Output rules:
- Output one complete rewritten paragraph only.
- No explanations.
- No bullet points.
- No quotes around the answer.
- Do not write phrases like "Here is the paraphrase".
- Do not mention the classifier, attack, labels, confidence, or feedback in the output.

Feedback available:
{feedback_block}

Original text:
{original_text}

Rewritten paragraph:"""

    USER_PROMPT_TEMPLATE_V3_DIVERSE_CONSERVATIVE = """Rewrite the text below as one conservative, meaning-preserving paraphrase.
Make a small but real paraphrase. Do not copy the original text. Do not summarize, interpret, correct, or embellish the text.

Hard preservation rules:
- Preserve the exact same meaning.
- Preserve the original truthfulness/deceptiveness status.
- Preserve all numbers exactly as written.
- Preserve all dates, years, date-like strings, and time markers exactly as written.
- Preserve names and locations exactly as written.
- Preserve negation exactly.
- Preserve the order of events.
- Keep roughly the same length.
- Do not add any details.
- Do not remove any details.
- Do not summarize.
- Do not fix or normalize unusual source formatting. For example, if the source contains "2018I", keep that exact form.
- Use natural English only where it does not change the details above.

Required rewrite:
- The output must not be identical to the original text.
- Change wording or sentence structure in at least two small places.
- Keep every factual detail unchanged.
- Make only conservative edits that preserve meaning.

Rewrite strategy for this attempt:
{rewrite_strategy_block}

Output rules:
- Output one complete rewritten paragraph only.
- No explanations.
- No bullet points.
- No quotes around the answer.
- Do not write phrases like "Here is the paraphrase".
- Do not mention the classifier, attack, labels, confidence, or feedback in the output.

Feedback available:
{feedback_block}

Original text:
{original_text}

Rewritten paragraph:"""

    USER_PROMPT_TEMPLATE = USER_PROMPT_TEMPLATE_V2_CONSERVATIVE

    LABEL_ONLY_FEEDBACK_TEMPLATE = "- Current predicted label: {predicted_label_name}"
    SCORE_BASED_FEEDBACK_TEMPLATE = (
        "- Current predicted label: {predicted_label_name}\n"
        "- Current predicted-class confidence: {confidence:.4f}"
    )

    def __init__(
        self,
        model_id: str = DEFAULT_LOCAL_LLM_MODEL_ID,
        model_path: Path | None = DEFAULT_LOCAL_LLM_MODEL_PATH,
        local_files_only: bool = DEFAULT_LOCAL_LLM_LOCAL_FILES_ONLY,
        device: str = DEFAULT_LOCAL_LLM_DEVICE,
        dtype: str = DEFAULT_LOCAL_LLM_DTYPE,
        seed: int = SEED,
        max_new_tokens: int = DEFAULT_LOCAL_LLM_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_LOCAL_LLM_TEMPERATURE,
        top_p: float = DEFAULT_LOCAL_LLM_TOP_P,
        top_k: int = DEFAULT_LOCAL_LLM_TOP_K,
        repetition_penalty: float = DEFAULT_LOCAL_LLM_REPETITION_PENALTY,
        do_sample: bool = DEFAULT_LOCAL_LLM_DO_SAMPLE,
        allow_cpu: bool = DEFAULT_ALLOW_LOCAL_LLM_CPU,
        load_in_4bit: bool = DEFAULT_LOCAL_LLM_LOAD_IN_4BIT,
        load_in_8bit: bool = DEFAULT_LOCAL_LLM_LOAD_IN_8BIT,
        prompt_version: str = LOCAL_LLM_PROMPT_VERSION,
    ) -> None:
        self.model_id = model_id
        self.model_path = Path(model_path) if model_path is not None else None
        self.local_files_only = local_files_only
        self.requested_device = device
        self.requested_dtype = dtype
        self.allow_cpu = allow_cpu
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.base_seed = int(seed)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.repetition_penalty = float(repetition_penalty)
        self.do_sample = bool(do_sample)
        self.prompt_version = self._validate_prompt_version(prompt_version)
        self.user_prompt_template = self._select_user_prompt_template(self.prompt_version)
        self._last_generation_runtime = GenerationRuntime()

        self._torch = self._import_torch()
        self._transformers = self._import_transformers()
        self.transformers_version = getattr(self._transformers, "__version__", "unknown")
        self.torch_version = getattr(self._torch, "__version__", "unknown")

        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Use at most one of --local-llm-load-in-4bit and --local-llm-load-in-8bit.")

        self.device = self._resolve_device(device, allow_cpu=allow_cpu)
        self.dtype = self._resolve_dtype(dtype)
        self.quantization_config = self._build_quantization_config()
        self.model_loading_method, load_target = self._resolve_load_target()

        logging.info(
            "Loading local LLM paraphraser | target=%s | device=%s | dtype=%s | "
            "local_files_only=%s | load_in_4bit=%s | load_in_8bit=%s",
            load_target,
            self.device,
            self._dtype_to_metadata_value(self.dtype),
            self.local_files_only,
            self.load_in_4bit,
            self.load_in_8bit,
        )

        self.tokenizer = self._transformers.AutoTokenizer.from_pretrained(
            load_target,
            local_files_only=self.local_files_only,
        )
        model_kwargs: dict[str, object] = {
            "local_files_only": self.local_files_only,
        }
        if self.dtype is not None:
            model_kwargs["torch_dtype"] = self.dtype
        if self.quantization_config is not None:
            model_kwargs["quantization_config"] = self.quantization_config
            model_kwargs["device_map"] = self._quantized_device_map()

        self.model = self._transformers.AutoModelForCausalLM.from_pretrained(
            load_target,
            **model_kwargs,
        )
        if self.quantization_config is None:
            self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def _import_torch() -> Any:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "The local_llm paraphraser requires torch. Install the project "
                "dependencies before running --paraphraser local_llm."
            ) from exc
        return torch

    @staticmethod
    def _import_transformers() -> Any:
        try:
            import transformers
        except ImportError as exc:
            raise ImportError(
                "The local_llm paraphraser requires transformers. Install the "
                "project dependencies before running --paraphraser local_llm."
            ) from exc
        return transformers

    def _resolve_device(self, requested_device: str, allow_cpu: bool) -> Any:
        normalized = str(requested_device).strip().lower()
        if normalized == "auto":
            if self._torch.cuda.is_available():
                return self._torch.device("cuda")
            if not allow_cpu:
                raise RuntimeError(
                    "CUDA is not available. Running Qwen 4B on CPU is too slow. "
                    "Install CUDA-enabled PyTorch, use --local-llm-device cuda, "
                    "or explicitly pass --allow-local-llm-cpu for tiny debugging only."
                )
            return self._torch.device("cpu")

        resolved = self._torch.device(requested_device)
        if resolved.type == "cuda" and not self._torch.cuda.is_available():
            raise RuntimeError(
                "Requested CUDA for the local Qwen paraphraser, but PyTorch reports "
                "CUDA is not available. Install CUDA-enabled PyTorch or run "
                "--print-runtime-info to inspect the environment."
            )
        if resolved.type == "cpu" and not allow_cpu:
            raise RuntimeError(
                "Running Qwen 4B on CPU is too slow and is disabled by default. "
                "Use CUDA, or explicitly pass --allow-local-llm-cpu for tiny debugging only."
            )
        return resolved

    def _resolve_dtype(self, requested_dtype: str) -> Any:
        normalized = str(requested_dtype).strip().lower()
        if normalized == "auto":
            if self.device.type == "cuda" and self._cuda_bfloat16_supported():
                return self._torch.bfloat16
            if self.device.type == "cuda":
                return self._torch.float16
            return self._torch.float32

        dtype_lookup = {
            "float32": self._torch.float32,
            "fp32": self._torch.float32,
            "float16": self._torch.float16,
            "fp16": self._torch.float16,
            "bfloat16": self._torch.bfloat16,
            "bf16": self._torch.bfloat16,
        }
        if normalized not in dtype_lookup:
            raise ValueError(
                "--local-llm-dtype must be one of auto, float32, float16, or bfloat16; "
                f"got {requested_dtype!r}."
            )
        return dtype_lookup[normalized]

    def _cuda_bfloat16_supported(self) -> bool:
        if self.device.type != "cuda" or not self._torch.cuda.is_available():
            return False
        checker = getattr(self._torch.cuda, "is_bf16_supported", None)
        if checker is None:
            return False
        try:
            return bool(checker())
        except Exception:
            return False

    def _build_quantization_config(self) -> Any | None:
        if not (self.load_in_4bit or self.load_in_8bit):
            return None
        if self.device.type != "cuda":
            raise RuntimeError("bitsandbytes quantized loading requires CUDA for this script.")
        if importlib.util.find_spec("bitsandbytes") is None:
            raise ImportError(
                "Quantized local LLM loading requires bitsandbytes, but it is not installed. "
                "Install bitsandbytes or rerun without --local-llm-load-in-4bit/--local-llm-load-in-8bit."
            )

        BitsAndBytesConfig = self._transformers.BitsAndBytesConfig
        if self.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True,
            )
        return BitsAndBytesConfig(load_in_8bit=True)

    def _quantized_device_map(self) -> dict[str, int]:
        if self.device.type != "cuda":
            raise RuntimeError("Quantized device_map is only configured for CUDA.")
        device_index = self.device.index
        if device_index is None:
            device_index = self._torch.cuda.current_device()
        return {"": int(device_index)}

    @staticmethod
    def _dtype_to_metadata_value(dtype: Any) -> str:
        return str(dtype).replace("torch.", "")

    def _resolve_load_target(self) -> tuple[str, str]:
        if self.model_path is not None and self.model_path.exists():
            return "local_path", str(self.model_path)

        if self.local_files_only:
            expected_path = self.model_path or DEFAULT_LOCAL_LLM_MODEL_PATH
            raise FileNotFoundError(
                "Local Qwen paraphraser files were not found.\n"
                f"Expected local path: {expected_path}\n"
                f"Fixed model id: {self.model_id}\n\n"
                "Freeze the model into the project folder first:\n"
                "python Scripts_code/freeze_local_llm_model.py\n\n"
                "Then rerun the pilot with:\n"
                "python Scripts_code/05_attack_pilot.py --paraphraser local_llm "
                "--max-examples 2 --max-generation-attempts 3 "
                "--max-classifier-queries 3 --overwrite"
            )

        return "huggingface_model_id_or_cache", self.model_id

    def generate(
        self,
        original_text: str,
        attempt_index: int,
        feedback: AttackFeedback,
        row_id: int,
    ) -> str:
        self._last_generation_runtime = GenerationRuntime()
        attempt_seed = self.compute_attempt_seed(
            base_seed=self.base_seed,
            row_id=row_id,
            feedback_condition=feedback.feedback_condition,
            attempt_index=attempt_index,
        )
        self._transformers.set_seed(attempt_seed)

        messages = self._build_messages(
            original_text=normalize_text(original_text),
            feedback=feedback,
            attempt_index=attempt_index,
        )
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encodings = self.tokenizer(prompt_text, return_tensors="pt")
        encodings = {key: value.to(self.device) for key, value in encodings.items()}
        prompt_token_count = int(encodings["input_ids"].shape[-1])

        start_time = time.perf_counter()
        with self._torch.inference_mode():
            output_ids = self.model.generate(
                **encodings,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        if self.device.type == "cuda":
            self._torch.cuda.synchronize(self.device)
        generation_seconds = time.perf_counter() - start_time

        continuation_ids = output_ids[0][prompt_token_count:]
        generated_token_count = int(continuation_ids.shape[-1])
        tokens_per_second = (
            generated_token_count / generation_seconds
            if generation_seconds > 0
            else None
        )
        self._last_generation_runtime = GenerationRuntime(
            generation_seconds=generation_seconds,
            generated_token_count=generated_token_count,
            generation_tokens_per_second=tokens_per_second,
        )
        logging.info(
            "Local LLM generation timing | row_id=%s | condition=%s | attempt=%s | "
            "seconds=%.3f | generated_tokens=%s | tokens_per_second=%s",
            row_id,
            feedback.feedback_condition,
            attempt_index,
            generation_seconds,
            generated_token_count,
            f"{tokens_per_second:.2f}" if tokens_per_second is not None else None,
        )

        generated_text = self.tokenizer.decode(
            continuation_ids,
            skip_special_tokens=True,
        )
        return self._clean_generated_text(generated_text)

    def get_last_generation_runtime(self) -> GenerationRuntime:
        return self._last_generation_runtime

    def _build_messages(
        self,
        original_text: str,
        feedback: AttackFeedback,
        attempt_index: int,
    ) -> list[dict[str, str]]:
        if feedback.feedback_condition == LABEL_ONLY_CONDITION:
            feedback_block = self.LABEL_ONLY_FEEDBACK_TEMPLATE.format(
                predicted_label_name=feedback.predicted_label_name,
            )
        elif feedback.feedback_condition == SCORE_BASED_CONDITION:
            if feedback.confidence is None:
                raise ValueError("score_based feedback requires a confidence value.")
            feedback_block = self.SCORE_BASED_FEEDBACK_TEMPLATE.format(
                predicted_label_name=feedback.predicted_label_name,
                confidence=float(feedback.confidence),
            )
        else:
            raise ValueError(f"Unknown feedback condition: {feedback.feedback_condition}")

        user_prompt = self.user_prompt_template.format(
            feedback_block=feedback_block,
            original_text=original_text,
            rewrite_strategy_block=LocalLLMParaphraser.rewrite_strategy_for_attempt(
                attempt_index
            ),
        )
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT_TEMPLATE},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _validate_prompt_version(prompt_version: str) -> str:
        normalized = str(prompt_version).strip()
        if normalized not in LOCAL_LLM_PROMPT_VERSIONS:
            raise ValueError(
                f"Unknown local LLM prompt version {prompt_version!r}. "
                f"Choose one of: {list(LOCAL_LLM_PROMPT_VERSIONS)}"
            )
        return normalized

    @staticmethod
    def _select_user_prompt_template(prompt_version: str) -> str:
        if prompt_version == LOCAL_LLM_PROMPT_VERSION_V2_CONSERVATIVE:
            return LocalLLMParaphraser.USER_PROMPT_TEMPLATE_V2_CONSERVATIVE
        if prompt_version == LOCAL_LLM_PROMPT_VERSION_V3_DIVERSE_CONSERVATIVE:
            return LocalLLMParaphraser.USER_PROMPT_TEMPLATE_V3_DIVERSE_CONSERVATIVE
        raise ValueError(f"Unknown local LLM prompt version: {prompt_version}")

    @staticmethod
    def rewrite_strategy_for_attempt(attempt_index: int) -> str:
        strategies = [
            "Make a light syntactic rewrite while preserving all details exactly.",
            "Change sentence openings or connective phrasing without changing the event order.",
            "Reorder clauses only where the meaning and timeline remain unchanged.",
            "Replace non-critical wording with close synonyms while keeping names, places, numbers, dates, and negation unchanged.",
            "Split or merge clauses naturally while preserving every stated detail.",
        ]
        return strategies[(int(attempt_index) - 1) % len(strategies)]

    @staticmethod
    def _clean_generated_text(generated_text: str) -> str:
        cleaned = re.sub(
            r"<\|(?:im_start|im_end|endoftext)\|>",
            "",
            str(generated_text),
        )
        return cleaned.strip()

    @staticmethod
    def compute_attempt_seed(
        base_seed: int,
        row_id: int,
        feedback_condition: str,
        attempt_index: int,
    ) -> int:
        seed_material = f"{base_seed}|{row_id}|{feedback_condition}|{attempt_index}"
        digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
        return int(digest[:16], 16) % (2**31 - 1)

    def get_metadata(self) -> dict[str, object]:
        prompt_material = "\n\n".join(
            [
                self.SYSTEM_PROMPT_TEMPLATE,
                self.user_prompt_template,
                self.LABEL_ONLY_FEEDBACK_TEMPLATE,
                self.SCORE_BASED_FEEDBACK_TEMPLATE,
                "\n".join(
                    self.rewrite_strategy_for_attempt(i) for i in range(1, 6)
                ),
            ]
        )
        prompt_hash = hashlib.sha256(prompt_material.encode("utf-8")).hexdigest()
        return {
            "mode": LOCAL_LLM_PARAPHRASER_MODE,
            "name": type(self).__name__,
            "backend": LOCAL_LLM_TRANSFORMERS_BACKEND,
            "uses_external_api": False,
            "model_id": self.model_id,
            "local_model_path": str(self.model_path) if self.model_path else None,
            "local_files_only": self.local_files_only,
            "device": str(self.device),
            "requested_device": self.requested_device,
            "dtype": self._dtype_to_metadata_value(self.dtype),
            "requested_dtype": self.requested_dtype,
            "allow_cpu": self.allow_cpu,
            "quantization": {
                "enabled": bool(self.load_in_4bit or self.load_in_8bit),
                "load_in_4bit": self.load_in_4bit,
                "load_in_8bit": self.load_in_8bit,
                "bnb_4bit_quant_type": "nf4" if self.load_in_4bit else None,
                "bnb_4bit_use_double_quant": True if self.load_in_4bit else None,
                "bnb_4bit_compute_dtype": self._dtype_to_metadata_value(self.dtype)
                if self.load_in_4bit
                else None,
            },
            "generation_seed": self.base_seed,
            "attempt_seed_logic": (
                "sha256(base_seed|row_id|feedback_condition|attempt_index) "
                "modulo 2^31-1; transformers.set_seed is called before each generation"
            ),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "prompt_version": self.prompt_version,
            "prompt_template_sha256": prompt_hash,
            "prompt_template": {
                "system": self.SYSTEM_PROMPT_TEMPLATE,
                "user": self.user_prompt_template,
                "label_only_feedback_block": self.LABEL_ONLY_FEEDBACK_TEMPLATE,
                "score_based_feedback_block": self.SCORE_BASED_FEEDBACK_TEMPLATE,
                "rewrite_strategies": [
                    self.rewrite_strategy_for_attempt(i) for i in range(1, 6)
                ],
            },
            "transformers_version": self.transformers_version,
            "torch_version": self.torch_version,
            "model_loading_method": self.model_loading_method,
            "description": (
                "Fixed Qwen local paraphraser used identically in label-only and "
                "score-based conditions except for the feedback fields exposed in "
                "the prompt."
            ),
        }


class LlamaCppServerLocalLLMParaphraser:
    """
    Local llama.cpp server backend for the fixed Qwen paraphraser.

    The experiment script still builds prompts and controls feedback exposure.
    The server is only a local generation backend, so the comparison between
    feedback conditions remains unchanged.
    """

    SYSTEM_PROMPT_TEMPLATE = LocalLLMParaphraser.SYSTEM_PROMPT_TEMPLATE
    USER_PROMPT_TEMPLATE = LocalLLMParaphraser.USER_PROMPT_TEMPLATE
    USER_PROMPT_TEMPLATE_V2_CONSERVATIVE = (
        LocalLLMParaphraser.USER_PROMPT_TEMPLATE_V2_CONSERVATIVE
    )
    USER_PROMPT_TEMPLATE_V3_DIVERSE_CONSERVATIVE = (
        LocalLLMParaphraser.USER_PROMPT_TEMPLATE_V3_DIVERSE_CONSERVATIVE
    )
    LABEL_ONLY_FEEDBACK_TEMPLATE = LocalLLMParaphraser.LABEL_ONLY_FEEDBACK_TEMPLATE
    SCORE_BASED_FEEDBACK_TEMPLATE = LocalLLMParaphraser.SCORE_BASED_FEEDBACK_TEMPLATE

    def __init__(
        self,
        model_id: str = DEFAULT_LOCAL_LLM_MODEL_ID,
        server_url: str = DEFAULT_LOCAL_LLM_SERVER_URL,
        server_model: str = DEFAULT_LOCAL_LLM_SERVER_MODEL,
        gguf_model_path: Path | None = None,
        seed: int = SEED,
        max_new_tokens: int = DEFAULT_LOCAL_LLM_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_LOCAL_LLM_TEMPERATURE,
        top_p: float = DEFAULT_LOCAL_LLM_TOP_P,
        top_k: int = DEFAULT_LOCAL_LLM_TOP_K,
        repetition_penalty: float = DEFAULT_LOCAL_LLM_REPETITION_PENALTY,
        do_sample: bool = DEFAULT_LOCAL_LLM_DO_SAMPLE,
        context_size: int = DEFAULT_LOCAL_LLM_CONTEXT_SIZE,
        gpu_layers: str = DEFAULT_LOCAL_LLM_GPU_LAYERS,
        allow_nonlocal_server: bool = DEFAULT_ALLOW_NONLOCAL_LLM_SERVER,
        request_timeout_seconds: float = DEFAULT_LOCAL_LLM_REQUEST_TIMEOUT_SECONDS,
        prompt_version: str = LOCAL_LLM_PROMPT_VERSION,
    ) -> None:
        self.model_id = model_id
        self.backend = LOCAL_LLM_LLAMA_CPP_SERVER_BACKEND
        self.server_url = self._normalize_server_url(server_url)
        self.server_root_url = self._server_root_from_v1_url(self.server_url)
        self.server_model = str(server_model)
        self.gguf_model_path = Path(gguf_model_path) if gguf_model_path else None
        self.base_seed = int(seed)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.repetition_penalty = float(repetition_penalty)
        self.do_sample = bool(do_sample)
        self.context_size = int(context_size)
        self.gpu_layers = str(gpu_layers)
        self.allow_nonlocal_server = bool(allow_nonlocal_server)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.prompt_version = LocalLLMParaphraser._validate_prompt_version(
            prompt_version
        )
        self.user_prompt_template = LocalLLMParaphraser._select_user_prompt_template(
            self.prompt_version
        )
        self._last_generation_runtime = GenerationRuntime()

        self._validate_local_server_url()
        if self.gguf_model_path is None:
            logging.warning(
                "No --local-llm-gguf-model-path was provided. The run can still "
                "test speed, but final thesis metadata should include a GGUF path "
                "and file hash."
            )
        elif not self.gguf_model_path.exists():
            logging.warning(
                "Configured GGUF model path does not exist: %s",
                self.gguf_model_path,
            )

        self.server_models_response = self._http_get_json(
            self._join_url(self.server_url, "models"),
            required=True,
        )
        self.server_props_response = self._http_get_json(
            self._join_url(self.server_root_url, "props"),
            required=False,
        )

        logging.info(
            "Using llama.cpp server local LLM backend | url=%s | model=%s | "
            "gguf=%s | ctx=%s | gpu_layers=%s",
            self.server_url,
            self.server_model,
            self.gguf_model_path,
            self.context_size,
            self.gpu_layers,
        )

    @staticmethod
    def _normalize_server_url(server_url: str) -> str:
        normalized = str(server_url).strip().rstrip("/")
        if not normalized:
            raise ValueError("--local-llm-server-url must not be empty.")
        return normalized

    @staticmethod
    def _server_root_from_v1_url(server_url: str) -> str:
        if server_url.endswith("/v1"):
            return server_url[:-3].rstrip("/")
        return server_url

    @staticmethod
    def _join_url(base_url: str, path: str) -> str:
        return f"{base_url.rstrip('/')}/{path.lstrip('/')}"

    def _validate_local_server_url(self) -> None:
        parsed = urlparse(self.server_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(
                "--local-llm-server-url must be an http(s) URL for the local "
                "llama.cpp server."
            )

        local_hosts = {"127.0.0.1", "localhost", "::1"}
        host = (parsed.hostname or "").lower()
        if host not in local_hosts and not self.allow_nonlocal_server:
            raise ValueError(
                "Refusing to use a non-local LLM server URL. The BEP backend test "
                "should be local-only; pass --allow-nonlocal-llm-server only for "
                "explicit engineering diagnostics."
            )

    def _http_get_json(self, url: str, required: bool) -> dict[str, object] | None:
        request = Request(url, method="GET")
        try:
            with urlopen(request, timeout=self.request_timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError) as exc:
            if required:
                raise RuntimeError(
                    "Could not reach the local llama.cpp OpenAI-compatible server "
                    f"at {url}. Start llama-server first, then rerun the command."
                ) from exc
            logging.warning("Optional llama.cpp metadata endpoint unavailable: %s", url)
            return None

        if not raw.strip():
            return {}
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            if required:
                raise RuntimeError(f"Server returned non-JSON response from {url}.") from exc
            logging.warning("Optional llama.cpp metadata endpoint returned non-JSON: %s", url)
            return None
        return decoded if isinstance(decoded, dict) else {"response": decoded}

    def _http_post_json(self, url: str, payload: dict[str, object]) -> dict[str, object]:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.request_timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                "llama.cpp server generation request failed with HTTP "
                f"{exc.code}: {error_body[:1000]}"
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(
                "llama.cpp server generation request failed. Confirm that "
                "llama-server is still running locally."
            ) from exc

        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("llama.cpp server returned non-JSON generation output.") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError("llama.cpp server returned an unexpected JSON shape.")
        return decoded

    def generate(
        self,
        original_text: str,
        attempt_index: int,
        feedback: AttackFeedback,
        row_id: int,
    ) -> str:
        self._last_generation_runtime = GenerationRuntime()
        attempt_seed = LocalLLMParaphraser.compute_attempt_seed(
            base_seed=self.base_seed,
            row_id=row_id,
            feedback_condition=feedback.feedback_condition,
            attempt_index=attempt_index,
        )
        messages = self._build_messages(
            original_text=normalize_text(original_text),
            feedback=feedback,
            attempt_index=attempt_index,
        )
        payload: dict[str, object] = {
            "model": self.server_model,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature if self.do_sample else 0.0,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repetition_penalty,
            "seed": attempt_seed,
            "stream": False,
        }

        start_time = time.perf_counter()
        response = self._http_post_json(
            self._join_url(self.server_url, "chat/completions"),
            payload,
        )
        generation_seconds = time.perf_counter() - start_time

        generated_text = self._extract_generated_text(response)
        generated_token_count = self._extract_generated_token_count(response)
        tokens_per_second = (
            generated_token_count / generation_seconds
            if generated_token_count is not None and generation_seconds > 0
            else None
        )
        self._last_generation_runtime = GenerationRuntime(
            generation_seconds=generation_seconds,
            generated_token_count=generated_token_count,
            generation_tokens_per_second=tokens_per_second,
        )
        logging.info(
            "llama.cpp generation timing | row_id=%s | condition=%s | attempt=%s | "
            "seconds=%.3f | generated_tokens=%s | tokens_per_second=%s",
            row_id,
            feedback.feedback_condition,
            attempt_index,
            generation_seconds,
            generated_token_count,
            f"{tokens_per_second:.2f}" if tokens_per_second is not None else None,
        )
        return LocalLLMParaphraser._clean_generated_text(generated_text)

    @staticmethod
    def _extract_generated_text(response: dict[str, object]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("llama.cpp response did not include choices.")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError("llama.cpp response choice had an unexpected shape.")
        message = first_choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if content is not None:
                return str(content)
        text = first_choice.get("text")
        if text is not None:
            return str(text)
        raise RuntimeError("llama.cpp response did not include generated content.")

    @staticmethod
    def _extract_generated_token_count(response: dict[str, object]) -> int | None:
        usage = response.get("usage")
        if isinstance(usage, dict):
            completion_tokens = usage.get("completion_tokens")
            if completion_tokens is not None:
                return int(completion_tokens)

        timings = response.get("timings")
        if isinstance(timings, dict):
            predicted_n = timings.get("predicted_n")
            if predicted_n is not None:
                return int(predicted_n)

        return None

    def get_last_generation_runtime(self) -> GenerationRuntime:
        return self._last_generation_runtime

    def _build_messages(
        self,
        original_text: str,
        feedback: AttackFeedback,
        attempt_index: int,
    ) -> list[dict[str, str]]:
        return LocalLLMParaphraser._build_messages(
            self,
            original_text=original_text,
            feedback=feedback,
            attempt_index=attempt_index,
        )

    def _gguf_metadata(self) -> dict[str, object]:
        if self.gguf_model_path is None:
            return {
                "path": None,
                "file_name": None,
                "exists": False,
                "sha256": None,
                "quantization_type": None,
            }

        exists = self.gguf_model_path.exists()
        return {
            "path": str(self.gguf_model_path),
            "file_name": self.gguf_model_path.name,
            "exists": bool(exists),
            "sha256": compute_file_hash(self.gguf_model_path) if exists else None,
            "quantization_type": self._infer_quantization_type(self.gguf_model_path.name),
        }

    @staticmethod
    def _infer_quantization_type(file_name: str) -> str | None:
        upper_name = file_name.upper()
        for quant in ["Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q4_0"]:
            if quant in upper_name:
                return quant
        return None

    def get_metadata(self) -> dict[str, object]:
        prompt_material = "\n\n".join(
            [
                self.SYSTEM_PROMPT_TEMPLATE,
                self.user_prompt_template,
                self.LABEL_ONLY_FEEDBACK_TEMPLATE,
                self.SCORE_BASED_FEEDBACK_TEMPLATE,
                "\n".join(
                    LocalLLMParaphraser.rewrite_strategy_for_attempt(i)
                    for i in range(1, 6)
                ),
            ]
        )
        prompt_hash = hashlib.sha256(prompt_material.encode("utf-8")).hexdigest()
        server_version = None
        if isinstance(self.server_props_response, dict):
            server_version = (
                self.server_props_response.get("version")
                or self.server_props_response.get("build_number")
                or self.server_props_response.get("build_commit")
            )

        return {
            "mode": LOCAL_LLM_PARAPHRASER_MODE,
            "name": type(self).__name__,
            "backend": self.backend,
            "uses_external_api": False,
            "local_only_request": True,
            "model_id": self.model_id,
            "server_url": self.server_url,
            "server_root_url": self.server_root_url,
            "server_model": self.server_model,
            "backend_version": server_version,
            "llama_cpp_server_metadata": self.server_props_response,
            "llama_cpp_models_metadata": self.server_models_response,
            "gguf_model": self._gguf_metadata(),
            "context_size": self.context_size,
            "gpu_layers": self.gpu_layers,
            "generation_seed": self.base_seed,
            "attempt_seed_logic": (
                "sha256(base_seed|row_id|feedback_condition|attempt_index) "
                "modulo 2^31-1; seed is sent with each llama.cpp server request; "
                "exact determinism depends on server/build settings"
            ),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "llama_cpp_repeat_penalty_parameter": "repeat_penalty",
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "request_timeout_seconds": self.request_timeout_seconds,
            "prompt_version": self.prompt_version,
            "prompt_template_sha256": prompt_hash,
            "prompt_template": {
                "system": self.SYSTEM_PROMPT_TEMPLATE,
                "user": self.user_prompt_template,
                "label_only_feedback_block": self.LABEL_ONLY_FEEDBACK_TEMPLATE,
                "score_based_feedback_block": self.SCORE_BASED_FEEDBACK_TEMPLATE,
                "rewrite_strategies": [
                    LocalLLMParaphraser.rewrite_strategy_for_attempt(i)
                    for i in range(1, 6)
                ],
            },
            "description": (
                "Fixed Qwen local paraphraser using a local llama.cpp "
                "OpenAI-compatible server. Prompt construction, feedback exposure, "
                "budgets, validity checks, and success definition are unchanged "
                "from the Transformers backend."
            ),
        }


def build_paraphraser(args: argparse.Namespace) -> Paraphraser:
    mode = args.paraphraser
    if mode == DEFAULT_PARAPHRASER_MODE:
        return PlaceholderParaphraser()
    if mode == INVALID_DEBUG_PARAPHRASER_MODE:
        return InvalidDebugParaphraser()
    if mode == LOCAL_LLM_PARAPHRASER_MODE:
        if args.local_llm_backend == LOCAL_LLM_LLAMA_CPP_SERVER_BACKEND:
            return LlamaCppServerLocalLLMParaphraser(
                model_id=args.local_llm_model_id,
                server_url=args.local_llm_server_url,
                server_model=args.local_llm_server_model,
                gguf_model_path=Path(args.local_llm_gguf_model_path)
                if args.local_llm_gguf_model_path
                else None,
                seed=args.seed,
                max_new_tokens=args.local_llm_max_new_tokens,
                temperature=args.local_llm_temperature,
                top_p=args.local_llm_top_p,
                top_k=args.local_llm_top_k,
                repetition_penalty=args.local_llm_repetition_penalty,
                do_sample=args.local_llm_do_sample,
                context_size=args.local_llm_context_size,
                gpu_layers=args.local_llm_gpu_layers,
                allow_nonlocal_server=args.allow_nonlocal_llm_server,
                request_timeout_seconds=args.local_llm_request_timeout_seconds,
                prompt_version=args.local_llm_prompt_version,
            )
        return LocalLLMParaphraser(
            model_id=args.local_llm_model_id,
            model_path=Path(args.local_llm_model_path)
            if args.local_llm_model_path
            else None,
            local_files_only=args.local_llm_local_files_only,
            device=args.local_llm_device,
            dtype=args.local_llm_dtype,
            seed=args.seed,
            max_new_tokens=args.local_llm_max_new_tokens,
            temperature=args.local_llm_temperature,
            top_p=args.local_llm_top_p,
            top_k=args.local_llm_top_k,
            repetition_penalty=args.local_llm_repetition_penalty,
            do_sample=args.local_llm_do_sample,
            allow_cpu=args.allow_local_llm_cpu,
            load_in_4bit=args.local_llm_load_in_4bit,
            load_in_8bit=args.local_llm_load_in_8bit,
            prompt_version=args.local_llm_prompt_version,
        )
    raise ValueError(f"Unknown paraphraser mode: {mode}")


# =========================
# CLI / logging
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the first pilot paraphrase attack loop."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing pilot attack outputs.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--paraphraser",
        default=DEFAULT_PARAPHRASER_MODE,
        choices=[
            DEFAULT_PARAPHRASER_MODE,
            INVALID_DEBUG_PARAPHRASER_MODE,
            LOCAL_LLM_PARAPHRASER_MODE,
        ],
        help=(
            "Paraphraser mode. local_llm uses the fixed local Qwen paraphraser."
        ),
    )
    parser.add_argument(
        "--print-runtime-info",
        action="store_true",
        help="Print local LLM runtime/CUDA diagnostics and exit before loading models.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Global run seed. local_llm derives deterministic attempt seeds from it.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional row limit for tiny smoke tests. Uses the first N sorted pilot rows.",
    )
    parser.add_argument(
        "--max-classifier-queries",
        type=int,
        default=MAX_CLASSIFIER_QUERIES,
        help="Maximum valid candidate classifier queries per row and feedback condition.",
    )
    parser.add_argument(
        "--max-generation-attempts",
        type=int,
        default=MAX_GENERATION_ATTEMPTS,
        help="Maximum paraphrase generation attempts per row and feedback condition.",
    )
    parser.add_argument(
        "--pool",
        default=PILOT_POOL_NAME,
        choices=[PILOT_POOL_NAME, MAIN_POOL_NAME, EXPLORATORY_POOL_NAME],
        help=(
            "Candidate pool to attack. pilot preserves the previous train-dev "
            "pilot behavior; main uses the frozen 160-row final-test pool; "
            "exploratory uses the held-out final-test remainder."
        ),
    )
    parser.add_argument(
        "--output-tag",
        default=None,
        help=(
            "Optional safe suffix for output files, e.g. gen6_query3. "
            "When omitted, preserves the original filenames."
        ),
    )
    parser.add_argument(
        "--local-llm-model-id",
        default=DEFAULT_LOCAL_LLM_MODEL_ID,
        help="Fixed Hugging Face model id for engineering/freeze metadata.",
    )
    parser.add_argument(
        "--local-llm-model-path",
        default=str(DEFAULT_LOCAL_LLM_MODEL_PATH),
        help="Local frozen Qwen model path used by --paraphraser local_llm.",
    )
    parser.add_argument(
        "--local-llm-backend",
        default=DEFAULT_LOCAL_LLM_BACKEND,
        choices=[
            LOCAL_LLM_TRANSFORMERS_BACKEND,
            LOCAL_LLM_LLAMA_CPP_SERVER_BACKEND,
        ],
        help=(
            "Backend for --paraphraser local_llm. transformers loads the frozen "
            "HF model in this process; llama_cpp_server calls a local llama.cpp "
            "OpenAI-compatible server."
        ),
    )
    parser.add_argument(
        "--local-llm-server-url",
        default=DEFAULT_LOCAL_LLM_SERVER_URL,
        help=(
            "OpenAI-compatible base URL for the local llama.cpp server, used only "
            "with --local-llm-backend llama_cpp_server."
        ),
    )
    parser.add_argument(
        "--local-llm-server-model",
        default=DEFAULT_LOCAL_LLM_SERVER_MODEL,
        help=(
            "Model name sent to the llama.cpp /v1/chat/completions endpoint. "
            "For reproducibility, keep this aligned with the GGUF file."
        ),
    )
    parser.add_argument(
        "--local-llm-gguf-model-path",
        default=DEFAULT_LOCAL_LLM_GGUF_MODEL_PATH,
        help=(
            "Local GGUF model file path for llama_cpp_server metadata hashing. "
            "The server must be started separately with the same file."
        ),
    )
    parser.add_argument(
        "--local-llm-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_LOCAL_LLM_LOCAL_FILES_ONLY,
        help=(
            "Load the local LLM with local_files_only=True. Use "
            "--no-local-llm-local-files-only only for engineering tests."
        ),
    )
    parser.add_argument(
        "--local-llm-device",
        default=DEFAULT_LOCAL_LLM_DEVICE,
        help="Device for local LLM generation: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--local-llm-dtype",
        default=DEFAULT_LOCAL_LLM_DTYPE,
        choices=["auto", "float32", "fp32", "float16", "fp16", "bfloat16", "bf16"],
        help="Torch dtype for local LLM loading.",
    )
    parser.add_argument(
        "--local-llm-max-new-tokens",
        type=int,
        default=DEFAULT_LOCAL_LLM_MAX_NEW_TOKENS,
        help="Maximum generated tokens for each local LLM paraphrase.",
    )
    parser.add_argument(
        "--local-llm-temperature",
        type=float,
        default=DEFAULT_LOCAL_LLM_TEMPERATURE,
        help="Sampling temperature for local LLM paraphrases.",
    )
    parser.add_argument(
        "--local-llm-top-p",
        type=float,
        default=DEFAULT_LOCAL_LLM_TOP_P,
        help="Nucleus sampling top_p for local LLM paraphrases.",
    )
    parser.add_argument(
        "--local-llm-top-k",
        type=int,
        default=DEFAULT_LOCAL_LLM_TOP_K,
        help="Top-k sampling value for local LLM paraphrases.",
    )
    parser.add_argument(
        "--local-llm-repetition-penalty",
        type=float,
        default=DEFAULT_LOCAL_LLM_REPETITION_PENALTY,
        help="Repetition penalty for local LLM paraphrases.",
    )
    parser.add_argument(
        "--local-llm-do-sample",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_LOCAL_LLM_DO_SAMPLE,
        help="Whether local LLM generation uses sampling.",
    )
    parser.add_argument(
        "--allow-local-llm-cpu",
        action="store_true",
        default=DEFAULT_ALLOW_LOCAL_LLM_CPU,
        help="Explicitly allow slow CPU Qwen runs for tiny debugging only.",
    )
    parser.add_argument(
        "--local-llm-load-in-4bit",
        action="store_true",
        default=DEFAULT_LOCAL_LLM_LOAD_IN_4BIT,
        help="Load local Qwen with bitsandbytes 4-bit NF4 quantization.",
    )
    parser.add_argument(
        "--local-llm-load-in-8bit",
        action="store_true",
        default=DEFAULT_LOCAL_LLM_LOAD_IN_8BIT,
        help="Load local Qwen with bitsandbytes 8-bit quantization.",
    )
    parser.add_argument(
        "--local-llm-context-size",
        type=int,
        default=DEFAULT_LOCAL_LLM_CONTEXT_SIZE,
        help=(
            "Context size recorded for llama_cpp_server runs. Start llama-server "
            "with the same --ctx-size value."
        ),
    )
    parser.add_argument(
        "--local-llm-gpu-layers",
        default=DEFAULT_LOCAL_LLM_GPU_LAYERS,
        help=(
            "GPU offload setting recorded for llama_cpp_server runs, for example "
            "auto or 999. Start llama-server with the same -ngl/--n-gpu-layers value."
        ),
    )
    parser.add_argument(
        "--allow-nonlocal-llm-server",
        action="store_true",
        default=DEFAULT_ALLOW_NONLOCAL_LLM_SERVER,
        help=(
            "Allow a non-local llama.cpp server URL. This should remain disabled "
            "for BEP runs."
        ),
    )
    parser.add_argument(
        "--local-llm-request-timeout-seconds",
        type=float,
        default=DEFAULT_LOCAL_LLM_REQUEST_TIMEOUT_SECONDS,
        help="HTTP timeout for each llama_cpp_server generation request.",
    )
    parser.add_argument(
        "--local-llm-prompt-version",
        default=LOCAL_LLM_PROMPT_VERSION,
        choices=list(LOCAL_LLM_PROMPT_VERSIONS),
        help=(
            "Prompt template version for local_llm. The v2 default preserves "
            "previous behavior; v3 adds conservative non-identical rewrite "
            "instructions and attempt-specific strategies."
        ),
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("--max-examples must be positive when provided.")
    if args.max_classifier_queries <= 0:
        raise ValueError("--max-classifier-queries must be positive.")
    if args.max_generation_attempts <= 0:
        raise ValueError("--max-generation-attempts must be positive.")
    if args.local_llm_max_new_tokens <= 0:
        raise ValueError("--local-llm-max-new-tokens must be positive.")
    if args.local_llm_temperature <= 0:
        raise ValueError("--local-llm-temperature must be positive.")
    if not 0 < args.local_llm_top_p <= 1:
        raise ValueError("--local-llm-top-p must be in (0, 1].")
    if args.local_llm_top_k <= 0:
        raise ValueError("--local-llm-top-k must be positive.")
    if args.local_llm_repetition_penalty <= 0:
        raise ValueError("--local-llm-repetition-penalty must be positive.")
    if args.local_llm_load_in_4bit and args.local_llm_load_in_8bit:
        raise ValueError("Use only one of --local-llm-load-in-4bit or --local-llm-load-in-8bit.")
    if args.local_llm_context_size <= 0:
        raise ValueError("--local-llm-context-size must be positive.")
    if args.local_llm_request_timeout_seconds <= 0:
        raise ValueError("--local-llm-request-timeout-seconds must be positive.")
    if args.output_tag is not None:
        normalized_tag = str(args.output_tag).strip()
        if not normalized_tag:
            raise ValueError("--output-tag must not be empty when provided.")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", normalized_tag):
            raise ValueError(
                "--output-tag may contain only letters, numbers, underscore, dash, and dot."
            )
        args.output_tag = normalized_tag


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _runtime_dtype_to_string(dtype: Any) -> str:
    if dtype is None:
        return "none"
    return str(dtype).replace("torch.", "")


def _cuda_bfloat16_supported(torch_module: Any) -> bool:
    if not torch_module.cuda.is_available():
        return False
    checker = getattr(torch_module.cuda, "is_bf16_supported", None)
    if checker is None:
        return False
    try:
        return bool(checker())
    except Exception:
        return False


def _resolve_runtime_info_device(
    torch_module: Any,
    requested_device: str,
    allow_cpu: bool,
) -> tuple[str, str | None]:
    normalized = str(requested_device).strip().lower()
    if normalized == "auto":
        if torch_module.cuda.is_available():
            return "cuda", None
        if allow_cpu:
            return "cpu", None
        return (
            "blocked_cpu",
            "CUDA is not available and --allow-local-llm-cpu was not set.",
        )

    resolved = torch_module.device(requested_device)
    if resolved.type == "cuda" and not torch_module.cuda.is_available():
        return str(resolved), "Requested CUDA, but PyTorch reports CUDA is not available."
    if resolved.type == "cpu" and not allow_cpu:
        return str(resolved), "CPU Qwen runs are disabled unless --allow-local-llm-cpu is set."
    return str(resolved), None


def _resolve_runtime_info_dtype(
    torch_module: Any,
    selected_device: str,
    requested_dtype: str,
) -> str:
    normalized = str(requested_dtype).strip().lower()
    device_type = selected_device.split(":", maxsplit=1)[0]
    if normalized == "auto":
        if device_type == "cuda" and _cuda_bfloat16_supported(torch_module):
            return "bfloat16"
        if device_type == "cuda":
            return "float16"
        return "float32"
    dtype_lookup = {
        "float32": "float32",
        "fp32": "float32",
        "float16": "float16",
        "fp16": "float16",
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
    }
    return dtype_lookup.get(normalized, normalized)


def collect_runtime_info(args: argparse.Namespace) -> dict[str, object]:
    try:
        import torch
    except ImportError:
        torch = None

    try:
        import transformers
    except ImportError:
        transformers = None

    cuda_available = bool(torch is not None and torch.cuda.is_available())
    selected_device = "unknown"
    selected_dtype = "unknown"
    device_warning = None
    gpu_name = None
    gpu_memory_total_gb = None
    gpu_memory_free_gb = None

    if torch is not None:
        selected_device, device_warning = _resolve_runtime_info_device(
            torch,
            args.local_llm_device,
            args.allow_local_llm_cpu,
        )
        selected_dtype = _resolve_runtime_info_dtype(
            torch,
            selected_device,
            args.local_llm_dtype,
        )
        if cuda_available:
            device_index = torch.device(selected_device).index
            if device_index is None:
                device_index = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(device_index)
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
                gpu_memory_total_gb = round(total_bytes / (1024**3), 2)
                gpu_memory_free_gb = round(free_bytes / (1024**3), 2)
            except Exception:
                props = torch.cuda.get_device_properties(device_index)
                gpu_memory_total_gb = round(props.total_memory / (1024**3), 2)

    return {
        "torch_version": getattr(torch, "__version__", None) if torch else None,
        "transformers_version": getattr(transformers, "__version__", None)
        if transformers
        else None,
        "cuda_available": cuda_available,
        "torch_cuda_version": getattr(torch.version, "cuda", None) if torch else None,
        "gpu_name": gpu_name,
        "gpu_memory_total_gb": gpu_memory_total_gb,
        "gpu_memory_free_gb": gpu_memory_free_gb,
        "local_llm_backend": args.local_llm_backend,
        "requested_local_llm_device": args.local_llm_device,
        "selected_local_llm_device": selected_device,
        "device_warning": device_warning,
        "requested_dtype": args.local_llm_dtype,
        "selected_dtype": selected_dtype,
        "quantization_enabled": bool(
            args.local_llm_load_in_4bit or args.local_llm_load_in_8bit
        ),
        "local_llm_load_in_4bit": args.local_llm_load_in_4bit,
        "local_llm_load_in_8bit": args.local_llm_load_in_8bit,
        "bitsandbytes_installed": importlib.util.find_spec("bitsandbytes") is not None,
        "local_model_path": args.local_llm_model_path,
        "local_model_path_exists": bool(
            args.local_llm_model_path and Path(args.local_llm_model_path).exists()
        ),
        "local_llm_server_url": args.local_llm_server_url,
        "local_llm_server_model": args.local_llm_server_model,
        "local_llm_gguf_model_path": args.local_llm_gguf_model_path,
        "local_llm_gguf_model_path_exists": bool(
            args.local_llm_gguf_model_path
            and Path(args.local_llm_gguf_model_path).exists()
        ),
        "local_llm_prompt_version": args.local_llm_prompt_version,
        "local_files_only": args.local_llm_local_files_only,
        "allow_local_llm_cpu": args.allow_local_llm_cpu,
    }


def print_runtime_info(args: argparse.Namespace) -> None:
    runtime_info = collect_runtime_info(args)
    formatted = json.dumps(runtime_info, indent=2)
    logging.info("Runtime diagnostics:\n%s", formatted)
    print(formatted)


# =========================
# Validation / IO helpers
# =========================
def normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def canonicalize_for_candidate_identity(text: str) -> str:
    normalized = normalize_text(text).lower()
    return normalized


def compute_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_output_paths(
    experiment_split: str,
    paraphraser_mode: str,
    output_tag: str | None = None,
) -> OutputPaths:
    suffix = f"_{output_tag}" if output_tag else ""
    return OutputPaths(
        attempts=OUTPUT_DIR
        / f"attack_attempts_{experiment_split}_{paraphraser_mode}{suffix}.csv",
        results=OUTPUT_DIR
        / f"attack_results_{experiment_split}_{paraphraser_mode}{suffix}.csv",
        metadata=OUTPUT_DIR
        / f"attack_{experiment_split}_meta_{paraphraser_mode}{suffix}.json",
    )


def check_output_paths(output_paths: OutputPaths, overwrite: bool) -> None:
    existing = [
        path
        for path in [
            output_paths.attempts,
            output_paths.results,
            output_paths.metadata,
        ]
        if path.exists()
    ]
    if existing and not overwrite:
        formatted = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(
            "Pilot attack outputs already exist. Refusing to overwrite them.\n"
            f"{formatted}\n"
            "Rerun with --overwrite only when you intentionally want to replace them."
        )


def pool_config(experiment_split: str) -> dict[str, object]:
    if experiment_split not in POOL_CONFIGS:
        raise ValueError(
            f"Unknown pool {experiment_split!r}; choose one of {sorted(POOL_CONFIGS)}."
        )
    return POOL_CONFIGS[experiment_split]


def validate_attack_pool(df: pd.DataFrame, experiment_split: str, pool_path: Path) -> None:
    config = pool_config(experiment_split)
    expected_source_split = str(config["expected_source_split"])
    required_columns = {
        SOURCE_SPLIT_COLUMN,
        SOURCE_TEXT_COLUMN,
        ROW_ID_COLUMN,
        ATTACK_TEXT_COLUMN,
        GOLD_LABEL_NAME_COLUMN,
        CORRECT_COLUMN,
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"{pool_path.name} is missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(f"{pool_path.name} is empty.")

    splits = set(df[SOURCE_SPLIT_COLUMN].astype(str).str.strip().unique())
    if splits != {expected_source_split}:
        raise ValueError(
            f"{experiment_split} attack must use only {expected_source_split!r}; "
            f"found {sorted(splits)}."
        )

    text_columns = set(df[SOURCE_TEXT_COLUMN].astype(str).str.strip().unique())
    if text_columns != {"text_truncated"}:
        raise ValueError(
            f"{experiment_split} pool must use text_truncated as the attacked text column; "
            f"found {sorted(text_columns)}."
        )

    if not df[CORRECT_COLUMN].isin([1, True]).all():
        bad = int((~df[CORRECT_COLUMN].isin([1, True])).sum())
        raise ValueError(
            f"{experiment_split} pool contains {bad} rows that were not originally correct."
        )

    empty_texts = int(
        df[ATTACK_TEXT_COLUMN].fillna("").astype(str).str.strip().eq("").sum()
    )
    if empty_texts:
        raise ValueError(f"{pool_path.name} contains {empty_texts} empty attack texts.")


def load_attack_pool(experiment_split: str) -> tuple[pd.DataFrame, Path]:
    config = pool_config(experiment_split)
    pool_path = Path(config["path"])
    if not pool_path.exists():
        raise FileNotFoundError(
            f"Missing {experiment_split} candidate pool: {pool_path}. "
            "Run Scripts_code/02_make_attack_pool.py first."
        )

    df = pd.read_csv(pool_path)
    validate_attack_pool(df, experiment_split=experiment_split, pool_path=pool_path)
    return df.sort_values(ROW_ID_COLUMN).reset_index(drop=True), pool_path


def failed_checks_to_json(failed_checks: list[str]) -> str:
    return json.dumps(failed_checks, ensure_ascii=True)


def get_original_prediction(classifier: Any, text: str) -> Any:
    result = classifier.predict_label_and_confidence(text)
    classifier.reset_query_counts()
    return result


def query_candidate(
    classifier: Any,
    text: str,
    feedback_condition: str,
) -> CandidatePrediction:
    if feedback_condition == LABEL_ONLY_CONDITION:
        result = classifier.predict_label(text)
        return CandidatePrediction(
            predicted_label_name=result.predicted_label_name,
            confidence=None,
        )

    if feedback_condition == SCORE_BASED_CONDITION:
        result = classifier.predict_label_and_confidence(text)
        return CandidatePrediction(
            predicted_label_name=result.predicted_label_name,
            confidence=result.confidence,
        )

    raise ValueError(f"Unknown feedback condition: {feedback_condition}")


# =========================
# Attack loop
# =========================
def run_single_attack(
    row: pd.Series,
    feedback_condition: str,
    run_id: str,
    classifier: Any,
    validity_checker: Any,
    paraphraser: Paraphraser,
    max_classifier_queries: int,
    max_generation_attempts: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    row_id = int(row[ROW_ID_COLUMN])
    gold_label_name = str(row[GOLD_LABEL_NAME_COLUMN])
    source_split = str(row[SOURCE_SPLIT_COLUMN])
    source_text_column = str(row[SOURCE_TEXT_COLUMN])
    original_text = normalize_text(str(row[ATTACK_TEXT_COLUMN]))

    logging.info(
        "Verifying original prediction | row_id=%s | condition=%s",
        row_id,
        feedback_condition,
    )
    original_prediction = get_original_prediction(classifier, original_text)
    original_pred_label_name = original_prediction.predicted_label_name
    original_confidence = float(original_prediction.confidence)

    if original_pred_label_name != gold_label_name:
        logging.warning(
            "Skipping row because verified original prediction is not correct | "
            "row_id=%s | gold=%s | original_pred=%s",
            row_id,
            gold_label_name,
            original_pred_label_name,
        )
        return [], {
            "run_id": run_id,
            "row_id": row_id,
            "feedback_condition": feedback_condition,
            "gold_label_name": gold_label_name,
            "source_split": source_split,
            "source_text_column": source_text_column,
            "original_pred_label_name": original_pred_label_name,
            "original_confidence": original_confidence,
            "success": False,
            "queries_used": 0,
            "generation_attempts_used": 0,
            "valid_candidates": 0,
            "invalid_candidates": 0,
            "successful_candidate_text": None,
            "successful_similarity": None,
            "final_pred_label_name": original_pred_label_name,
            "stop_reason": "original_prediction_mismatch",
        }

    classifier.reset_query_counts()

    feedback = AttackFeedback(
        feedback_condition=feedback_condition,
        predicted_label_name=original_pred_label_name,
        confidence=(
            original_confidence if feedback_condition == SCORE_BASED_CONDITION else None
        ),
    )

    attempt_rows: list[dict[str, object]] = []
    queries_used = 0
    generation_attempts_used = 0
    success = False
    successful_candidate_text: str | None = None
    successful_similarity: float | None = None
    final_pred_label_name = original_pred_label_name
    stop_reason = "not_started"
    valid_candidates = 0
    invalid_candidates = 0
    original_identity_key = canonicalize_for_candidate_identity(original_text)
    seen_non_original_candidate_keys: set[str] = set()

    for attempt_index in range(1, max_generation_attempts + 1):
        if queries_used >= max_classifier_queries:
            stop_reason = "budget_exhausted"
            break

        generation_attempts_used = attempt_index
        candidate_text = paraphraser.generate(
            original_text=original_text,
            attempt_index=attempt_index,
            feedback=feedback,
            row_id=row_id,
        )
        generation_runtime = paraphraser.get_last_generation_runtime()
        candidate_identity_key = canonicalize_for_candidate_identity(candidate_text)
        is_duplicate_candidate = (
            candidate_identity_key != original_identity_key
            and candidate_identity_key in seen_non_original_candidate_keys
        )
        if is_duplicate_candidate:
            is_valid = False
            failed_checks = ["duplicate_candidate"]
            semantic_similarity = None
        else:
            if candidate_identity_key != original_identity_key:
                seen_non_original_candidate_keys.add(candidate_identity_key)
            validity = validity_checker.evaluate(original_text, candidate_text)
            is_valid = bool(validity.is_valid)
            failed_checks = list(validity.failed_checks)
            semantic_similarity = validity.semantic_similarity

        if is_valid:
            valid_candidates += 1
        else:
            invalid_candidates += 1

        logging.info(
            "Attempt generated | row_id=%s | condition=%s | attempt=%s | valid=%s | "
            "queries_used=%s | generation_seconds=%s | generated_tokens=%s",
            row_id,
            feedback_condition,
            attempt_index,
            is_valid,
            queries_used,
            (
                f"{generation_runtime.generation_seconds:.3f}"
                if generation_runtime.generation_seconds is not None
                else None
            ),
            generation_runtime.generated_token_count,
        )
        if is_duplicate_candidate:
            logging.info(
                "Duplicate candidate rejected before classifier query | row_id=%s | "
                "condition=%s | attempt=%s",
                row_id,
                feedback_condition,
                attempt_index,
            )

        queried_classifier = False
        classifier_query_index: int | None = None
        candidate_pred_label_name: str | None = None
        candidate_confidence: float | None = None
        is_success = False
        attempt_stop_reason = "invalid_candidate"

        if is_valid:
            queries_used += 1
            classifier_query_index = queries_used
            queried_classifier = True

            candidate_prediction = query_candidate(
                classifier=classifier,
                text=candidate_text,
                feedback_condition=feedback_condition,
            )
            candidate_pred_label_name = candidate_prediction.predicted_label_name
            candidate_confidence = candidate_prediction.confidence
            final_pred_label_name = candidate_pred_label_name

            is_success = candidate_pred_label_name != original_pred_label_name
            feedback = AttackFeedback(
                feedback_condition=feedback_condition,
                predicted_label_name=candidate_pred_label_name,
                confidence=candidate_confidence,
            )

            if is_success:
                success = True
                successful_candidate_text = candidate_text
                successful_similarity = (
                    float(semantic_similarity)
                    if semantic_similarity is not None
                    else None
                )
                stop_reason = "success"
                attempt_stop_reason = "success"
                logging.info(
                    "Successful flip | row_id=%s | condition=%s | attempt=%s | "
                    "query=%s | original=%s | candidate=%s",
                    row_id,
                    feedback_condition,
                    attempt_index,
                    queries_used,
                    original_pred_label_name,
                    candidate_pred_label_name,
                )
            elif queries_used >= max_classifier_queries:
                stop_reason = "budget_exhausted"
                attempt_stop_reason = "budget_exhausted"
            else:
                attempt_stop_reason = "continue"

        if (
            not is_valid
            and attempt_index >= max_generation_attempts
            and not success
        ):
            stop_reason = "generation_attempts_exhausted"
            attempt_stop_reason = "generation_attempts_exhausted"

        attempt_rows.append(
            {
                "run_id": run_id,
                "row_id": row_id,
                "feedback_condition": feedback_condition,
                "gold_label_name": gold_label_name,
                "original_pred_label_name": original_pred_label_name,
                "original_confidence": original_confidence,
                "attempt_index": attempt_index,
                "classifier_query_index": classifier_query_index,
                "original_attack_text": original_text,
                "candidate_text": candidate_text,
                "is_valid": is_valid,
                "failed_checks": failed_checks_to_json(failed_checks),
                "semantic_similarity": semantic_similarity,
                "queried_classifier": queried_classifier,
                "candidate_pred_label_name": candidate_pred_label_name,
                "candidate_confidence": candidate_confidence,
                "is_success": is_success,
                "stop_reason": attempt_stop_reason,
                "generation_seconds": generation_runtime.generation_seconds,
                "generated_token_count": generation_runtime.generated_token_count,
                "generation_tokens_per_second": (
                    generation_runtime.generation_tokens_per_second
                ),
            }
        )

        if success or stop_reason in {
            "budget_exhausted",
            "generation_attempts_exhausted",
        }:
            break

    if stop_reason == "not_started":
        stop_reason = (
            "budget_exhausted"
            if queries_used >= max_classifier_queries
            else "generation_attempts_exhausted"
        )

    result_row = {
        "run_id": run_id,
        "row_id": row_id,
        "feedback_condition": feedback_condition,
        "gold_label_name": gold_label_name,
        "source_split": source_split,
        "source_text_column": source_text_column,
        "original_pred_label_name": original_pred_label_name,
        "original_confidence": original_confidence,
        "success": success,
        "queries_used": queries_used,
        "generation_attempts_used": generation_attempts_used,
        "valid_candidates": valid_candidates,
        "invalid_candidates": invalid_candidates,
        "successful_candidate_text": successful_candidate_text,
        "successful_similarity": successful_similarity,
        "final_pred_label_name": final_pred_label_name,
        "stop_reason": stop_reason,
    }

    logging.info(
        "Attack finished | row_id=%s | condition=%s | success=%s | queries=%s | "
        "generations=%s | stop=%s",
        row_id,
        feedback_condition,
        success,
        queries_used,
        generation_attempts_used,
        stop_reason,
    )

    return attempt_rows, result_row


def run_pilot_attack(
    pilot_df: pd.DataFrame,
    run_id: str,
    classifier: Any,
    validity_checker: Any,
    paraphraser: Paraphraser,
    max_classifier_queries: int,
    max_generation_attempts: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_attempt_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []

    for _, row in pilot_df.iterrows():
        for feedback_condition in FEEDBACK_CONDITIONS:
            attempt_rows, result_row = run_single_attack(
                row=row,
                feedback_condition=feedback_condition,
                run_id=run_id,
                classifier=classifier,
                validity_checker=validity_checker,
                paraphraser=paraphraser,
                max_classifier_queries=max_classifier_queries,
                max_generation_attempts=max_generation_attempts,
            )
            all_attempt_rows.extend(attempt_rows)
            result_rows.append(result_row)

    attempts_df = pd.DataFrame(all_attempt_rows, columns=ATTEMPT_COLUMNS)
    results_df = pd.DataFrame(result_rows, columns=RESULT_COLUMNS)
    return attempts_df, results_df


def as_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    normalized = series.fillna(False).astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes"})


def run_post_run_integrity_checks(
    attempts_path: Path,
    results_path: Path,
    pool_df: pd.DataFrame,
    experiment_split: str,
    max_classifier_queries: int,
    require_full_pool_size: bool,
) -> None:
    attempts_df = pd.read_csv(attempts_path)
    results_df = pd.read_csv(results_path)
    config = pool_config(experiment_split)
    expected_full_pool_size = int(config["expected_full_pool_size"])

    allowed_stop_reasons = {
        "success",
        "budget_exhausted",
        "generation_attempts_exhausted",
        "original_prediction_mismatch",
    }
    errors: list[str] = []

    expected_result_rows = len(pool_df) * len(FEEDBACK_CONDITIONS)
    if require_full_pool_size and len(pool_df) != expected_full_pool_size:
        errors.append(
            f"{experiment_split} pool has {len(pool_df)} rows, "
            f"expected {expected_full_pool_size}."
        )
    if len(results_df) != expected_result_rows:
        errors.append(
            f"Results file has {len(results_df)} rows, expected {expected_result_rows}."
        )

    pool_row_ids = set(pool_df[ROW_ID_COLUMN].astype(int).tolist())
    for row_id in sorted(pool_row_ids):
        row_results = results_df[results_df["row_id"].astype(int) == row_id]
        condition_counts = row_results["feedback_condition"].value_counts().to_dict()
        for condition in FEEDBACK_CONDITIONS:
            if int(condition_counts.get(condition, 0)) != 1:
                errors.append(
                    f"row_id={row_id} has {condition_counts.get(condition, 0)} "
                    f"{condition} result rows, expected 1."
                )

    unexpected_result_rows = set(results_df["row_id"].astype(int)) - pool_row_ids
    if unexpected_result_rows:
        errors.append(
            f"Results include row_ids outside the {experiment_split} pool: "
            f"{sorted(unexpected_result_rows)}."
        )

    original_mismatches = results_df[
        results_df["stop_reason"].astype(str) == "original_prediction_mismatch"
    ]
    if not original_mismatches.empty:
        errors.append(
            "Verified original predictions were not correct for rows: "
            f"{original_mismatches[['row_id', 'feedback_condition', 'gold_label_name', 'original_pred_label_name']].to_dict('records')}"
        )

    query_counts = pd.to_numeric(results_df["queries_used"], errors="coerce")
    over_budget = results_df[query_counts > max_classifier_queries]
    if not over_budget.empty:
        errors.append(
            "queries_used exceeds budget for result rows: "
            f"{over_budget[['row_id', 'feedback_condition']].to_dict('records')}"
        )

    stop_reasons = set(results_df["stop_reason"].dropna().astype(str))
    unexpected_stop_reasons = stop_reasons - allowed_stop_reasons
    if unexpected_stop_reasons:
        errors.append(
            f"Unexpected result stop_reason values: {sorted(unexpected_stop_reasons)}."
        )

    if not attempts_df.empty:
        is_valid = as_bool_series(attempts_df["is_valid"])
        queried_classifier = as_bool_series(attempts_df["queried_classifier"])
        is_success = as_bool_series(attempts_df["is_success"])
        classifier_query_index = pd.to_numeric(
            attempts_df["classifier_query_index"],
            errors="coerce",
        )

        invalid_queried = attempts_df[~is_valid & queried_classifier]
        if not invalid_queried.empty:
            errors.append(
                "Invalid candidates were queried for rows: "
                f"{invalid_queried[['row_id', 'feedback_condition', 'attempt_index']].to_dict('records')}"
            )

        duplicate_candidate = attempts_df["failed_checks"].fillna("").astype(str).str.contains(
            "duplicate_candidate",
            regex=False,
        )
        duplicate_queried = attempts_df[duplicate_candidate & queried_classifier]
        if not duplicate_queried.empty:
            errors.append(
                "Duplicate candidates were queried for rows: "
                f"{duplicate_queried[['row_id', 'feedback_condition', 'attempt_index']].to_dict('records')}"
            )

        has_query_index = classifier_query_index.notna()
        query_index_without_query = attempts_df[has_query_index & ~queried_classifier]
        if not query_index_without_query.empty:
            errors.append(
                "classifier_query_index is present when queried_classifier is false: "
                f"{query_index_without_query[['row_id', 'feedback_condition', 'attempt_index']].to_dict('records')}"
            )

        query_without_query_index = attempts_df[queried_classifier & ~has_query_index]
        if not query_without_query_index.empty:
            errors.append(
                "queried_classifier is true but classifier_query_index is missing: "
                f"{query_without_query_index[['row_id', 'feedback_condition', 'attempt_index']].to_dict('records')}"
            )

        for _, result_row in results_df.iterrows():
            row_id = int(result_row["row_id"])
            condition = str(result_row["feedback_condition"])
            mask = (
                (attempts_df["row_id"].astype(int) == row_id)
                & (attempts_df["feedback_condition"].astype(str) == condition)
            )

            matching_attempts = attempts_df[mask]
            matching_success = bool(is_success[mask].any()) if not matching_attempts.empty else False
            result_success = bool(as_bool_series(pd.Series([result_row["success"]])).iloc[0])
            if result_success != matching_success:
                errors.append(
                    f"Result success mismatch for row_id={row_id}, condition={condition}: "
                    f"result={result_success}, attempts={matching_success}."
                )

            matching_query_index = classifier_query_index[mask]
            if matching_query_index.notna().any():
                max_query_index = int(matching_query_index.max())
            else:
                max_query_index = 0

            result_queries_used = int(result_row["queries_used"])
            if result_queries_used != max_query_index:
                errors.append(
                    f"queries_used mismatch for row_id={row_id}, condition={condition}: "
                    f"result={result_queries_used}, max attempt query index={max_query_index}."
                )

    if errors:
        formatted = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Post-run integrity checks failed:\n{formatted}")

    logging.info("Post-run integrity checks passed.")


def build_metadata(
    run_id: str,
    pool_df: pd.DataFrame,
    experiment_split: str,
    pool_path: Path,
    classifier: Any,
    validity_checker: Any,
    attempts_df: pd.DataFrame,
    results_df: pd.DataFrame,
    paraphraser: Paraphraser,
    paraphraser_mode: str,
    output_paths: OutputPaths,
    integrity_checks_passed: bool,
    seed: int,
    max_classifier_queries: int,
    max_generation_attempts: int,
    max_examples: int | None,
    output_tag: str | None,
) -> dict[str, object]:
    if not attempts_df.empty and "generation_seconds" in attempts_df.columns:
        generation_seconds = pd.to_numeric(
            attempts_df["generation_seconds"],
            errors="coerce",
        ).dropna()
        generated_tokens = pd.to_numeric(
            attempts_df["generated_token_count"],
            errors="coerce",
        ).dropna()
        tokens_per_second = pd.to_numeric(
            attempts_df["generation_tokens_per_second"],
            errors="coerce",
        ).dropna()
    else:
        generation_seconds = pd.Series(dtype="float64")
        generated_tokens = pd.Series(dtype="float64")
        tokens_per_second = pd.Series(dtype="float64")

    paraphraser_metadata = paraphraser.get_metadata() | {
        "mode": paraphraser_mode,
        "name": type(paraphraser).__name__,
    }

    return {
        "run_id": run_id,
        "run_name": RUN_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "experiment_split": experiment_split,
        "output_tag": output_tag,
        "input_pool": {
            "path": str(pool_path),
            "sha256": compute_file_hash(pool_path),
            "num_rows": int(len(pool_df)),
            "row_limit": max_examples,
            "source_splits": sorted(
                set(pool_df[SOURCE_SPLIT_COLUMN].astype(str).str.strip())
            ),
            "source_text_columns": sorted(
                set(pool_df[SOURCE_TEXT_COLUMN].astype(str).str.strip())
            ),
        },
        "feedback_conditions": list(FEEDBACK_CONDITIONS),
        "budgets": {
            "max_classifier_queries": max_classifier_queries,
            "max_generation_attempts": max_generation_attempts,
            "budget_config": (
                f"gen{max_generation_attempts}_query{max_classifier_queries}"
            ),
            "invalid_paraphrases_count_as_classifier_queries": False,
            "original_prediction_verification_counts_against_attack_budget": False,
            "duplicate_candidates_count_as_classifier_queries": False,
        },
        "budget_config": f"gen{max_generation_attempts}_query{max_classifier_queries}",
        "success_definition": (
            "valid paraphrase + classifier prediction flip from the verified "
            "original prediction + within classifier-query budget"
        ),
        "prompt_version": paraphraser_metadata.get("prompt_version"),
        "prompt_template_sha256": paraphraser_metadata.get("prompt_template_sha256"),
        "paraphraser": paraphraser_metadata,
        "integrity_checks_passed": integrity_checks_passed,
        "classifier": classifier.get_metadata(),
        "validity_checker": validity_checker.get_metadata(),
        "outputs": {
            "attempts_path": str(output_paths.attempts),
            "attempt_rows": int(len(attempts_df)),
            "results_path": str(output_paths.results),
            "result_rows": int(len(results_df)),
            "metadata_path": str(output_paths.metadata),
        },
        "summary": {
            "successes_by_condition": {
                str(k): int(v)
                for k, v in results_df.groupby("feedback_condition")["success"]
                .sum()
                .to_dict()
                .items()
            },
            "stop_reasons": {
                str(k): int(v)
                for k, v in results_df["stop_reason"].value_counts().to_dict().items()
            },
            "valid_attempts": int(attempts_df["is_valid"].sum())
            if not attempts_df.empty
            else 0,
            "classifier_queries": int(attempts_df["queried_classifier"].sum())
            if not attempts_df.empty
            else 0,
            "generation_runtime": {
                "timed_generation_attempts": int(generation_seconds.count()),
                "total_generation_seconds": float(generation_seconds.sum())
                if not generation_seconds.empty
                else None,
                "average_generation_seconds": float(generation_seconds.mean())
                if not generation_seconds.empty
                else None,
                "total_generated_tokens": int(generated_tokens.sum())
                if not generated_tokens.empty
                else None,
                "average_tokens_per_second": float(tokens_per_second.mean())
                if not tokens_per_second.empty
                else None,
            },
        },
    }


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if args.print_runtime_info:
        print_runtime_info(args)
        return

    output_paths = build_output_paths(args.pool, args.paraphraser, args.output_tag)
    check_output_paths(output_paths=output_paths, overwrite=args.overwrite)

    run_id = datetime.now(timezone.utc).strftime(f"{args.pool}_%Y%m%dT%H%M%SZ")
    logging.info("Starting %s attack run | run_id=%s", args.pool, run_id)
    paraphraser = build_paraphraser(args)
    logging.info("Using paraphraser mode: %s", args.paraphraser)

    pool_df, pool_path = load_attack_pool(args.pool)
    if args.max_examples is not None:
        pool_df = pool_df.head(args.max_examples).reset_index(drop=True)
        logging.info(
            "Applying %s row limit | max_examples=%s",
            args.pool,
            args.max_examples,
        )
    logging.info(
        "Loaded %s pool | path=%s | rows=%s",
        args.pool,
        pool_path,
        len(pool_df),
    )

    logging.info("Loading classifier wrapper.")
    classifier = DeceptionClassifierWrapper()

    logging.info("Loading validity checker.")
    validity_checker = ParaphraseValidityChecker()

    attempts_df, results_df = run_pilot_attack(
        pilot_df=pool_df,
        run_id=run_id,
        classifier=classifier,
        validity_checker=validity_checker,
        paraphraser=paraphraser,
        max_classifier_queries=args.max_classifier_queries,
        max_generation_attempts=args.max_generation_attempts,
    )

    attempts_df.to_csv(output_paths.attempts, index=False)
    results_df.to_csv(output_paths.results, index=False)

    run_post_run_integrity_checks(
        attempts_path=output_paths.attempts,
        results_path=output_paths.results,
        pool_df=pool_df,
        experiment_split=args.pool,
        max_classifier_queries=args.max_classifier_queries,
        require_full_pool_size=args.max_examples is None,
    )

    metadata = build_metadata(
        run_id=run_id,
        pool_df=pool_df,
        experiment_split=args.pool,
        pool_path=pool_path,
        classifier=classifier,
        validity_checker=validity_checker,
        attempts_df=attempts_df,
        results_df=results_df,
        paraphraser=paraphraser,
        paraphraser_mode=args.paraphraser,
        output_paths=output_paths,
        integrity_checks_passed=True,
        seed=args.seed,
        max_classifier_queries=args.max_classifier_queries,
        max_generation_attempts=args.max_generation_attempts,
        max_examples=args.max_examples,
        output_tag=args.output_tag,
    )
    with open(output_paths.metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logging.info("%s attack complete.", args.pool.capitalize())
    logging.info("Attempts saved to: %s", output_paths.attempts)
    logging.info("Results saved to:  %s", output_paths.results)
    logging.info("Metadata saved to: %s", output_paths.metadata)
    logging.info(
        "Success counts by condition: %s",
        metadata["summary"]["successes_by_condition"],
    )


if __name__ == "__main__":
    main()
