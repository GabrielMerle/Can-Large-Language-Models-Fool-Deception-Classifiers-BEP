# Project Status

This document summarizes the current state of the repository at the first clean GitHub push.

---

## Current Goal

Evaluate whether semantic-preserving LLM paraphrases can fool a fixed DistilBERT deception classifier.

The main comparison is between:

- label-only feedback;
- score-based feedback.

The research question is about feedback granularity, not about building a new classifier or comparing many LLMs.

---

## Implemented Scripts

### `01_baseline_inference.py`

Runs the victim classifier on the available split data and creates baseline predictions, summaries, and correctly-classified candidate pools.

### `02_make_attack_pool.py`

Freezes the pilot and main attack pools from the correctly-classified candidate pools.

### `03_classifier_wrapper.py`

Provides the controlled black-box interface for the fixed victim classifier.

Public feedback modes:

- label-only;
- label plus confidence.

### `04_validity_checks.py`

Checks whether a paraphrase is valid enough to count as an eligible adversarial candidate.

It uses local SBERT similarity and lightweight checks for meaning drift.

### `05_attack_pilot.py`

Runs the pilot attack loop on the frozen pilot pool.

Supported paraphraser modes:

- `placeholder`
- `invalid_debug`
- `local_llm`

Supported local LLM backends:

- `transformers`
- `llama_cpp_server`

### `freeze_sbert_model.py`

Downloads and saves the local SBERT model used by the validity checker.

### `freeze_local_llm_model.py`

Downloads and saves the local Qwen Transformers model used by the local LLM paraphraser path.

---

## Implemented Outputs

Generated output files are stored in:

```text
Scripts_code/outputs/
```

They are ignored by Git.

The only committed file in that folder should be:

```text
Scripts_code/outputs/.gitkeep
```

---

## Still To Do

Before final thesis results:

- freeze final local LLM settings;
- run full pilot under final settings;
- decide whether the pilot protocol is stable;
- implement or adapt a main experiment script;
- run the final paired main comparison;
- implement final analysis script;
- report ASR, query efficiency, and budget-level success;
- inspect representative successes/failures;
- write results and discussion chapters.

---

## Important Caution

Current local LLM pilot outputs should not automatically be treated as final thesis results. They are useful for validating the pipeline and tuning the protocol. Final results should come only after the protocol is frozen.
