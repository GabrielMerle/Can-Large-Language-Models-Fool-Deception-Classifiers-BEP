# LLM Paraphrase Attacks on a Deception Classifier

This repository contains the implementation for a Bachelor End Project on testing whether meaning-preserving LLM paraphrases can fool a fixed text-based deception classifier.

The project is a controlled black-box robustness study. It does **not** train a new deception classifier. Instead, it evaluates a fixed pretrained DistilBERT deception classifier under two feedback settings:

1. **Label-only feedback**: the attacker receives only the predicted class.
2. **Score-based feedback**: the attacker receives the predicted class plus the predicted-class confidence.

The intended experimental comparison keeps the victim classifier, attacked examples, paraphrase-generation setup, validity checker, budgets, success definition, and logging fixed. The main variable is the amount of classifier feedback exposed to the attack loop.

---

## Repository Status

Implemented:

- baseline inference on train/dev and final test splits;
- construction of candidate pools from examples originally classified correctly;
- frozen pilot and main attack pools;
- a controlled victim-model wrapper with two feedback modes;
- a shared semantic-preservation validity checker;
- local SBERT freezing helper;
- local Qwen paraphraser freezing helper;
- pilot attack loop with placeholder, invalid-debug, Transformers local LLM, and llama.cpp server local LLM modes;
- attempt-level, result-level, and metadata logging.

Not yet implemented:

- final main-experiment script;
- final analysis/statistical comparison script;
- final results tables and thesis-ready figures.

---

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── docs/
│   ├── LOCAL_ARTIFACTS.md
│   ├── RUNNING.md
│   └── REPRODUCIBILITY.md
└── Scripts_code/
    ├── 01_baseline_inference.py
    ├── 02_make_attack_pool.py
    ├── 03_classifier_wrapper.py
    ├── 04_validity_checks.py
    ├── 05_attack_pilot.py
    ├── freeze_sbert_model.py
    ├── freeze_local_llm_model.py
    └── outputs/
        └── .gitkeep
```

Large local files are intentionally excluded from Git. This includes dataset CSVs, model weights, local LLM folders, generated outputs, research-paper PDFs, and thesis drafts. See [`docs/LOCAL_ARTIFACTS.md`](docs/LOCAL_ARTIFACTS.md).

---

## Local Artifact Layout

The scripts expect this local folder structure next to `Scripts_code/`:

```text
Automated Deception Classifier (Projectfolder)/
├── hippocorpus_training_truncated.csv
├── hippocorpus_test_truncated.csv
├── DistilBERT/
│   ├── config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.txt
└── local_models/
    ├── all-MiniLM-L6-v2/
    ├── Qwen3-4B-Instruct-2507/
    └── Qwen3-4B-Instruct-2507-GGUF/
```

These files are required locally but should not be committed.

---

## Installation

Create and activate a virtual environment from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU-based Transformers inference, install a CUDA-enabled PyTorch build that matches the machine. The exact PyTorch command can depend on the local CUDA setup, so verify with:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## Standard Pipeline

Run commands from the repository root.

### 1. Run baseline inference

```powershell
python Scripts_code\01_baseline_inference.py
```

This creates split-specific baseline predictions, summaries, and candidate pools in `Scripts_code/outputs/`.

### 2. Freeze attack pools

```powershell
python Scripts_code\02_make_attack_pool.py
```

This creates:

```text
Scripts_code/outputs/candidate_pool_pilot_20.csv
Scripts_code/outputs/candidate_pool_main_160.csv
Scripts_code/outputs/candidate_pool_freeze_meta.json
```

Use `--overwrite` only when intentionally regenerating frozen pools.

### 3. Run a small pilot smoke test

```powershell
python Scripts_code\05_attack_pilot.py `
  --paraphraser placeholder `
  --max-examples 2 `
  --max-generation-attempts 3 `
  --max-classifier-queries 3 `
  --overwrite
```

This tests the attack loop without using a real LLM paraphraser.

### 4. Run invalid-candidate debug mode

```powershell
python Scripts_code\05_attack_pilot.py `
  --paraphraser invalid_debug `
  --max-examples 2 `
  --max-generation-attempts 3 `
  --max-classifier-queries 3 `
  --overwrite
```

This verifies that invalid candidates are logged but not queried against the classifier.

### 5. Run local LLM pilot mode

For the real local LLM paraphraser, use `--paraphraser local_llm`. The current preferred backend is the local llama.cpp server backend, because it keeps generation local while avoiding the slow Transformers path.

See [`docs/RUNNING.md`](docs/RUNNING.md) for full local LLM and llama.cpp instructions.

---

## Success Definition

An attack is counted as successful only when all of the following hold:

1. the original example was correctly classified by the victim classifier;
2. the generated paraphrase passes the validity checker;
3. the victim classifier prediction flips;
4. the flip occurs within the fixed classifier-query budget.

The original-text verification query is logged separately and does not count against the attack budget. Invalid paraphrases are logged but are not sent to the classifier.

---

## Main Outputs

Generated files are written to `Scripts_code/outputs/` and are ignored by Git.

Typical output groups:

```text
baseline_predictions_train_dev.csv
candidate_pool_train_dev.csv
baseline_summary_train_dev.json

baseline_predictions_test_final.csv
candidate_pool_test_final.csv
baseline_summary_test_final.json

candidate_pool_pilot_20.csv
candidate_pool_main_160.csv
candidate_pool_freeze_meta.json

attack_attempts_pilot_<mode>.csv
attack_results_pilot_<mode>.csv
attack_pilot_meta_<mode>.json
```

---

## Reproducibility Notes

The implementation uses:

- stable local paths relative to the repository root;
- explicit train/dev and test/final split names;
- fixed seeds where sampling or generation control is used;
- SHA-256 hashes for important file/model/prompt metadata where available;
- overwrite protection for generated artifacts;
- one shared validity checker across both feedback conditions;
- one shared victim-model inference path across both feedback conditions.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for more detail.

---

## Privacy and Repository Policy

This repository should contain source code and documentation only. Do not commit:

- dataset CSVs;
- model weights;
- Qwen/SBERT/GGUF local model files;
- generated experiment outputs;
- server logs;
- thesis drafts;
- professor feedback forms;
- research-paper PDFs.

The recommended GitHub setting is **private** until data/model licensing and supervisor expectations are fully clear.
