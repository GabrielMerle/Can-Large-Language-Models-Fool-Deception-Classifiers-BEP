# Local Artifacts

This repository intentionally excludes local datasets, model weights, generated outputs, research papers, and thesis drafts.

The code expects several files to exist locally. They are not included in Git because they may be large, private, licensed, copyrighted, or reproducible from scripts.

---

## Required Local Folder Structure

Place this folder in the repository root:

```text
Automated Deception Classifier (Projectfolder)/
├── hippocorpus_training_truncated.csv
├── hippocorpus_test_truncated.csv
├── hippocorpus_train_set.csv                    # optional historical/source file
├── hippocorpus_test_set.csv                     # optional historical/source file
├── DistilBERT/
│   ├── config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.txt
└── local_models/
    ├── all-MiniLM-L6-v2/
    │   ├── config.json
    │   ├── model.safetensors
    │   ├── tokenizer.json
    │   └── tokenizer_config.json
    ├── Qwen3-4B-Instruct-2507/
    │   ├── config.json
    │   ├── model-00001-of-00003.safetensors
    │   ├── model-00002-of-00003.safetensors
    │   ├── model-00003-of-00003.safetensors
    │   ├── model.safetensors.index.json
    │   ├── tokenizer.json
    │   ├── tokenizer_config.json
    │   ├── vocab.json
    │   └── merges.txt
    └── Qwen3-4B-Instruct-2507-GGUF/
        └── Qwen3-4B-Instruct-2507-Q4_K_M.gguf
```

Only the truncated Hippocorpus files are used by the current pipeline:

```text
hippocorpus_training_truncated.csv
hippocorpus_test_truncated.csv
```

The canonical attacked/model-input column is:

```text
text_truncated
```

The pipeline stores this internally as:

```text
attack_text
```

---

## Why These Files Are Not Committed

### Dataset CSVs

Dataset files may have licensing, privacy, or redistribution restrictions. They should stay local unless redistribution is explicitly allowed.

### DistilBERT model folder

The victim classifier folder contains model weights. These files are large and should not be stored in normal Git history.

### Local SBERT model folder

The validity checker uses a frozen local SBERT model for semantic similarity. This makes normal runs reproducible and avoids unexpected online downloads, but the model files should still remain local.

### Qwen Transformers folder

The local Qwen model is multi-GB and should not be committed.

### Qwen GGUF folder

The GGUF file is a large local inference artifact for llama.cpp. It should not be committed.

### Generated outputs

Files in `Scripts_code/outputs/` are generated artifacts. They should be reproducible from the scripts and local artifacts.

### Papers and thesis drafts

Research-paper PDFs, thesis drafts, proposal documents, and feedback forms are not source code and may contain copyrighted or private material.

---

## Recreating Local Models

### Freeze SBERT

Run this once when the local SBERT folder is missing:

```powershell
python Scripts_code\freeze_sbert_model.py
```

This creates:

```text
Automated Deception Classifier (Projectfolder)/local_models/all-MiniLM-L6-v2/
```

### Freeze Qwen Transformers model

Run this once when the local Qwen Transformers folder is missing:

```powershell
python Scripts_code\freeze_local_llm_model.py
```

This creates:

```text
Automated Deception Classifier (Projectfolder)/local_models/Qwen3-4B-Instruct-2507/
```

This command requires internet access and enough disk space.

### GGUF model

The GGUF file must be downloaded or placed manually:

```text
Automated Deception Classifier (Projectfolder)/local_models/Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
```

The repo stores only the expected path and run documentation, not the GGUF file itself.

---

## Generated Outputs

The following are created by the scripts and ignored by Git:

```text
Scripts_code/outputs/baseline_predictions_train_dev.csv
Scripts_code/outputs/candidate_pool_train_dev.csv
Scripts_code/outputs/baseline_summary_train_dev.json
Scripts_code/outputs/baseline_predictions_test_final.csv
Scripts_code/outputs/candidate_pool_test_final.csv
Scripts_code/outputs/baseline_summary_test_final.json
Scripts_code/outputs/candidate_pool_pilot_20.csv
Scripts_code/outputs/candidate_pool_main_160.csv
Scripts_code/outputs/candidate_pool_freeze_meta.json
Scripts_code/outputs/attack_attempts_pilot_<mode>.csv
Scripts_code/outputs/attack_results_pilot_<mode>.csv
Scripts_code/outputs/attack_pilot_meta_<mode>.json
```

The only file committed inside `Scripts_code/outputs/` should be:

```text
Scripts_code/outputs/.gitkeep
```
