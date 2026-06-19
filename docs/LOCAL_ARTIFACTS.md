# Local Artifacts

This repository intentionally contains code, documentation, and aggregate
results only. Data, model weights, raw outputs, generated text, and credentials
remain local because they may be large, licensed, private, or unsuitable for
normal Git history.

## Expected local paths

The current scripts expect this directory in the repository root:

```text
Automated Deception Classifier (Projectfolder)/
|-- hippocorpus_training_truncated.csv
|-- hippocorpus_test_truncated.csv
|-- DistilBERT/
`-- local_models/
    |-- all-MiniLM-L6-v2/
    |-- Qwen3-4B-Instruct-2507/          # historical local workflow
    `-- Qwen3-4B-Instruct-2507-GGUF/     # historical local workflow
```

The final experiment uses the held-out Hippocorpus pool derived from the
truncated test file, the fixed local DistilBERT classifier, and the local SBERT
model used by the validity gate. Do not change these paths unless the code and
documentation are updated together.

Generated files are written beneath:

```text
Scripts_code/outputs/
```

Only `Scripts_code/outputs/.gitkeep` is tracked.

## What is not committed

- Hippocorpus CSVs, including source text and labels.
- The fixed DistilBERT model folder and weight files.
- The frozen SBERT model folder and weight files.
- Llama or other local model weights, including GGUF and safetensors files.
- DeepInfra-hosted Llama weights; the API-served model is referenced by name
  only and no provider weights are stored locally.
- Raw attack attempts/results, generated paraphrases, metadata, diagnostics,
  and derived analysis files.
- PDFs, DOCX files, ZIP archives, thesis source/build files, working notes, and
  spreadsheets.
- API keys, `.env` files, credentials, passwords, and tokens.

## Model helpers

If the local SBERT folder is missing:

```powershell
python Scripts_code\freeze_sbert_model.py
```

`freeze_local_llm_model.py` supports the earlier local Qwen workflow retained
for provenance. It is not required for the final DeepInfra-served 70B run.

## Credential handling

Set the provider key only in the process environment:

```powershell
$env:DEEPINFRA_API_KEY = "your-key-here"
```

The attack runner receives only the environment-variable name through
`--api-llm-key-env`. Never store the value in tracked files or raw outputs.
