# Running the Project

Run commands from the repository root. Raw data, model folders, and experiment
outputs are local-only; confirm the paths in
[LOCAL_ARTIFACTS.md](LOCAL_ARTIFACTS.md) before starting.

## Environment setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional import check:

```powershell
python -c "import matplotlib, numpy, pandas, torch, transformers, tqdm; print('imports ok')"
```

## Pipeline

### 1. Baseline inference

```powershell
python Scripts_code\01_baseline_inference.py
```

This evaluates the fixed DistilBERT victim and writes local prediction and
candidate-pool files under `Scripts_code/outputs/`.

### 2. Pool construction

```powershell
python Scripts_code\02_make_attack_pool.py
python Scripts_code\02_make_exploratory_remaining_pool.py
```

The final unified attack uses the complete 389-row correctly classified
held-out pool (`candidate_pool_test_final.csv`). Pool files contain source text
and are never committed.

### 3. Final attack runner

`05_attack_pilot.py` is the main attack runner despite its historical name.
Inspect all options before any run:

```powershell
python Scripts_code\05_attack_pilot.py --help
```

The completed final experiment used the following configuration:

```powershell
$env:DEEPINFRA_API_KEY = "your-key-here"

python Scripts_code\05_attack_pilot.py `
  --paraphraser api_llm `
  --pool unified `
  --output-tag llama70b_gen5_query5 `
  --api-llm-base-url "https://api.deepinfra.com/v1/openai" `
  --api-llm-model "meta-llama/Llama-3.3-70B-Instruct-Turbo" `
  --api-llm-key-env DEEPINFRA_API_KEY `
  --api-llm-temperature 0.45 `
  --api-llm-top-p 0.85 `
  --api-llm-max-tokens 220 `
  --max-generation-attempts 5 `
  --max-classifier-queries 5 `
  --num-workers 4 `
  --seed 42
```

The runner refuses to overwrite existing outputs unless `--overwrite` is
provided. Use `--resume` only for a deliberately resumed run. API keys are read
from the named environment variable and must never be hardcoded or committed.

The final raw files are expected locally as:

```text
Scripts_code/outputs/attack_attempts_unified_api_llm_llama70b_gen5_query5.csv
Scripts_code/outputs/attack_results_unified_api_llm_llama70b_gen5_query5.csv
Scripts_code/outputs/attack_unified_meta_api_llm_llama70b_gen5_query5.json
```

Invalid candidates are logged but are not queried against the classifier.

### 4. Final analysis

```powershell
python Scripts_code\07_analysis_results.py `
  --analysis unified_llama70b_gen5_query5
```

Derived analysis files are written beneath:

```text
Scripts_code/outputs/analysis_unified_llama70b_gen5_query5/
```

### 5. Optional figures and diagnostics

```powershell
python Scripts_code\09_unified_figures.py
python Scripts_code\06_budget_diagnostic.py --help
```

`08_confidence_movement_analysis.py` and
`08_create_results_figures.py` preserve earlier post-hoc analysis workflows.
`09_unified_figures.py` is the figure script aligned with the final unified
70B analysis.

## Local model helpers

The fixed SBERT validity model can be frozen locally with:

```powershell
python Scripts_code\freeze_sbert_model.py
```

`freeze_local_llm_model.py` preserves the earlier local-model workflow for
reproducibility history. The final 70B paraphraser was API-served; its weights
are not downloaded or stored by this repository.
