# Can Large Language Models Fool Deception Classifiers? A Controlled Black-Box Study of Feedback Granularity in Semantic-Preserving Paraphrase Attacks

Research code for a controlled black-box study of feedback granularity in
semantic-preserving paraphrase attacks.

The project asks whether a strong instruction-tuned language model can fool a
fixed deception classifier by paraphrasing its input without changing its
meaning, and whether exposing classifier confidence improves the attack over
label-only feedback.

## Final experiment

- Victim: fixed pretrained DistilBERT deception classifier.
- Data: Hippocorpus truthful/deceptive text task.
- Attack pool: 389 correctly classified held-out examples (170 truthful and
  219 deceptive).
- Design: every example was attacked under both feedback conditions, for 778
  paired attack runs.
- Paraphraser: `meta-llama/Llama-3.3-70B-Instruct-Turbo`, a 70B-class
  open-weight instruction-tuned model served through the DeepInfra API.
- Budget: `gen5_query5` (at most five generation attempts and five valid
  classifier queries per example and condition).
- Success: a candidate must pass the validity gate and flip the classifier
  prediction within budget.

The two conditions differ only in the feedback returned to the paraphraser:

1. `label_only`: predicted class only.
2. `score_based`: predicted class and predicted-class confidence.

Invalid candidates are logged but never submitted to the victim classifier.
The shared validity gate applies an SBERT cosine-similarity threshold of 0.86,
length-ratio and basic-quality checks, number/date/negation consistency checks,
and duplicate handling.

## Headline results

| Feedback | Successes | N | Valid ASR |
| --- | ---: | ---: | ---: |
| Label-only | 97 | 389 | 24.9% |
| Score-based | 90 | 389 | 23.1% |

The exact McNemar test gave `p = .296`. Overall, 2,573 of 3,285 generated
candidates were valid (78.3%), and zero invalid candidates were queried.
Exposing confidence did not improve attack success or query efficiency under
the fixed low-budget protocol. See [RESULTS.md](RESULTS.md) for the full
summary.

## Repository layout

```text
.
|-- README.md
|-- RESULTS.md
|-- requirements.txt
|-- .gitignore
|-- docs/
|   |-- RUNNING.md
|   |-- REPRODUCIBILITY.md
|   `-- LOCAL_ARTIFACTS.md
`-- Scripts_code/
    |-- 01_baseline_inference.py
    |-- 02_make_attack_pool.py
    |-- 02_make_exploratory_remaining_pool.py
    |-- 03_classifier_wrapper.py
    |-- 04_validity_checks.py
    |-- 05_attack_pilot.py
    |-- 06_budget_diagnostic.py
    |-- 07_analysis_results.py
    |-- 08_confidence_movement_analysis.py
    |-- 08_create_results_figures.py
    |-- 09_unified_figures.py
    |-- freeze_sbert_model.py
    |-- freeze_local_llm_model.py
    `-- outputs/.gitkeep
```

`Scripts_code/05_attack_pilot.py` has a historical filename; it is the main
attack runner used for the final unified experiment. The two `08_*` scripts
serve different post-hoc analysis/figure roles and retain their established
names. Some scripts preserve earlier pilot configurations for provenance, but
the final thesis-relevant analysis is `unified_llama70b_gen5_query5`.

## Installation

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build separately if required by the local
machine.

## Local artifacts

Data, model weights, raw outputs, and generated paraphrases are intentionally
not committed. The current code expects local data and models beneath:

```text
Automated Deception Classifier (Projectfolder)/
```

See [docs/LOCAL_ARTIFACTS.md](docs/LOCAL_ARTIFACTS.md) for the expected paths.

## Pipeline

Run commands from the repository root.

```powershell
# 1. Fixed-classifier baseline inference
python Scripts_code\01_baseline_inference.py

# 2. Construct/freeze attack pools
python Scripts_code\02_make_attack_pool.py
python Scripts_code\02_make_exploratory_remaining_pool.py

# 3. Run the attack (historical filename, current main runner)
python Scripts_code\05_attack_pilot.py --help

# 4. Analyze the final unified output
python Scripts_code\07_analysis_results.py `
  --analysis unified_llama70b_gen5_query5

# 5. Optional figures and diagnostics
python Scripts_code\09_unified_figures.py
```

The final attack command and operational safeguards are documented in
[docs/RUNNING.md](docs/RUNNING.md). Do not rerun the completed experiment
unless you intentionally want a new set of raw outputs.

## API-key safety

Store the DeepInfra key only in an environment variable for the current shell:

```powershell
$env:DEEPINFRA_API_KEY = "your-key-here"
```

Pass the variable name with `--api-llm-key-env DEEPINFRA_API_KEY`. Never put a
key in source code, command examples, tracked configuration, notebooks, or
output files.

## Reproducibility and repository policy

The comparison uses the same victim, examples, prompt family, generation
settings, validity checks, budget, and stopping rules in both feedback
conditions. Generated outputs are excluded from Git; the repository contains
code, documentation, and aggregate results only.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the frozen protocol
and [docs/RUNNING.md](docs/RUNNING.md) for execution details.
