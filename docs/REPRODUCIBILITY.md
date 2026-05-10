# Reproducibility Notes

This project is designed as a controlled robustness evaluation. The goal is to isolate the effect of classifier feedback granularity, not to compare many classifiers, many LLMs, or many attack settings at once.

---

## Experimental Control

The comparison has two feedback conditions:

1. `label_only`
2. `score_based`

Everything else should remain fixed across conditions:

- same victim classifier;
- same attacked examples;
- same canonical input column;
- same paraphraser model/backend;
- same prompt version;
- same generation settings;
- same validity checker;
- same semantic-similarity threshold;
- same classifier-query budget;
- same success definition;
- same logging schema.

The only intended difference is whether the attacker receives only the predicted label or the predicted label plus confidence.

---

## Data Split Policy

The pipeline separates development/pilot work from final evaluation.

```text
hippocorpus_training_truncated.csv -> train/dev side
hippocorpus_test_truncated.csv     -> final test side
```

The pilot pool is sampled from train/dev data.

The main pool is sampled from final test data.

The final test side should not be used for prompt tuning, threshold tuning, or debugging decisions after the protocol is frozen.

---

## Canonical Text Field

The canonical model/attack input is:

```text
text_truncated
```

The pipeline stores this as:

```text
attack_text
```

This avoids attacking a text field that differs from what the victim classifier actually sees.

---

## Candidate Pool Policy

Only examples that the victim classifier originally classifies correctly are eligible for attack.

This prevents already-misclassified items from being counted as adversarial robustness failures.

Candidate pools are deduplicated by a stable hash of the attacked text where supported by the script metadata.

---

## Frozen Attack Pools

The project uses two frozen attacked subsets:

```text
candidate_pool_pilot_20.csv
candidate_pool_main_160.csv
```

The pilot set is for debugging and protocol stabilization.

The main set is for the final paired comparison.

These files are generated outputs and are not committed to Git. Their creation metadata is written to:

```text
candidate_pool_freeze_meta.json
```

---

## Query Accounting

The original-text verification query is logged separately.

The attack budget applies only to candidate classifier queries inside the attack loop.

Invalid candidates are logged but are not queried against the classifier.

This matters because the project measures query efficiency as part of the robustness comparison.

---

## Validity Checking

The validity checker decides whether a generated paraphrase is eligible to be considered an adversarial candidate.

It does not decide attack success.

A candidate must pass:

- basic quality checks;
- non-identical check;
- minimum token/length checks;
- SBERT semantic similarity;
- lightweight number/date/negation consistency checks.

Current semantic similarity threshold:

```text
SBERT cosine similarity >= 0.86
```

This threshold should be kept fixed once the final protocol is frozen.

---

## Success Definition

A successful attack requires all of the following:

1. the original item was correctly classified;
2. the candidate paraphrase is valid;
3. the classifier prediction flips;
4. the flip occurs within the fixed classifier-query budget.

A raw prediction flip is not enough if the paraphrase fails validity checks.

---

## Local Models

The project uses local model folders for reproducibility and privacy.

Expected local models:

```text
DistilBERT/                       # fixed victim classifier
local_models/all-MiniLM-L6-v2/    # SBERT validity model
local_models/Qwen3-4B-Instruct-2507/
local_models/Qwen3-4B-Instruct-2507-GGUF/
```

These are not committed to Git.

For final runs, record:

- victim model path/hash where available;
- SBERT model path/hash where available;
- Qwen model id;
- GGUF filename;
- GGUF file hash;
- backend type;
- llama.cpp version/build if using llama.cpp;
- context size;
- GPU offload setting;
- generation settings;
- prompt version and prompt hash.

---

## Output Metadata

The scripts are expected to write metadata JSON files for generated outputs.

Metadata should make it possible to reconstruct:

- input paths;
- output paths;
- row counts;
- hashes;
- split provenance;
- seed values;
- model/backend settings;
- validity-check settings;
- query-budget settings;
- prompt settings;
- integrity-check status.

Do not manually edit generated metadata files.

---

## Overwrite Policy

Scripts should refuse to overwrite important outputs unless `--overwrite` is explicitly provided.

Use `--overwrite` only when the output is intentionally being regenerated.

For final/paper/thesis runs, avoid overwriting previous run outputs unless the older run has been backed up or is known to be invalid.

---

## What Not to Change Between Conditions

Do not change these separately for `label_only` and `score_based`:

- attacked examples;
- prompt version, except the feedback block;
- paraphraser model;
- paraphraser backend;
- temperature/top-p/top-k settings;
- max tokens;
- classifier-query budget;
- validity thresholds;
- semantic model;
- stopping rules.

Changing any of these differently across conditions would make the comparison harder to interpret.
