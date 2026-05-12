# BEP Project Progress Log and Methods Record

## Project
**Can Large Language Models Fool Deception Classifiers?**

## Purpose of this document
This file records exactly what was done in the project session, why each step was taken, what design choices were made, what went wrong, how those issues were fixed, and how the current implementation aligns with the latest BEP plan. It is written to serve two purposes at once:

1. a practical reminder of what has already been built and why it was built that way;
2. a methods/approach draft source for the thesis.

The document therefore mixes project log, implementation notes, and thesis-style methodological justification.

---

# 1. Big-picture research logic

The current BEP is a **controlled black-box robustness study**. The thesis does **not** try to build a new deception classifier. It evaluates how robust the provided pretrained **DistilBERT deception classifier** remains when its input statements are paraphrased by an LLM under two different black-box feedback conditions:

- **label-only**
- **score-based** (label + confidence/probability signal)

The latest plan intentionally narrows the thesis to this comparison. The main independent variable is **feedback granularity**. Everything else is supposed to stay as constant as possible:

- same victim classifier
- same attacked examples
- same paraphrase generation procedure
- same validity checks
- same query budget logic
- same logging structure

The reason for this design is methodological control. If several pipeline parts change together, then any final difference in results becomes hard to interpret. The thesis therefore treats the comparison as a **paired, controlled experiment** rather than as a broad benchmark.

## Main outcomes the thesis is trying to measure
The project is built around two main dependent variables:

1. **Attack effectiveness**
   - operationalized as **Attack Success Rate (ASR)** under fixed validity constraints and fixed query budgets.

2. **Query efficiency**
   - operationalized as the number of classifier queries required to reach the **first valid successful prediction flip**.

The plan also says that a successful attack is **not** any paraphrase that changes the prediction. The paraphrase must also remain semantically close enough to the original and pass basic quality checks. This is crucial because otherwise the result could reflect semantic drift rather than a genuine robustness failure.

---

# 2. What we finished today at a high level

Today’s work covered the full transition from a vague project skeleton to a much cleaner and more defensible experimental foundation.

By the end of the session, the project had a working implementation of:

- a reproducible **baseline inference script**
- a reproducible **attack-pool freezing script**
- a clean **victim-model wrapper** exposing the two feedback conditions
- a clean **validity checker** for semantic-preserving paraphrases
- a frozen local **SBERT** similarity model
- a corrected and defensible split between:
  - **train/dev** for pilot work and threshold/prompt tuning
  - **test/final** for the main confirmatory experiment

So the session did **not** build the actual attack loop yet. Instead, it built the full foundation that the attack loop will rely on.

That was the right order. The attack loop should only be built after the data split, victim model interface, and validity gate are clean.

---

# 3. Why this order made sense

The project logic now follows a layered execution strategy:

## Layer 1. Make the target system and data setup clean
Before attacking anything, the project first needed a stable baseline showing:

- what the classifier predicts on the provided data,
- which examples are originally classified correctly,
- what text field the model should actually see,
- and which examples should even be eligible for attack.

## Layer 2. Freeze the attacked examples
The project should not repeatedly resample attack examples in a hidden or unstable way. Since the thesis compares two feedback conditions, both conditions should attack the **same fixed examples**. That means the attacked pools must be frozen.

## Layer 3. Define the only difference between conditions
The classifier wrapper had to be designed so that:

- the underlying model stays the same,
- the inference path stays the same,
- the preprocessing stays the same,
- and only the **returned signal** changes.

Without that, the comparison would not isolate feedback granularity.

## Layer 4. Define a validity gate before measuring attack success
The thesis is explicitly about **semantic-preserving paraphrase attacks**, not arbitrary rewrites. So before any future attack can count as successful, the project needed a reusable module that decides whether a candidate paraphrase is valid enough to count as an adversarial candidate.

Only after these four layers are stable does it make sense to build the actual attack loop.

---

# 4. Baseline inference: what we built and why

## Script
`01_baseline_inference.py`

## Purpose
This script runs the provided pretrained DistilBERT deception classifier on the provided Hippocorpus data and saves structured outputs that later stages depend on.

## What the script now does
The final version of the script now:

- loads the provided pretrained **DistilBERT** model from the project materials;
- runs inference on **two separate datasets**:
  - `hippocorpus_training_truncated.csv` → used as **train/dev**
  - `hippocorpus_test_truncated.csv` → used as **test/final**
- uses **`text_truncated`** as the canonical attack/model input field;
- stores that canonical input in a dedicated column named **`attack_text`**;
- stores the original full text separately as **`full_text`**;
- stores `source_text_column = text_truncated` in the outputs;
- maps labels consistently as:
  - `0 -> truthful`
  - `1 -> deceptive`
- computes predictions, confidence scores, probability vectors, correctness, confusion-style counts, and class-wise results;
- builds a **candidate pool** consisting only of examples that were originally classified correctly;
- deduplicates candidate rows by the hash of `attack_text`;
- writes separate outputs for the train/dev split and the final test split.

## Why we changed the canonical text field to `text_truncated`
This was one of the most important fixes of the session.

Originally, there was a real risk that the project would use the wrong text field. The truncated dataset files contain both full text and truncated text. At first, the baseline had been using the full `text` field. That creates a methodological problem because the classifier itself operates under a fixed token limit. If the paraphraser attacks the full text, but the classifier effectively only uses a truncated version of it, then the attack may rewrite content that the victim model never actually sees.

That would make the experiment conceptually messy.

Using **`text_truncated`** everywhere fixes this by aligning:

- the text seen by the classifier,
- the text treated as the attacked statement,
- and the text later passed into the paraphrase pipeline.

This is why the baseline now stores:

- `attack_text` = canonical attacked text = `text_truncated`
- `full_text` = original full text kept only as extra context

## Why separate train/dev and test/final outputs were necessary
At first, the project was using a single candidate pool derived from the final test side. That created a methodological problem because the pilot subset was being sampled from the same held-out test-derived pool that the main evaluation was supposed to use.

That is bad practice because it means development decisions would be influenced by the final test data.

The baseline was therefore redesigned to produce:

- `baseline_predictions_train_dev.csv`
- `candidate_pool_train_dev.csv`
- `baseline_summary_train_dev.json`

and

- `baseline_predictions_test_final.csv`
- `candidate_pool_test_final.csv`
- `baseline_summary_test_final.json`

This cleanly separates:

- **development / pilot work** from
- **final confirmatory evaluation**.

## Why candidate pools contain only originally correctly classified examples
This follows the standard adversarial-robustness logic.

A meaningful robustness failure happens when:

1. the original example is classified correctly;
2. a valid paraphrase is generated;
3. the classifier prediction flips.

If the model already gets the original example wrong, then later paraphrasing it does not demonstrate a real robustness failure. It just starts from an existing error.

That is why only correctly classified examples are eligible for the attack pools.

## Why deduplication by text hash was added
The train/dev pool in particular can contain repeated or effectively duplicated attack texts. If duplicates remain, then:

- pilot results can be artificially inflated,
- the apparent sample size becomes less meaningful,
- and the later frozen subset is less diverse than it appears.

To avoid this, candidate pools are deduplicated by a stable SHA-256 hash of `attack_text`.

## Final baseline outputs after the fixes
The corrected outputs reported during the session were:

### Train/dev baseline
- rows: 4542
- accuracy: 0.8653
- correctly classified before dedup: 3930
- duplicate attack-text hashes dropped: 3
- final train/dev candidate pool rows: 3927

### Test/final baseline
- rows: 505
- accuracy: 0.7703
- correctly classified: 389
- final test candidate pool rows: 389

These numbers matter because they define the size of the available correctly classified pool for later pilot and main subset freezing.

---

# 5. Attack-pool freezing: what we built and why

## Script
`02_make_attack_pool.py`

## Purpose
This script freezes the attacked subsets that later stages will use.

## What the script now does
The corrected version now:

- loads the **train/dev candidate pool** as the source for the pilot subset;
- loads the **test/final candidate pool** as the source for the main subset;
- samples a balanced **pilot subset of 20 rows**:
  - 10 truthful
  - 10 deceptive
- samples a balanced **main subset of 160 rows**:
  - 80 truthful
  - 80 deceptive
- checks that the two frozen subsets do not overlap;
- checks source-key separation and text-hash separation;
- stores rich metadata describing where the subsets came from;
- refuses to overwrite existing frozen outputs unless `--overwrite` is explicitly passed.

## Why the pilot must come from train/dev and the main subset from test/final
This was one of the two biggest methodological blockers found during the session.

Earlier, both pilot and main had been sampled from the same test-derived candidate pool. That means the pilot would have contaminated the final held-out evaluation side.

The corrected logic is now:

- **Pilot / development subset** ← `candidate_pool_train_dev.csv`
- **Main / confirmatory subset** ← `candidate_pool_test_final.csv`

This ensures that:

- prompt tuning,
- threshold tuning,
- and debugging

do not implicitly use the final test examples.

## Why balanced class sampling was used
The project compares truthful and deceptive classes and also keeps an exploratory directional analysis in scope. A strongly imbalanced attacked subset could distort the comparison, especially if one class is harder to flip than the other.

Balanced sampling therefore makes the comparison cleaner and the later analysis easier to interpret.

## Why fixed seed and overwrite protection matter
The script uses `SEED = 42` to make sampling reproducible.

However, a fixed seed alone is not enough if the outputs can still be silently overwritten. That was another issue discovered during the session. If frozen pools can be recreated invisibly, then the word “frozen” stops meaning anything.

That is why overwrite protection was added. Now the script:

- refuses to overwrite by default,
- only replaces the existing frozen files when `--overwrite` is explicitly passed.

This makes the attack pools genuinely frozen artifacts rather than temporary convenience files.

## Why rich metadata was added
The freeze metadata now stores:

- source dataset paths,
- source split counts,
- source pool file hashes,
- canonical text column,
- selection rule,
- seed,
- and row identifiers / hashes for the frozen subsets.

This is useful for both reproducibility and thesis writing. If you later need to explain exactly how the pilot and main subsets were constructed, the metadata file already contains that information.

## Final frozen outputs after the fixes
The corrected frozen results reported during the session were:

### Pilot
- 20 rows total
- all from **train_dev**
- balanced:
  - 10 truthful
  - 10 deceptive

### Main
- 160 rows total
- all from **test_final**
- balanced:
  - 80 truthful
  - 80 deceptive

### Separation checks
- pilot/main source-key overlap: 0
- pilot/main text-hash overlap: 0

This means the attacked subsets are now methodologically clean.

---

# 6. Classifier wrapper: what we built and why

## Script
`03_classifier_wrapper.py`

## Purpose
This script turns the fixed pretrained DistilBERT deception classifier into a narrow, controlled black-box victim interface with two feedback modes.

## Design principle
The wrapper was designed to obey a very strict rule:

> same underlying model, same preprocessing, same forward pass, same label mapping, same text input;
> only the exposed feedback differs.

That rule is the heart of the whole thesis.

## What the wrapper now does
The final wrapper:

- loads the fixed pretrained DistilBERT victim model;
- uses a single shared inference path internally;
- exposes two public methods:
  - `predict_label(...)`
  - `predict_label_and_confidence(...)`
- returns only the predicted class in the first case;
- returns predicted class plus predicted-class confidence in the second case;
- tracks query counts separately for:
  - `label_only`
  - `score_based`
- also tracks total query count;
- exposes metadata about the model path, tokenizer path, device, and label mapping.

## Why we did not put attack logic into this file
The wrapper intentionally does **not** contain:

- paraphrase generation,
- validity checks,
- stopping rules,
- experiment orchestration,
- logging of attack runs.

This separation is important because the thesis wants to isolate feedback granularity as the main variable. If the wrapper started to include other logic, the design would become harder to interpret.

## Why the query-count definition matters
At one point we discussed whether every public wrapper call should count as a query, even if the input was empty or invalid. The final choice was:

- a query is counted only after local validation/preprocessing succeeds and the victim model is actually queried.

This is cleaner than counting invalid local inputs as model queries.

The wrapper therefore counts only **successful victim-model calls**.

## Why score-based output returns confidence only
There was also discussion about whether the score-based interface should expose the full probability vector. In a binary classifier, the predicted-class confidence already implies the other class probability. The public interface was therefore tightened so that score-based feedback means:

- predicted label
- predicted-class confidence

That keeps the public semantics clean and aligned with the thesis framing.

## Smoke test result
The final smoke test showed:

- the wrapper loads the model successfully;
- label-only returns label information;
- score-based returns label + confidence;
- query counts by condition behave as expected.

So the wrapper is now in a good state and aligned with the BEP design.

---

# 7. Validity checker: what we built and why

## Script
`04_validity_checks.py`

## Purpose
This script is the **gatekeeper** for semantic-preserving paraphrase attacks.

It does **not** decide whether an attack succeeded. It decides only whether a generated paraphrase is valid enough to count as an adversarial candidate.

That distinction matters:

- **validity** and
- **successful attack**

are not the same thing.

A successful attack later requires:

1. original example was correctly classified,
2. paraphrase passes the validity checks,
3. classifier prediction flips,
4. all within the fixed query budget.

## Why a validity gate was necessary
This follows directly from the thesis logic.

A raw prediction flip is not enough, because text attacks can easily become misleading if the paraphrase:

- changes meaning,
- becomes ungrammatical,
- adds prompt junk,
- changes numbers or dates,
- flips negation,
- or becomes so degraded that it is no longer a fair version of the original statement.

The project therefore needed a dedicated reusable module that filters candidate paraphrases before they can count toward attack success.

## What the checker now does
The final checker implements three layers:

### Stage A. Hard rejection checks
The candidate is rejected if it is:

- non-string,
- empty after normalization,
- too short,
- identical to the original,
- or clearly malformed.

The malformed check includes things like:

- only punctuation,
- repeated junk characters,
- prompt-meta outputs such as “Here is the paraphrase: …”,
- obvious LLM wrapper text.

### Stage B. Semantic-preservation check
The checker computes **SBERT cosine similarity** between the original and the candidate.

The project uses this as the main semantic thresholding criterion.

### Stage C. Lightweight meaning-drift checks
The checker also applies simple rule-based checks for:

- number consistency,
- date-like marker consistency,
- negation consistency.

These are intentionally lightweight. They are not meant to solve semantics perfectly. They are there to catch obvious task-relevant drift.

## Why SBERT was chosen
SBERT was chosen because it is efficient and specifically designed for sentence-level semantic similarity.

The project is not using BERTScore as the main gate, because BERTScore is more useful as a descriptive secondary quality signal than as the primary iterative filter.

## Why the threshold ended up at 0.86
The threshold discussion was one of the more important parts of the session.

At first, the checker used **0.88**. In the smoke test, a clearly reasonable paraphrase scored about **0.8706** and was rejected. That was a sign that the threshold was likely just a bit too strict for this paraphrase-based setup.

After reviewing the methodological context and the literature, the threshold was lowered slightly to:

- **`MIN_SBERT_SIMILARITY = 0.86`**

Why this made sense:

- `0.88` looked slightly too strict in practice for natural paraphrases;
- much lower values would risk letting in looser rewrites;
- the checker already has additional conservative safeguards for numbers, dates, and negation;
- so SBERT does not need to do all the work by itself.

With the new threshold:

- the “good” candidate passed;
- the “bad” candidate still failed.

That is exactly the behavior we wanted.

## Why bad candidate outputs now return invalid results instead of crashing
Earlier, there was a real problem: empty or malformed candidate outputs could trigger exceptions and crash the run.

This was fixed by changing candidate normalization so that bad **candidate outputs** now return a structured `ValidityEvaluation` with:

- `is_valid = False`
- the failed check name
- no semantic similarity score

This is much better for the future attack loop because an LLM producing a bad candidate is not a reason for the experiment to crash. It is just an invalid attempt that should be logged and skipped.

The checker still raises for invalid **original inputs**, which is appropriate because that would indicate a data/configuration problem rather than a normal attack failure.

## Why the checker uses a frozen local SBERT path
At first, the validity checker still loaded the SBERT model by its Hugging Face name. That meant it could download the model on first run and was not fully frozen locally.

This was fixed in two steps:

1. a one-time `freeze_sbert_model.py` script was used to save the SBERT model into the project folder;
2. `04_validity_checks.py` was then changed to load the model only from the local frozen path using `local_files_only=True`.

This improved reproducibility and removed online dependency from the validity-checking step.

## Why the date/number checks are intentionally conservative
The checker currently requires exact matches for extracted numbers and date-like markers.

That is strict. It means the checker may reject some paraphrases that humans might still consider acceptable, such as:

- “2 hours” → “two hours”
- “Saturday” → “that day”

This is a deliberate conservative choice.

Why it is still acceptable:

- the same validity filter is applied to both feedback conditions,
- so the final comparison stays fair,
- and for a thesis on semantic-preserving attacks, being conservative is better than accidentally counting meaning-drifting paraphrases as valid.

This will need to be described honestly in the thesis: the rule-based validity filter is conservative and operational, not a perfect semantic-equivalence oracle.

---

# 8. What went wrong during the session and how we fixed it

A lot of today’s progress came from discovering hidden problems and correcting them before they could contaminate later experiments.

## Problem 1. The wrong attacked text field was being used
### What went wrong
The baseline had originally been using the full `text` field rather than `text_truncated`.

### Why this was a problem
This risks a mismatch between:

- the text the paraphraser attacks,
- and the text the classifier effectively sees under its own token-length limit.

### Fix
We made `text_truncated` the canonical input everywhere and stored it explicitly as `attack_text`.

---

## Problem 2. Pilot and main were both sampled from the test-derived pool
### What went wrong
The pilot subset had originally been sampled from the same final test-derived candidate pool as the main subset.

### Why this was a problem
That contaminates the held-out test side and makes development decisions less clean.

### Fix
We changed the logic so that:

- pilot comes from `candidate_pool_train_dev.csv`
- main comes from `candidate_pool_test_final.csv`

---

## Problem 3. Frozen outputs could be overwritten silently
### What went wrong
The freezer script could recreate “frozen” subsets without much protection.

### Why this was a problem
That weakens reproducibility and makes later results harder to defend.

### Fix
`02_make_attack_pool.py` now refuses overwrites unless `--overwrite` is explicitly passed.

---

## Problem 4. Label naming was inconsistent across scripts
### What went wrong
Earlier baseline outputs could still use generic labels like `LABEL_0` and `LABEL_1`, while the wrapper and thesis logic used `truthful` and `deceptive`.

### Why this was a problem
That creates avoidable ambiguity and can later lead to bugs or interpretation mistakes.

### Fix
The label naming was aligned consistently as:

- `truthful`
- `deceptive`

---

## Problem 5. Empty or bad candidate paraphrases could crash the validity checker
### What went wrong
The validity checker originally normalized the candidate in a way that could raise errors before the checker could even record the failure as an invalid candidate.

### Why this was a problem
An attack experiment should survive bad LLM outputs and log them, not crash.

### Fix
Candidate normalization was redesigned so that bad candidate outputs return structured invalid results instead of stopping the run.

---

## Problem 6. The SBERT threshold was slightly too strict
### What went wrong
The first threshold (`0.88`) rejected a reasonable paraphrase that looked valid.

### Why this was a problem
That would make the project unnecessarily strict and could suppress genuinely meaningful attacks.

### Fix
The threshold was adjusted to `0.86`, after which the good smoke-test paraphrase passed and the bad prompt-meta paraphrase still failed.

---

## Problem 7. The wrong version of `04_validity_checks.py` was being run
### What went wrong
At one point, the code file being executed still contained the old model-name loading path and old threshold values even after the corrected version had been prepared.

### Why this was a problem
This created confusion about whether the checker was really using the frozen local SBERT model and the new threshold.

### Fix
We verified the actual file contents, replaced the stale version, reran the checker, and confirmed from the output metadata that:

- it loads from the local SBERT path,
- it uses threshold `0.86`,
- and it no longer behaves like the older online-loading version.

---

# 9. Why the current implementation is now methodologically defensible

At this point, scripts 01–04 are in a good place because they now match the intended BEP logic much more closely.

## Controlled comparison logic now in place
The project now has:

- one fixed victim model;
- one fixed canonical attacked text field;
- one clean train/dev vs test/final split;
- one fixed pilot subset;
- one fixed main subset;
- one wrapper exposing only the two intended feedback modes;
- one separate validity checker used identically across conditions.

That means the actual future comparison can now isolate **feedback granularity** much more credibly.

## Reproducibility logic now in place
The project now also has:

- stable paths,
- explicit split names,
- hashes,
- fixed seed,
- explicit overwrite control,
- metadata files,
- frozen local SBERT,
- and clearer logging structure.

These are the kinds of things that make both the thesis methods section and the later final results easier to defend.

---

# 10. Outputs and artifacts produced or stabilized today

## Baseline-related outputs
- `baseline_predictions_train_dev.csv`
- `candidate_pool_train_dev.csv`
- `baseline_summary_train_dev.json`
- `baseline_predictions_test_final.csv`
- `candidate_pool_test_final.csv`
- `baseline_summary_test_final.json`
- refreshed generic aliases for the test/final side

## Frozen attacked subsets
- `candidate_pool_pilot_20.csv`
- `candidate_pool_main_160.csv`
- `candidate_pool_freeze_meta.json`

## Core scripts stabilized
- `01_baseline_inference.py`
- `02_make_attack_pool.py`
- `03_classifier_wrapper.py`
- `04_validity_checks.py`
- `freeze_sbert_model.py`

## Frozen local model
- local SBERT model directory inside the project

---

# 11. What is still not done yet

Even though the foundation is now strong, the project is **not finished**. The following parts still belong to the next phase:

## 1. The actual attack loop
`05_attack_pilot.py` still needs to be designed and built.

## 2. End-to-end query semantics
The project still needs one final explicit rule for how query budgets are counted end-to-end in the attack loop.

The best current rule is:

- the original-text verification query is logged separately;
- the **20-query attack budget** applies only to candidate classifier queries during the attack loop.

## 3. The paraphrase-generation component
The actual LLM paraphrase generation logic still needs to be connected into the attack loop.

## 4. Main experiment execution
The paired main run under both feedback conditions still needs to be executed later.

## 5. Analysis scripts
Later scripts for ASR, query-efficiency comparison, cumulative budget analysis, and exploratory directional analysis still need to be built.

---

# 12. What the next step should be

The correct next step is now:

## `05_attack_pilot.py`

This script should:

1. load the frozen pilot subset;
2. take one original attacked statement at a time;
3. query the classifier on the original text for logging/verification;
4. generate one paraphrase candidate per iteration using a fixed prompt;
5. run the validity checker;
6. if valid, query the classifier under the current feedback condition;
7. stop on the first valid successful flip or when the budget is exhausted;
8. log every attempt.

That script will be the first time the current components are exercised together in one real loop.

---

# 13. Thesis-ready methods wording you can reuse later

The following text is intentionally written in a style close to thesis methods prose.

## 13.1 Study design summary
This project implements a controlled black-box robustness evaluation of a pretrained DistilBERT deception classifier. The study compares two feedback conditions, label-only and score-based, while holding the remainder of the attack pipeline fixed. Across both conditions, the same target classifier, attacked examples, paraphrase-generation logic, semantic-validity checks, query-budget logic, and logging procedure are used. The only intended difference between conditions is the amount of information returned by the classifier after each query.

## 13.2 Data handling summary
The empirical setup uses the provided Hippocorpus split. To avoid hidden inconsistencies between the attacked text and the effective model input, the canonical model/attack input is defined as `text_truncated`. This field is stored in the pipeline as `attack_text`, while the original full text is retained separately as contextual information only. The training truncated split is reserved for development and pilot work, whereas the test truncated split is reserved for the final confirmatory comparison.

## 13.3 Candidate-pool construction summary
Only examples that are originally classified correctly by the target model are eligible for attack. This follows the standard adversarial definition of a robustness failure, where the model first makes the correct decision and is then moved away from it by a valid perturbation. Candidate pools are deduplicated by a stable SHA-256 hash of the attacked text to avoid repeated or effectively identical statements entering the frozen subsets.

## 13.4 Frozen subset summary
Two frozen attacked subsets are used. The pilot subset consists of 20 examples sampled from the train/dev correctly classified pool and is used only for pipeline debugging and protocol stabilization. The main subset consists of 160 examples sampled from the correctly classified held-out test pool and serves as the confirmatory evaluation set. Both subsets are balanced by class as far as possible, and the freezing procedure stores split provenance, file hashes, and row-level identifiers to support reproducibility.

## 13.5 Victim-model interface summary
The victim classifier is exposed through a dedicated wrapper that preserves one shared inference path and two public black-box feedback modes. In the label-only condition, the attacker receives only the predicted class. In the score-based condition, the attacker receives the predicted class together with the predicted-class confidence. Query counting is tracked explicitly by condition and in total, and attack logic is intentionally kept separate from the wrapper to preserve methodological control.

## 13.6 Validity-check summary
A candidate paraphrase is treated as a valid adversarial candidate only if it passes three layers of checks. First, hard rejection checks exclude empty, malformed, trivially identical, or excessively degraded outputs. Second, sentence-level semantic similarity is measured using SBERT cosine similarity with a fixed threshold of 0.86. Third, lightweight rule-based checks reject obvious meaning drift, including major number changes, explicit date-like changes, and simple negation reversals where detectable. These checks are intentionally conservative and are applied identically across both feedback conditions.

## 13.7 Interpretation rule summary
A successful attack is defined strictly. The original statement must be classified correctly, the paraphrase must pass the validity checks, the classifier prediction must flip, and this must happen within the fixed query budget. This design avoids interpreting arbitrary or semantically drifting rewrites as genuine robustness failures.

---

# 14. Short final status

At the end of this session, the project is in a strong pre-attack-loop state.

What is now clean:

- the core BEP logic,
- the train/dev vs test/final split,
- the canonical attacked text definition,
- the frozen subsets,
- the wrapper interface,
- and the semantic-preserving validity gate.

What remains is no longer “fix the foundation.”
What remains is:

- build the pilot attack loop,
- keep the query-budget semantics explicit,
- and then move toward the full paired main comparison.

That is a good place to be.

---

# 15. References used to justify the current design

- Constâncio et al. (2023). *Deception detection with machine learning: A systematic review and statistical analysis.*
- Kleinberg et al. (2025). *Effective faking of verbal deception detection with target-aligned adversarial attacks.*
- Glenski et al. (2021). *Evaluating deception detection model robustness to linguistic variation.*
- Morris et al. (2020). *Reevaluating Adversarial Examples in Natural Language.*
- Maheshwary et al. (2021). *Generating Natural Language Attacks in a Hard Label Black Box Setting.*
- Maheshwary et al. (2021). *A Strong Baseline for Query Efficient Attacks in a Black Box Setting.*
- Mozes et al. (2021). *Contrasting Human- and Machine-Generated Word-Level Adversarial Examples for Text Classification.*
- Morris et al. (2020). *TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP.*
- Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.*
- Zhang et al. (2020). *BERTScore: Evaluating Text Generation with BERT.*
- Cheng et al. (2025). *Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text.*

