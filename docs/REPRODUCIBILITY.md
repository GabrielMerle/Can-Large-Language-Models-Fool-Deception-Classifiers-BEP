# Reproducibility

## Frozen final design

The final study is a paired, controlled black-box comparison:

- fixed pretrained DistilBERT victim classifier;
- Hippocorpus truthful/deceptive task;
- 389 correctly classified held-out examples (170 truthful, 219 deceptive);
- each example attacked once with `label_only` feedback and once with
  `score_based` feedback;
- 778 total attack runs;
- `meta-llama/Llama-3.3-70B-Instruct-Turbo` served through DeepInfra;
- seed 42;
- temperature 0.45, `top_p` 0.85, maximum 220 completion tokens;
- `gen5_query5`: at most five generation attempts and five valid classifier
  queries per example and condition.

The paraphraser is correctly described as a 70B-class open-weight
instruction-tuned model served through an API, not as a fully local model.

## Controlled comparison

The following are held fixed across feedback conditions:

- victim classifier and inference path;
- attacked examples and ordering-independent row/condition tasks;
- paraphraser model and provider;
- prompt family and generation settings;
- validity checker and thresholds;
- generation and classifier-query budgets;
- success definition and stopping rules;
- logging schema.

Only the feedback differs: the predicted label alone, or the predicted label
plus confidence.

## Eligibility and success

Only examples correctly classified by the victim on the held-out split enter
the attack pool. An attack succeeds only if:

1. the generated paraphrase passes every required validity check;
2. the victim prediction flips;
3. the flip occurs within the fixed budget.

The original-text verification is separate from the attack query budget.
Invalid candidates are logged but never queried.

## Validity gate

The shared gate includes:

- SBERT cosine similarity of at least 0.86;
- minimum length-ratio and basic text-quality checks;
- non-identical and duplicate handling;
- number consistency;
- date consistency;
- negation consistency.

The same gate is applied before victim inference in both conditions.

## Reproducing derived results

Place the frozen raw output files in `Scripts_code/outputs/`, then run:

```powershell
python Scripts_code\07_analysis_results.py `
  --analysis unified_llama70b_gen5_query5
```

This analysis is read-only with respect to the raw result, attempt, and
metadata files. Optional final figures are generated with:

```powershell
python Scripts_code\09_unified_figures.py
```

Aggregate results are recorded in the repository-level `RESULTS.md`; raw text,
candidate paraphrases, and run outputs remain local.

## Provenance and limitations

Reproduction requires access to the Hippocorpus files, the fixed victim model,
the frozen SBERT validity model, and a compatible API endpoint/model. Provider
infrastructure can introduce operational variation even when model name,
prompt, seed, and sampling settings are fixed. Preserve the run metadata and
file hashes produced by the scripts when conducting a new run.

Earlier `main_gen3_query3` and `exploratory_remaining_gen3_query3`
configurations remain in some scripts for provenance. They are not the final
thesis-relevant experiment.
