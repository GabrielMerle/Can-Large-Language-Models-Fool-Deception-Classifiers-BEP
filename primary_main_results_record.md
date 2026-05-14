# Primary Main Experiment Results Record

## Project
Can Large Language Models Fool Deception Classifiers?

## Primary experiment
This file records the verified primary main-experiment result for the frozen BEP setup.

## Frozen setup
- Victim model: fixed pretrained DistilBERT deception classifier
- Dataset split: held-out test/final side
- Main pool: 160 originally correctly classified examples
- Design: paired comparison, same 160 examples attacked under both feedback conditions
- Feedback conditions:
  - label_only: predicted label only
  - score_based: predicted label + predicted-class confidence
- Paraphraser: fixed local Qwen3 GGUF model via llama.cpp server
- Prompt/budget setup: main_gen3_query3
- Maximum generation attempts per condition/example: 3
- Maximum classifier queries per condition/example: 3
- Validity checks: shared semantic/quality checks applied before classifier query
- Success definition: valid paraphrase + prediction flip within budget

## Integrity checks
The analysis script reported:
- results_rows_is_320: true
- unique_row_ids_is_160: true
- each_row_id_has_label_only_and_score_based_once: true
- metadata_integrity_checks_passed_is_true: true
- metadata_experiment_split_is_main: true
- metadata_input_pool_num_rows_is_160: true
- metadata_budget_config_is_gen3_query3: true
- metadata_prompt_version_matches_expected: true
- original_predictions_equal_gold_labels: true

## Headline ASR results
- label_only: 19 / 160 = 11.875%
- score_based: 15 / 160 = 9.375%
- score_based minus label_only: -2.5 percentage points

## Paired success table
- both success: 15
- label_only only: 4
- score_based only: 0
- neither success: 141
- total pairs: 160

## McNemar paired comparison
- b = 4
- c = 0
- discordant pairs = 4
- exact two-sided p-value = 0.125
- interpretation: no statistically detectable paired success difference at alpha = 0.05

## Query efficiency among successful attacks
- overall mean queries among successes: 1.382
- label_only mean queries among successes: 1.579
- score_based mean queries among successes: 1.133

Interpretation: score_based was descriptively more query-efficient among successful attacks, but this should be interpreted cautiously because the number of successes is small.

## Validity
- total attempts: 910
- valid attempts: 579
- invalid attempts: 331
- overall validity rate: 63.626%
- invalid candidates queried: 0

## Top failed individual checks
- negation_consistency: 177 attempts, 19.451%
- numbers_consistency: 137 attempts, 15.055%
- sbert_similarity: 32 attempts, 3.516%
- dates_consistency: 16 attempts, 1.758%
- duplicate_candidate: 9 attempts, 0.989%
- not_identical: 3 attempts, 0.330%

## Thesis-safe interpretation
In the primary frozen main experiment, score-based feedback did not increase valid attack success rate compared with label-only feedback. Label-only feedback had a descriptively higher ASR, but the paired McNemar test did not show a statistically detectable difference at alpha = 0.05. Among successful attacks, score-based feedback required fewer classifier queries on average, but this efficiency result should be treated as descriptive because only a small number of attacks succeeded. Overall, the primary result does not support H1 and gives only cautious descriptive support for H2.

## Source files
Raw experiment files:
- Scripts_code/outputs/attack_results_main_local_llm_main_gen3_query3.csv
- Scripts_code/outputs/attack_attempts_main_local_llm_main_gen3_query3.csv
- Scripts_code/outputs/attack_main_meta_local_llm_main_gen3_query3.json

Analysis folder:
- Scripts_code/outputs/analysis_main_gen3_query3/

Analysis summary:
- Scripts_code/outputs/analysis_main_gen3_query3/summary_main_gen3_query3.json
