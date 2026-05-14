# Exploratory Remaining-Pool Results Record

## Project
Can Large Language Models Fool Deception Classifiers?

## Status
This file records the exploratory remaining-pool result. This result does not replace the primary main experiment. The primary result remains main_160_gen3_query3.

## Exploratory setup
- Pool: remaining held-out correctly classified test/final examples not used in candidate_pool_main_160.csv
- Pool file: Scripts_code/outputs/candidate_pool_exploratory_remaining.csv
- Pool size: 229 examples
- Class counts:
  - deceptive: 139
  - truthful: 90
- Overlap with primary main pool:
  - source_split + row_id overlap: 0
  - text_hash overlap: 0
- Design: paired comparison, same examples attacked under both feedback conditions
- Feedback conditions:
  - label_only
  - score_based
- Budget: gen3_query3
- Maximum generation attempts per condition/example: 3
- Maximum classifier queries per condition/example: 3
- Prompt version: qwen3_paraphrase_feedback_v3_diverse_conservative
- Paraphraser/backend: local Qwen3 GGUF via llama.cpp server
- Success definition: valid paraphrase + prediction flip within budget
- Invalid candidates were not queried

## Integrity checks
The analysis script reported:
- results_rows_is_458: true
- unique_row_ids_is_229: true
- each_row_id_has_label_only_and_score_based_once: true
- metadata_integrity_checks_passed_is_true: true
- metadata_experiment_split_is_exploratory: true
- metadata_input_pool_num_rows_is_229: true
- metadata_budget_config_is_gen3_query3: true
- metadata_prompt_version_matches_expected: true
- original_predictions_equal_gold_labels: true

## Headline ASR results
- label_only: 17 / 229 = 7.424%
- score_based: 20 / 229 = 8.734%
- score_based minus label_only: +1.310 percentage points

## Paired success table
- both success: 11
- label_only only: 6
- score_based only: 9
- neither success: 203
- total pairs: 229

## McNemar paired comparison
- b = 6
- c = 9
- discordant pairs = 15
- exact two-sided p-value = 0.6072
- interpretation: no statistically detectable paired success difference at alpha = 0.05

## Query efficiency among successful attacks
- overall mean queries among successes: 1.432
- label_only mean queries among successes: 1.412
- score_based mean queries among successes: 1.450

Interpretation: score_based was not more query-efficient in the exploratory remaining-pool run. Query efficiency among successes was very similar across conditions.

## Directional success
- label_only, deceptive: 13 / 139 = 9.353%
- label_only, truthful: 4 / 90 = 4.444%
- score_based, deceptive: 16 / 139 = 11.511%
- score_based, truthful: 4 / 90 = 4.444%

## Validity
- total attempts: 1319
- valid attempts: 922
- invalid attempts: 397
- overall validity rate: 69.901%
- invalid candidates queried: 0

## Top failed individual checks
- negation_consistency: 177 attempts, 13.419%
- numbers_consistency: 175 attempts, 13.268%
- sbert_similarity: 37 attempts, 2.805%
- dates_consistency: 25 attempts, 1.895%
- duplicate_candidate: 21 attempts, 1.592%
- not_identical: 9 attempts, 0.682%

## Relation to primary result
Primary main_160 result:
- label_only: 19 / 160 = 11.875%
- score_based: 15 / 160 = 9.375%
- label_only was descriptively higher, but not statistically detectable.

Exploratory remaining_229 result:
- label_only: 17 / 229 = 7.424%
- score_based: 20 / 229 = 8.734%
- score_based was descriptively higher, but not statistically detectable.

Together, the two held-out analyses do not provide reliable evidence that score-based feedback improves valid attack success rate over label-only feedback. The small descriptive differences change direction across subsets, suggesting that any effect of feedback granularity is weak or unstable under this strict semantic-preserving paraphrase attack protocol.

## Source files
Raw exploratory files:
- Scripts_code/outputs/attack_results_exploratory_local_llm_remaining_gen3_query3.csv
- Scripts_code/outputs/attack_attempts_exploratory_local_llm_remaining_gen3_query3.csv
- Scripts_code/outputs/attack_exploratory_meta_local_llm_remaining_gen3_query3.json

Exploratory analysis folder:
- Scripts_code/outputs/analysis_exploratory_remaining_gen3_query3/

Analysis summary:
- Scripts_code/outputs/analysis_exploratory_remaining_gen3_query3/summary_exploratory_remaining_gen3_query3.json
