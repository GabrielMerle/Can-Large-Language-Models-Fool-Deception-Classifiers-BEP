# Results — Unified Llama-3.3-70B Experiment

Each of 389 correctly classified held-out Hippocorpus statements (170
truthful, 219 deceptive) was attacked once under label-only feedback and once
under score-based feedback, giving 778 paired attack runs. The experiment used
a fixed DistilBERT victim classifier, a fixed
`meta-llama/Llama-3.3-70B-Instruct-Turbo` paraphraser served through the
DeepInfra API, a strict validity gate, and a `gen5_query5` budget.

## Headline results

| Feedback condition | Successes | N | Valid ASR |
| --- | ---: | ---: | ---: |
| Label-only | 97 | 389 | 24.9% |
| Score-based | 90 | 389 | 23.1% |

Paired outcomes:

- both success: 77
- label-only only: 20
- score-based only: 13
- neither success: 279

Exact McNemar test: `p = .296`.

There was no statistically detectable attack-success advantage for
score-based feedback over label-only feedback.

## Query efficiency among successful attacks

- Label-only mean classifier queries: 1.52.
- Score-based mean classifier queries: 1.58.
- Wilcoxon `p = .634`.

Score-based feedback did not improve query efficiency among successful
attacks.

## Validity

- Total generated candidates: 3,285.
- Valid candidates: 2,573.
- Validity rate: 78.3%.
- Invalid candidates queried: 0.

## Class asymmetry

Attack success was concentrated mainly in deceptive-predicted examples:

- deceptive-predicted: 42.9% label-only, 39.7% score-based;
- truthful-predicted: 1.8% in both conditions.

## Main conclusion

A strong 70B paraphraser can fool the fixed deception classifier in about one
quarter of strictly valid cases, but exposing classifier confidence did not
improve attack success or query efficiency under the fixed low-budget
protocol.
