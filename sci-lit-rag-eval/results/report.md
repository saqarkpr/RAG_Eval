# Results summary

| Retriever | Recall@1 | Recall@3 | Recall@5 | MRR | QA acc. | ECE | Brier | Hallucination rate | False-abstention rate | Evidence-grounding rate |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.984 | 0.984 | 0.984 | 0.984 | 0.571 | 0.173 | 0.268 | 0.125 (2/16) | 0.180 (11/61) | 1.000 (50/50) |
| tfidf_lsa | 0.869 | 0.967 | 0.984 | 0.919 | 0.494 | 0.162 | 0.262 | 0.312 (5/16) | 0.295 (18/61) | 0.907 (39/43) |

## Efficiency (CPU-only)

| Retriever | Build time (s) | Mean latency (ms) | P95 latency (ms) | Approx. memory (KB) |
|---|---|---|---|---|
| bm25 | 0.0024 | 0.122 | 0.198 | 361.0 |
| tfidf_lsa | 0.0179 | 0.762 | 1.087 | 197.6 |

## Robustness probe (negation flip test, bm25 features, full-data model)

- Questions probed: 34
- Flipped prediction on negated paraphrase: 13
- Negation flip rate: 0.382 (higher = more logically consistent; 1.0 = always flips as it should)
