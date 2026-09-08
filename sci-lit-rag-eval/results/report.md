# Results summary

| Retriever | Recall@1 | Recall@3 | Recall@5 | MRR | QA acc. | ECE | Brier | Hallucination rate | False-abstention rate | Evidence-grounding rate |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.984 | 0.984 | 0.984 | 0.984 | 0.571 | 0.173 | 0.268 | 0.125 (2/16) | 0.180 (11/61) | 1.000 (50/50) |
| tfidf_lsa | 0.869 | 0.967 | 0.984 | 0.919 | 0.494 | 0.162 | 0.262 | 0.312 (5/16) | 0.295 (18/61) | 0.907 (39/43) |

## Efficiency (CPU-only)

| Retriever | Build time (s) | Mean latency (ms) | P95 latency (ms) | Approx. memory (KB) |
|---|---|---|---|---|
| bm25 | 0.0029 | 0.130 | 0.188 | 384.5 |
| tfidf_lsa | 0.0275 | 0.667 | 0.726 | 197.6 |

## Robustness probe (negation flip test, BM25 features, full-data model)

- Questions probed: 34
- Flipped prediction on negated paraphrase: 13
- Negation flip rate: 0.382 (higher = more logically consistent; 1.0 = always flips as it should)
