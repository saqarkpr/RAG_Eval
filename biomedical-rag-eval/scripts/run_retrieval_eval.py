"""
Evaluate BM25 vs. TF-IDF+LSA retrieval on the test split: Recall@{1,3,5,10},
MRR, and mean per-query latency. Results are bootstrapped (1000 resamples)
to report 95% CIs, since a single point estimate on 200 test queries is not
enough to claim one retriever beats another (cf. portfolio convention:
report variance, not just a mean).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics import recall_at_k, reciprocal_rank
from src.retrieval import BM25Retriever, LsaRetriever

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_jsonl(path):
    return [json.loads(line) for line in open(path)]


def bootstrap_ci(values, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    boots = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(np.mean(values)), float(lo), float(hi)


def main():
    corpus = load_jsonl(DATA_DIR / "corpus.jsonl")
    test = load_jsonl(DATA_DIR / "test.jsonl")

    doc_ids = [d["doc_id"] for d in corpus]
    texts = [d["text"] for d in corpus]

    t0 = time.perf_counter()
    bm25 = BM25Retriever(doc_ids, texts)
    bm25_index_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    lsa = LsaRetriever(doc_ids, texts, n_components=128)
    lsa_index_time = time.perf_counter() - t0

    retrievers = {"bm25": bm25, "tfidf_lsa": lsa}
    ks = [1, 3, 5, 10]
    rows = []
    per_query_records = []

    for name, retriever in retrievers.items():
        recalls = {k: [] for k in ks}
        rrs = []
        latencies = []
        for q in test:
            result = retriever.retrieve(q["question"], k=max(ks))
            for k in ks:
                recalls[k].append(recall_at_k(q["doc_id"], result.doc_ids, k))
            rrs.append(reciprocal_rank(q["doc_id"], result.doc_ids))
            latencies.append(result.latency_s)
            per_query_records.append(
                {
                    "retriever": name,
                    "qid": q["qid"],
                    "gold_doc_id": q["doc_id"],
                    "rank_of_gold": (
                        result.doc_ids.index(q["doc_id"]) + 1
                        if q["doc_id"] in result.doc_ids
                        else None
                    ),
                }
            )
        for k in ks:
            mean, lo, hi = bootstrap_ci(recalls[k])
            rows.append(
                {
                    "retriever": name,
                    "metric": f"recall@{k}",
                    "mean": mean,
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )
        mean, lo, hi = bootstrap_ci(rrs)
        rows.append({"retriever": name, "metric": "mrr", "mean": mean, "ci_lo": lo, "ci_hi": hi})
        rows.append(
            {
                "retriever": name,
                "metric": "mean_latency_ms",
                "mean": 1000 * float(np.mean(latencies)),
                "ci_lo": None,
                "ci_hi": None,
            }
        )

    rows.append(
        {"retriever": "bm25", "metric": "index_build_s", "mean": bm25_index_time, "ci_lo": None, "ci_hi": None}
    )
    rows.append(
        {"retriever": "tfidf_lsa", "metric": "index_build_s", "mean": lsa_index_time, "ci_lo": None, "ci_hi": None}
    )

    df = pd.DataFrame(rows)
    RESULTS_DIR.joinpath("tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "tables" / "retrieval_results.csv", index=False)
    pd.DataFrame(per_query_records).to_csv(
        RESULTS_DIR / "tables" / "retrieval_per_query.csv", index=False
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
