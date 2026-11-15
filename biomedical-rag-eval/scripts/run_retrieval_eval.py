"""
Evaluate BM25 vs. TF-IDF+LSA retrieval on the test split: Recall@{1,3,5,10},
MRR, and mean per-query latency. Results are bootstrapped (1000 resamples)
to report 95% CIs, since a single point estimate on 200 test queries is not
enough to claim one retriever beats another (report variance, not just a
mean).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import DATA_DIR, RESULTS_DIR, load_processed  # noqa: E402
from src.metrics import bootstrap_ci, recall_at_k, reciprocal_rank  # noqa: E402
from src.retrieval import BM25Retriever, LsaRetriever  # noqa: E402

KS = [1, 3, 5, 10]


def evaluate_retriever(name, retriever, test, ks=KS):
    """Return (summary rows, per-query rows) for one retriever."""
    recalls = {k: [] for k in ks}
    rrs, latencies, per_query = [], [], []
    for q in test:
        result = retriever.retrieve(q["question"], k=max(ks))
        for k in ks:
            recalls[k].append(recall_at_k(q["doc_id"], result.doc_ids, k))
        rrs.append(reciprocal_rank(q["doc_id"], result.doc_ids))
        latencies.append(result.latency_s)
        per_query.append(
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
    rows = []
    for k in ks:
        mean, lo, hi = bootstrap_ci(recalls[k])
        rows.append(
            {"retriever": name, "metric": f"recall@{k}", "mean": mean, "ci_lo": lo, "ci_hi": hi}
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
    return rows, per_query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--lsa-components", type=int, default=128)
    args = parser.parse_args()

    corpus, _, test = load_processed("test", data_dir=args.data_dir)
    doc_ids = [d["doc_id"] for d in corpus]
    texts = [d["text"] for d in corpus]

    t0 = time.perf_counter()
    bm25 = BM25Retriever(doc_ids, texts)
    bm25_index_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    lsa = LsaRetriever(doc_ids, texts, n_components=args.lsa_components)
    lsa_index_time = time.perf_counter() - t0

    rows, per_query_records = [], []
    for name, retriever in {"bm25": bm25, "tfidf_lsa": lsa}.items():
        r, pq = evaluate_retriever(name, retriever, test)
        rows.extend(r)
        per_query_records.extend(pq)

    for name, build_s in (("bm25", bm25_index_time), ("tfidf_lsa", lsa_index_time)):
        rows.append(
            {
                "retriever": name,
                "metric": "index_build_s",
                "mean": build_s,
                "ci_lo": None,
                "ci_hi": None,
            }
        )

    tables = args.results_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(tables / "retrieval_results.csv", index=False)
    pd.DataFrame(per_query_records).to_csv(tables / "retrieval_per_query.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
