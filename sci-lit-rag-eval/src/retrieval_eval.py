"""Retrieval-only evaluation: Recall@k and MRR against gold arXiv IDs,
computed over every *answerable* question (unanswerable questions have no
gold document by construction)."""
import numpy as np


def evaluate_retriever(retriever, qa_items, ks=(1, 3, 5)):
    answerable = [q for q in qa_items if q["answer_type"] != "unanswerable"]
    max_k = max(ks)
    recall_hits = {k: 0 for k in ks}
    reciprocal_ranks = []
    per_query_ranks = []

    for q in answerable:
        gold = q["gold_arxiv_id"]
        ranked = retriever.retrieve(q["question"], k=max_k)
        ranked_ids = [rid for rid, _ in ranked]
        if gold in ranked_ids:
            rank = ranked_ids.index(gold) + 1
        else:
            rank = None
        per_query_ranks.append({"qid": q["qid"], "gold": gold, "rank": rank})
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        for k in ks:
            if rank is not None and rank <= k:
                recall_hits[k] += 1

    n = len(answerable)
    metrics = {f"recall@{k}": recall_hits[k] / n for k in ks}
    metrics["mrr"] = float(np.mean(reciprocal_ranks))
    metrics["n_queries"] = n
    return metrics, per_query_ranks
