#!/usr/bin/env python3
"""Fit the cascaded classifier-based generator under 5-fold cross-validation
for one retriever, and dump out-of-fold predictions + retrieval metrics +
efficiency numbers to results/.

Usage:
    python train.py --retriever bm25 --seed 42
    python train.py --retriever tfidf_lsa --seed 42
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.data_utils import load_corpus, load_qa, corpus_index_by_id
from src.retrieval import build_retriever
from src.retrieval_eval import evaluate_retriever
from src.efficiency import benchmark_retriever
from src.generator import extract_features, predict_extractive_span, FEATURE_NAMES

ROOT = Path(__file__).resolve().parent


def build_dataset(qa_items, retriever, docs_by_id, k=5):
    """Pre-compute retrieval + features once per item (retrieval isn't
    fit on QA labels, so this is leakage-free w.r.t. the CV below)."""
    rows = []
    for q in qa_items:
        retrieved = retriever.retrieve(q["question"], k=k)
        feats, top1_id = extract_features(q["question"], retrieved, docs_by_id)
        rows.append({
            "qid": q["qid"],
            "question": q["question"],
            "gold_answer_type": q["answer_type"],
            "gold_answer": q["gold_answer"],
            "gold_arxiv_id": q["gold_arxiv_id"],
            "features": feats,
            "top1_id": top1_id,
            "top1_text": docs_by_id[top1_id]["text"],
        })
    return rows


def run_cv(rows, seed=42, n_splits=5):
    X = np.array([r["features"] for r in rows])
    y_unanswerable = np.array([1 if r["gold_answer_type"] == "unanswerable" else 0 for r in rows])
    y_extractive = np.array([1 if r["gold_answer_type"] == "extractive" else 0 for r in rows])
    y_is_yes = np.array([1 if r["gold_answer"] == "Yes" else 0 for r in rows])
    is_answerable = (y_unanswerable == 0)
    is_yes_no = np.array([r["gold_answer_type"] == "yes_no" for r in rows])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    strata = np.array([r["gold_answer_type"] for r in rows])

    predictions = [None] * len(rows)

    for train_idx, test_idx in skf.split(X, strata):
        # Stage A: answerable vs unanswerable
        clf_a = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf_a.fit(X[train_idx], y_unanswerable[train_idx])
        proba_a = clf_a.predict_proba(X[test_idx])
        classes_a = list(clf_a.classes_)
        p_unanswerable = proba_a[:, classes_a.index(1)]
        pred_unanswerable = p_unanswerable >= 0.5

        # Stage B: trained only on truly-answerable TRAIN items
        train_answerable_idx = train_idx[is_answerable[train_idx]]
        clf_b = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf_b.fit(X[train_answerable_idx], y_extractive[train_answerable_idx])
        classes_b = list(clf_b.classes_)

        # Stage C: trained only on truly-yes/no TRAIN items
        train_yesno_idx = train_idx[is_yes_no[train_idx]]
        clf_c = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf_c.fit(X[train_yesno_idx], y_is_yes[train_yesno_idx])
        classes_c = list(clf_c.classes_)

        for local_i, global_i in enumerate(test_idx):
            r = rows[global_i]
            p_una = float(p_unanswerable[local_i])
            if pred_unanswerable[local_i]:
                predictions[global_i] = {
                    **r, "predicted_answer_type": "unanswerable",
                    "predicted_value": None, "confidence": p_una,
                }
                continue

            proba_b = clf_b.predict_proba(X[global_i:global_i + 1])[0]
            p_extractive = proba_b[classes_b.index(1)] if 1 in classes_b else 0.0
            pred_extractive = p_extractive >= 0.5
            p_answerable = 1.0 - p_una

            if pred_extractive:
                span = predict_extractive_span(r["top1_text"])
                conf = p_answerable * float(p_extractive)
                predictions[global_i] = {
                    **r, "predicted_answer_type": "extractive",
                    "predicted_value": span, "confidence": conf,
                }
            else:
                proba_c = clf_c.predict_proba(X[global_i:global_i + 1])[0]
                p_yes = proba_c[classes_c.index(1)] if 1 in classes_c else 0.5
                pred_yes = p_yes >= 0.5
                p_notextractive = 1.0 - float(p_extractive)
                conf = p_answerable * p_notextractive * (p_yes if pred_yes else (1 - p_yes))
                predictions[global_i] = {
                    **r, "predicted_answer_type": "yes_no",
                    "predicted_value": "Yes" if pred_yes else "No", "confidence": conf,
                }

    return predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", choices=["bm25", "tfidf_lsa"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()

    docs = load_corpus(ROOT / "data" / "corpus.json")
    qa = load_qa(ROOT / "data" / "qa_benchmark.json")
    docs_by_id = corpus_index_by_id(docs)

    retriever = build_retriever(args.retriever, docs)

    retrieval_metrics, per_query_ranks = evaluate_retriever(retriever, qa)
    eff = benchmark_retriever(retriever, [q["question"] for q in qa])

    rows = build_dataset(qa, retriever, docs_by_id, k=args.k)
    predictions = run_cv(rows, seed=args.seed, n_splits=args.n_splits)

    # strip large text fields before dumping
    clean_predictions = []
    for p in predictions:
        p2 = {k: v for k, v in p.items() if k not in ("top1_text", "features")}
        clean_predictions.append(p2)

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / f"records_{args.retriever}.json", "w") as f:
        json.dump(clean_predictions, f, indent=2)
    with open(results_dir / f"retrieval_metrics_{args.retriever}.json", "w") as f:
        json.dump({"metrics": retrieval_metrics, "per_query": per_query_ranks}, f, indent=2)
    with open(results_dir / f"efficiency_{args.retriever}.json", "w") as f:
        json.dump(eff, f, indent=2)

    print(f"[{args.retriever}] retrieval:", retrieval_metrics)
    print(f"[{args.retriever}] efficiency:", eff)
    print(f"[{args.retriever}] wrote {len(clean_predictions)} out-of-fold predictions")


if __name__ == "__main__":
    main()
