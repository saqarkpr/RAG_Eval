#!/usr/bin/env python3
"""Fit the cascaded classifier-based generator under stratified k-fold
cross-validation for one retriever, and dump out-of-fold predictions +
retrieval metrics + efficiency numbers to results/.

Defaults come from configs/default.yaml (the values behind
results/report.md); any CLI flag overrides the config.

Usage:
    python train.py --retriever bm25 --seed 42
    python train.py --retriever tfidf_lsa --seed 42
    python train.py --config my_config.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.config import DEFAULT_CONFIG_PATH, load_config, resolve_data_paths
from src.data_utils import corpus_index_by_id, load_corpus, load_qa
from src.efficiency import benchmark_retriever
from src.generator import extract_features, predict_extractive_span
from src.retrieval import RETRIEVERS, build_retriever
from src.retrieval_eval import evaluate_retriever

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


def _positive_proba(clf, X, default):
    """P(class == 1) for each row of X, or `default` when the classifier
    never saw a positive example (a single-class fold)."""
    classes = list(clf.classes_)
    if 1 not in classes:
        return np.full(X.shape[0], default)
    return clf.predict_proba(X)[:, classes.index(1)]


def _fit_stage(X, y):
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, y)
    return clf


def run_cv(rows, seed=42, n_splits=5):
    """Out-of-fold predictions for the three-stage cascade. One fold split
    is shared by all stages, so a Stage-A error propagates into B/C exactly
    as it would at inference time."""
    if len(rows) < n_splits:
        raise ValueError(f"need at least n_splits={n_splits} items, got {len(rows)}")
    X = np.array([r["features"] for r in rows])
    y_unanswerable = np.array([1 if r["gold_answer_type"] == "unanswerable" else 0 for r in rows])
    y_extractive = np.array([1 if r["gold_answer_type"] == "extractive" else 0 for r in rows])
    y_is_yes = np.array([1 if r["gold_answer"] == "Yes" else 0 for r in rows])
    is_answerable = y_unanswerable == 0
    is_yes_no = np.array([r["gold_answer_type"] == "yes_no" for r in rows])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    strata = np.array([r["gold_answer_type"] for r in rows])

    predictions = [None] * len(rows)

    for train_idx, test_idx in skf.split(X, strata):
        # Stage A: answerable vs unanswerable, on every training item
        clf_a = _fit_stage(X[train_idx], y_unanswerable[train_idx])
        # Stage B: yes/no vs extractive, trained only on truly-answerable TRAIN items
        train_answerable_idx = train_idx[is_answerable[train_idx]]
        clf_b = _fit_stage(X[train_answerable_idx], y_extractive[train_answerable_idx])
        # Stage C: yes vs no, trained only on truly-yes/no TRAIN items
        train_yesno_idx = train_idx[is_yes_no[train_idx]]
        clf_c = _fit_stage(X[train_yesno_idx], y_is_yes[train_yesno_idx])

        X_test = X[test_idx]
        p_unanswerable = _positive_proba(clf_a, X_test, default=0.0)
        p_extractive = _positive_proba(clf_b, X_test, default=0.0)
        p_yes = _positive_proba(clf_c, X_test, default=0.5)

        for local_i, global_i in enumerate(test_idx):
            r = rows[global_i]
            p_una = float(p_unanswerable[local_i])
            if p_una >= 0.5:
                predictions[global_i] = {
                    **r, "predicted_answer_type": "unanswerable",
                    "predicted_value": None, "confidence": p_una,
                }
                continue

            p_answerable = 1.0 - p_una
            p_ext = float(p_extractive[local_i])
            if p_ext >= 0.5:
                predictions[global_i] = {
                    **r, "predicted_answer_type": "extractive",
                    "predicted_value": predict_extractive_span(r["top1_text"]),
                    "confidence": p_answerable * p_ext,
                }
            else:
                py = float(p_yes[local_i])
                pred_yes = py >= 0.5
                predictions[global_i] = {
                    **r, "predicted_answer_type": "yes_no",
                    "predicted_value": "Yes" if pred_yes else "No",
                    "confidence": p_answerable * (1.0 - p_ext) * (py if pred_yes else 1.0 - py),
                }

    return predictions


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                    help="YAML config providing defaults for every flag below")
    ap.add_argument("--retriever", choices=sorted(RETRIEVERS), default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--k", type=int, default=None, help="evidence passages retrieved per question")
    ap.add_argument("--n-splits", type=int, default=None, help="stratified CV folds")
    ap.add_argument("--lsa-components", type=int, default=None,
                    help="truncated-SVD dimensionality for the tfidf_lsa retriever")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    retriever_name = args.retriever or cfg["retriever"]
    seed = args.seed if args.seed is not None else cfg["seed"]
    k = args.k if args.k is not None else cfg["top_k"]
    n_splits = args.n_splits if args.n_splits is not None else cfg["n_splits"]
    n_components = (
        args.lsa_components if args.lsa_components is not None
        else cfg["tfidf_lsa"]["n_components"]
    )

    corpus_path, qa_path = resolve_data_paths(cfg, ROOT)
    docs = load_corpus(corpus_path)
    qa = load_qa(qa_path)
    docs_by_id = corpus_index_by_id(docs)

    kwargs = {}
    if retriever_name == "tfidf_lsa":
        kwargs = {"n_components": n_components, "random_state": seed}
    retriever = build_retriever(retriever_name, docs, **kwargs)

    retrieval_metrics, per_query_ranks = evaluate_retriever(retriever, qa)
    eff = benchmark_retriever(retriever, [q["question"] for q in qa], k=k)

    rows = build_dataset(qa, retriever, docs_by_id, k=k)
    predictions = run_cv(rows, seed=seed, n_splits=n_splits)

    # strip large text fields before dumping
    clean_predictions = [
        {key: v for key, v in p.items() if key not in ("top1_text", "features")}
        for p in predictions
    ]

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"records_{retriever_name}.json", "w") as f:
        json.dump(clean_predictions, f, indent=2)
    with open(results_dir / f"retrieval_metrics_{retriever_name}.json", "w") as f:
        json.dump({"metrics": retrieval_metrics, "per_query": per_query_ranks}, f, indent=2)
    with open(results_dir / f"efficiency_{retriever_name}.json", "w") as f:
        json.dump(eff, f, indent=2)
    with open(results_dir / f"run_config_{retriever_name}.json", "w") as f:
        json.dump({"retriever": retriever_name, "seed": seed, "top_k": k, "n_splits": n_splits,
                   "lsa_n_components": n_components if retriever_name == "tfidf_lsa" else None},
                  f, indent=2)

    print(f"[{retriever_name}] retrieval:", retrieval_metrics)
    print(f"[{retriever_name}] efficiency:", eff)
    print(
        f"[{retriever_name}] wrote {len(clean_predictions)} out-of-fold predictions "
        f"to {results_dir}"
    )


if __name__ == "__main__":
    main()
