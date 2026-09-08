#!/usr/bin/env python3
"""Compute trustworthy-evaluation metrics (calibration, hallucination rate,
evidence grounding, retrieval Recall@k/MRR, efficiency) from the
out-of-fold predictions written by train.py, for both retrievers, and
produce a comparison report + reliability diagrams under results/."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data_utils import load_corpus, load_qa, corpus_index_by_id
from src.retrieval import build_retriever
from src.generator import extract_features, fuzzy_match, FEATURE_NAMES, NEGATION_CUES

from src.evaluate_trust import (
    expected_calibration_error, brier_score, hallucination_rate,
    false_abstention_rate, evidence_grounding_rate,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def is_correct(rec):
    gt, pt = rec["gold_answer_type"], rec["predicted_answer_type"]
    if gt == "unanswerable":
        return pt == "unanswerable"
    if pt != gt:
        return False
    if gt == "yes_no":
        return rec["predicted_value"] == rec["gold_answer"]
    if gt == "extractive":
        return fuzzy_match(rec["predicted_value"] or "", rec["gold_answer"] or "")
    return False


def reliability_plot(confidences, corrects, out_path, title, n_bins=10):
    ece, bins = expected_calibration_error(confidences, corrects, n_bins=n_bins)
    xs = [(b["lo"] + b["hi"]) / 2 for b in bins if b["count"] > 0]
    accs = [b["acc"] for b in bins if b["count"] > 0]
    confs = [b["conf"] for b in bins if b["count"] > 0]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    ax.bar(xs, accs, width=0.08, alpha=0.7, label="observed accuracy", color="#4C78A8")
    ax.scatter(confs, accs, color="#F58518", zorder=5, label="bin (conf, acc)")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(f"{title}\nECE={ece:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return ece, bins


def summarize(retriever_name):
    with open(RESULTS / f"records_{retriever_name}.json") as f:
        records = json.load(f)
    with open(RESULTS / f"retrieval_metrics_{retriever_name}.json") as f:
        retrieval = json.load(f)["metrics"]
    with open(RESULTS / f"efficiency_{retriever_name}.json") as f:
        efficiency = json.load(f)

    corrects = np.array([1.0 if is_correct(r) else 0.0 for r in records])
    confidences = np.array([r["confidence"] for r in records])
    accuracy = float(corrects.mean())

    ece, bins = reliability_plot(
        confidences, corrects,
        RESULTS / f"reliability_{retriever_name}.png",
        f"Reliability diagram ({retriever_name})",
    )
    brier = brier_score(confidences, corrects)

    hrate, n_hall, n_una = hallucination_rate(records)
    farate, n_fa, n_ans = false_abstention_rate(records)
    ground_rate, n_ground, n_answered = evidence_grounding_rate(records)

    return {
        "retriever": retriever_name,
        "accuracy": accuracy,
        "ece": ece,
        "brier": brier,
        "hallucination_rate": hrate,
        "hallucination_count": f"{n_hall}/{n_una}",
        "false_abstention_rate": farate,
        "false_abstention_count": f"{n_fa}/{n_ans}",
        "evidence_grounding_rate": ground_rate,
        "evidence_grounding_count": f"{n_ground}/{n_answered}",
        "retrieval": retrieval,
        "efficiency": efficiency,
    }


def robustness_probe(seed=42):
    """Diagnostic (not a generalization estimate): train Stage-A/B/C on ALL
    77 items, then check whether negating a yes/no question's polarity
    (prepending 'Is it NOT the case that ...') flips the predicted yes/no
    value. Uses the BM25 retriever. Reports a consistency rate."""
    docs = load_corpus(ROOT / "data" / "corpus.json")
    qa = load_qa(ROOT / "data" / "qa_benchmark.json")
    docs_by_id = corpus_index_by_id(docs)
    retriever = build_retriever("bm25", docs)

    yesno_items = [q for q in qa if q["answer_type"] == "yes_no"]

    def featurize(question):
        retrieved = retriever.retrieve(question, k=5)
        feats, top1_id = extract_features(question, retrieved, docs_by_id)
        return feats

    X, y_is_yes = [], []
    for q in yesno_items:
        X.append(featurize(q["question"]))
        y_is_yes.append(1 if q["gold_answer"] == "Yes" else 0)
    X = np.array(X)
    y_is_yes = np.array(y_is_yes)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, y_is_yes)
    classes = list(clf.classes_)

    flips = 0
    for q in yesno_items:
        orig_feats = np.array([featurize(q["question"])])
        p_yes_orig = clf.predict_proba(orig_feats)[0][classes.index(1)]
        pred_orig = p_yes_orig >= 0.5

        stripped = q["question"].rstrip("?").strip()
        # drop a leading aux verb (Does/Is/Did/Has/Do) if present, then negate
        for aux in ["Does ", "Is ", "Did ", "Has ", "Do "]:
            if stripped.startswith(aux):
                stripped = stripped[len(aux):]
                break
        negated_q = f"Is it NOT the case that {stripped[0].lower()}{stripped[1:]}?"

        neg_feats = np.array([featurize(negated_q)])
        p_yes_neg = clf.predict_proba(neg_feats)[0][classes.index(1)]
        pred_neg = p_yes_neg >= 0.5

        if pred_orig != pred_neg:
            flips += 1

    flip_rate = flips / len(yesno_items)
    return {
        "n_probed": len(yesno_items),
        "n_flipped_on_negation": flips,
        "negation_flip_rate": flip_rate,
        "note": "Diagnostic probe trained on all 77 items (not held out); "
                "measures whether the yes/no classifier's decision flips "
                "when a question's logical polarity is negated. A LOGICALLY "
                "SOUND classifier should flip on (nearly) every item, since "
                "negating the question's polarity negates the correct "
                "answer; a low flip rate means the classifier is tracking "
                "surface lexical cues rather than semantic polarity.",
    }


def write_report(summaries, robustness):
    lines = ["# Results summary\n"]
    lines.append("| Retriever | Recall@1 | Recall@3 | Recall@5 | MRR | QA acc. | ECE | Brier | Hallucination rate | False-abstention rate | Evidence-grounding rate |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in summaries:
        r = s["retrieval"]
        lines.append(
            f"| {s['retriever']} | {r['recall@1']:.3f} | {r['recall@3']:.3f} | "
            f"{r['recall@5']:.3f} | {r['mrr']:.3f} | {s['accuracy']:.3f} | "
            f"{s['ece']:.3f} | {s['brier']:.3f} | "
            f"{s['hallucination_rate']:.3f} ({s['hallucination_count']}) | "
            f"{s['false_abstention_rate']:.3f} ({s['false_abstention_count']}) | "
            f"{s['evidence_grounding_rate']:.3f} ({s['evidence_grounding_count']}) |"
        )
    lines.append("\n## Efficiency (CPU-only)\n")
    lines.append("| Retriever | Build time (s) | Mean latency (ms) | P95 latency (ms) | Approx. memory (KB) |")
    lines.append("|---|---|---|---|---|")
    for s in summaries:
        e = s["efficiency"]
        lines.append(
            f"| {s['retriever']} | {e['build_time_s']:.4f} | {e['mean_latency_ms']:.3f} | "
            f"{e['p95_latency_ms']:.3f} | {e['approx_memory_kb']:.1f} |"
        )
    lines.append("\n## Robustness probe (negation flip test, BM25 features, full-data model)\n")
    lines.append(f"- Questions probed: {robustness['n_probed']}")
    lines.append(f"- Flipped prediction on negated paraphrase: {robustness['n_flipped_on_negation']}")
    lines.append(f"- Negation flip rate: {robustness['negation_flip_rate']:.3f} "
                 f"(higher = more logically consistent; 1.0 = always flips as it should)")
    report = "\n".join(lines) + "\n"
    with open(RESULTS / "report.md", "w") as f:
        f.write(report)
    return report


def main():
    summaries = [summarize("bm25"), summarize("tfidf_lsa")]
    with open(RESULTS / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    robustness = robustness_probe()
    with open(RESULTS / "robustness.json", "w") as f:
        json.dump(robustness, f, indent=2)
    report = write_report(summaries, robustness)
    print(report)


if __name__ == "__main__":
    main()
