#!/usr/bin/env python3
"""Compute trustworthy-evaluation metrics (calibration, hallucination rate,
false-abstention rate, evidence grounding, retrieval Recall@k/MRR,
efficiency) from the out-of-fold predictions written by train.py, for
every retriever that has been trained, run the negation-robustness probe,
and produce a comparison report + reliability diagrams under results/.

Usage:
    python evaluate.py                       # after train.py for both retrievers
    python evaluate.py --retrievers bm25     # only what has been trained
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from src.config import DEFAULT_CONFIG_PATH, load_config, resolve_data_paths  # noqa: E402
from src.data_utils import corpus_index_by_id, load_corpus, load_qa  # noqa: E402
from src.evaluate_trust import (  # noqa: E402
    brier_score,
    evidence_grounding_rate,
    expected_calibration_error,
    false_abstention_rate,
    hallucination_rate,
)
from src.generator import extract_features, fuzzy_match  # noqa: E402
from src.retrieval import RETRIEVERS, build_retriever  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# Leading auxiliary verbs stripped before a yes/no question is negated in
# the robustness probe ("Does X do Y?" -> "Is it NOT the case that X does Y?").
AUX_PREFIXES = ("Does ", "Is ", "Did ", "Has ", "Do ", "Are ", "Was ", "Were ", "Can ", "Could ")


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


def summarize(retriever_name, results_dir=RESULTS, n_bins=10):
    with open(results_dir / f"records_{retriever_name}.json") as f:
        records = json.load(f)
    with open(results_dir / f"retrieval_metrics_{retriever_name}.json") as f:
        retrieval = json.load(f)["metrics"]
    with open(results_dir / f"efficiency_{retriever_name}.json") as f:
        efficiency = json.load(f)

    corrects = np.array([1.0 if is_correct(r) else 0.0 for r in records])
    confidences = np.array([r["confidence"] for r in records])
    accuracy = float(corrects.mean())

    ece, _ = reliability_plot(
        confidences, corrects,
        results_dir / f"reliability_{retriever_name}.png",
        f"Reliability diagram ({retriever_name})",
        n_bins=n_bins,
    )
    brier = brier_score(confidences, corrects)

    hrate, n_hall, n_una = hallucination_rate(records)
    farate, n_fa, n_ans = false_abstention_rate(records)
    ground_rate, n_ground, n_answered = evidence_grounding_rate(records)

    return {
        "retriever": retriever_name,
        "n_items": len(records),
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


def negate_question(question: str) -> str:
    """'Does X improve Y?' -> 'Is it NOT the case that X improve Y?'. The
    leading auxiliary is dropped (not conjugated), which is deliberate: the
    probe asks whether the classifier tracks polarity, so the paraphrase
    should keep every content word identical."""
    stripped = question.strip().rstrip("?").strip()
    for aux in AUX_PREFIXES:
        if stripped.startswith(aux):
            stripped = stripped[len(aux):]
            break
    if not stripped:
        return question
    return f"Is it NOT the case that {stripped[0].lower()}{stripped[1:]}?"


def robustness_probe(cfg, retriever_name="bm25"):
    """Diagnostic (not a generalization estimate): fit the Stage-C yes/no
    classifier on ALL yes/no items (no held-out split), then check whether
    negating a question's polarity (see negate_question) flips the
    predicted yes/no value. Reports the flip rate; a logically sound
    classifier should flip on nearly every item."""
    corpus_path, qa_path = resolve_data_paths(cfg, ROOT)
    docs = load_corpus(corpus_path)
    qa = load_qa(qa_path)
    docs_by_id = corpus_index_by_id(docs)
    kwargs = {}
    if retriever_name == "tfidf_lsa":
        kwargs = {"n_components": cfg["tfidf_lsa"]["n_components"], "random_state": cfg["seed"]}
    retriever = build_retriever(retriever_name, docs, **kwargs)
    k = cfg["top_k"]

    yesno_items = [q for q in qa if q["answer_type"] == "yes_no"]
    if not yesno_items:
        return {"n_probed": 0, "n_flipped_on_negation": 0, "negation_flip_rate": None,
                "retriever": retriever_name, "note": "no yes/no items to probe"}

    def featurize(question):
        retrieved = retriever.retrieve(question, k=k)
        feats, _ = extract_features(question, retrieved, docs_by_id)
        return feats

    X = np.array([featurize(q["question"]) for q in yesno_items])
    y_is_yes = np.array([1 if q["gold_answer"] == "Yes" else 0 for q in yesno_items])

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, y_is_yes)
    classes = list(clf.classes_)
    if 1 not in classes:  # every gold answer is "No": nothing to flip
        pred_yes_orig = np.zeros(len(yesno_items), dtype=bool)
        pred_yes_neg = pred_yes_orig
    else:
        pred_yes_orig = clf.predict_proba(X)[:, classes.index(1)] >= 0.5
        X_neg = np.array([featurize(negate_question(q["question"])) for q in yesno_items])
        pred_yes_neg = clf.predict_proba(X_neg)[:, classes.index(1)] >= 0.5

    flips = int(np.sum(pred_yes_orig != pred_yes_neg))
    return {
        "retriever": retriever_name,
        "n_probed": len(yesno_items),
        "n_flipped_on_negation": flips,
        "negation_flip_rate": flips / len(yesno_items),
        "note": f"Diagnostic probe: the Stage-C yes/no classifier is fit on all "
                f"{len(yesno_items)} yes/no items (not held out) using "
                f"{retriever_name} retrieval features, then each question is "
                "negated ('Is it NOT the case that ...?') and re-scored. It "
                "measures whether the classifier's decision flips when a "
                "question's logical polarity is negated. A LOGICALLY SOUND "
                "classifier should flip on (nearly) every item, since negating "
                "the question negates the correct answer; a low flip rate means "
                "the classifier is tracking surface lexical cues rather than "
                "semantic polarity.",
    }


def write_report(summaries, robustness, results_dir=RESULTS):
    lines = ["# Results summary\n"]
    lines.append(
        "| Retriever | Recall@1 | Recall@3 | Recall@5 | MRR | QA acc. | ECE | Brier | "
        "Hallucination rate | False-abstention rate | Evidence-grounding rate |"
    )
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
    lines.append(
        "| Retriever | Build time (s) | Mean latency (ms) | P95 latency (ms) "
        "| Approx. memory (KB) |"
    )
    lines.append("|---|---|---|---|---|")
    for s in summaries:
        e = s["efficiency"]
        lines.append(
            f"| {s['retriever']} | {e['build_time_s']:.4f} | {e['mean_latency_ms']:.3f} | "
            f"{e['p95_latency_ms']:.3f} | {e['approx_memory_kb']:.1f} |"
        )
    lines.append(
        f"\n## Robustness probe (negation flip test, {robustness['retriever']} features, "
        "full-data model)\n"
    )
    lines.append(f"- Questions probed: {robustness['n_probed']}")
    lines.append(
        f"- Flipped prediction on negated paraphrase: {robustness['n_flipped_on_negation']}"
    )
    rate = robustness["negation_flip_rate"]
    rate_str = f"{rate:.3f}" if rate is not None else "n/a"
    lines.append(f"- Negation flip rate: {rate_str} "
                 "(higher = more logically consistent; 1.0 = always flips as it should)")
    report = "\n".join(lines) + "\n"
    with open(results_dir / "report.md", "w") as f:
        f.write(report)
    return report


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--results-dir", type=Path, default=RESULTS)
    ap.add_argument("--retrievers", nargs="+", choices=sorted(RETRIEVERS), default=None,
                    help="retrievers to summarize (default: every one with records_*.json)")
    ap.add_argument("--probe-retriever", choices=sorted(RETRIEVERS), default="bm25",
                    help="retriever whose features the negation probe uses")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    results_dir = args.results_dir

    names = args.retrievers or [
        n for n in RETRIEVERS if (results_dir / f"records_{n}.json").exists()
    ]
    if not names:
        raise FileNotFoundError(
            f"no records_*.json in {results_dir}; run train.py (or scripts/train.sh) first"
        )
    summaries = [summarize(n, results_dir, n_bins=cfg["ece_bins"]) for n in names]
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    robustness = robustness_probe(cfg, retriever_name=args.probe_retriever)
    with open(results_dir / "robustness.json", "w") as f:
        json.dump(robustness, f, indent=2)
    print(write_report(summaries, robustness, results_dir))


if __name__ == "__main__":
    main()
