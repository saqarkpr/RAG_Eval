"""
Evaluate answer generation under three evidence conditions to isolate what
retrieval quality buys you (and costs you) end-to-end:

  - `no_context`  : classifier sees only the question (no retrieval at all)
  - `gold_context`: classifier sees the question's own true source abstract
                    (upper bound -- perfect retrieval)
  - `bm25_context`: classifier sees whatever BM25's top-1 result was
                    (realistic pipeline, inherits BM25's retrieval errors)

For each condition we report: decision accuracy vs. the expert
`final_decision` label, Expected Calibration Error + Brier score (is the
model's confidence trustworthy?), and mean lexical support ratio between
the generated answer and the cited evidence (a hallucination-risk proxy).
A reliability diagram is saved per condition.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.generator import ExtractiveClassifierGenerator
from src.metrics import (
    accuracy,
    brier_score_multiclass,
    expected_calibration_error,
    reliability_bins,
    support_ratio,
)
from src.retrieval import BM25Retriever

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CLASSES = ["maybe", "no", "yes"]  # sorted order matches sklearn's classes_


def load_jsonl(path):
    return [json.loads(line) for line in open(path)]


def plot_reliability(centers, accs, confs, counts, title, out_path):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    mask = ~np.isnan(accs)
    ax.bar(centers[mask], accs[mask], width=0.08, alpha=0.7, label="empirical accuracy")
    ax.scatter(confs[mask], accs[mask], color="darkred", zorder=5, s=20)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    corpus = load_jsonl(DATA_DIR / "corpus.jsonl")
    train = load_jsonl(DATA_DIR / "train.jsonl")
    test = load_jsonl(DATA_DIR / "test.jsonl")
    corpus_by_id = {d["doc_id"]: d for d in corpus}

    doc_ids = [d["doc_id"] for d in corpus]
    texts = [d["text"] for d in corpus]
    bm25 = BM25Retriever(doc_ids, texts)

    # Train the classifier-generator using each training query's OWN
    # (gold) abstract as evidence -- at train time we always have the
    # correct evidence; the point of the eval below is what happens at
    # test time when evidence instead comes from a retriever.
    gen = ExtractiveClassifierGenerator()
    train_questions = [q["question"] for q in train]
    train_evidence = [corpus_by_id[q["doc_id"]]["text"] for q in train]
    train_labels = [q["final_decision"] for q in train]
    gen.fit(train_questions, train_evidence, train_labels)

    conditions = ["no_context", "gold_context", "bm25_context"]
    summary_rows = []

    for condition in conditions:
        confidences, corrects, y_true_idx, prob_matrix, supports = [], [], [], [], []
        for q in test:
            gold_doc = corpus_by_id[q["doc_id"]]
            if condition == "no_context":
                doc_id, ev_text, ev_sents = "none", "", []
            elif condition == "gold_context":
                doc_id, ev_text, ev_sents = q["doc_id"], gold_doc["text"], gold_doc["sentences"]
            else:  # bm25_context
                result = bm25.retrieve(q["question"], k=1)
                doc_id = result.doc_ids[0]
                retrieved_doc = corpus_by_id[doc_id]
                ev_text, ev_sents = retrieved_doc["text"], retrieved_doc["sentences"]

            out = gen.generate(q["question"], doc_id, ev_text, ev_sents)
            is_correct = out.decision == q["final_decision"]
            confidences.append(out.confidence)
            corrects.append(is_correct)
            y_true_idx.append(CLASSES.index(q["final_decision"]))
            prob_matrix.append([out.class_probs.get(c, 0.0) for c in CLASSES])
            # faithfulness is only meaningful when there's evidence to be
            # faithful to
            if ev_text:
                supports.append(support_ratio(out.answer_text, ev_text))

        confidences = np.array(confidences)
        corrects = np.array(corrects, dtype=float)
        prob_matrix = np.array(prob_matrix)
        y_true_idx = np.array(y_true_idx)

        acc = accuracy(corrects.tolist(), [1.0] * len(corrects))
        ece = expected_calibration_error(confidences, corrects)
        brier = brier_score_multiclass(prob_matrix, y_true_idx, len(CLASSES))
        mean_support = float(np.mean(supports)) if supports else float("nan")

        centers, accs_bin, confs_bin, counts = reliability_bins(confidences, corrects)
        RESULTS_DIR.joinpath("figures").mkdir(parents=True, exist_ok=True)
        plot_reliability(
            centers,
            accs_bin,
            confs_bin,
            counts,
            title=f"Reliability: {condition}",
            out_path=RESULTS_DIR / "figures" / f"reliability_{condition}.png",
        )

        summary_rows.append(
            {
                "condition": condition,
                "accuracy": acc,
                "ece": ece,
                "brier": brier,
                "mean_confidence": float(confidences.mean()),
                "mean_support_ratio": mean_support,
                "n": len(test),
            }
        )

    df = pd.DataFrame(summary_rows)
    RESULTS_DIR.joinpath("tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "tables" / "generation_results.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
