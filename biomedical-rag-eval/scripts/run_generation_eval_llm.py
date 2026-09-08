"""
Same evaluation as run_generation_eval.py (no_context / gold_context /
bm25_context, decision accuracy + ECE + Brier + support ratio), but with
`HFLocalLLMGenerator` in place of the classical baseline. Run on Colab
(GPU strongly recommended; the model is scored 3x per query for the
decision distribution plus one short generation call per query, so CPU
will be slow):

    pip install torch transformers accelerate
    python scripts/run_generation_eval_llm.py --n-test 200

Use --n-test to run on a subset first (e.g. 20) to sanity-check the
pipeline before committing to a full run.

Writes results/tables/generation_results_llm.csv and
results/figures/reliability_llm_{condition}.png, in the same format as
the classical baseline's outputs, so they can be compared side by side
in the README.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.generator import HFLocalLLMGenerator
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
CLASSES = ["maybe", "no", "yes"]


def load_jsonl(path):
    return [json.loads(line) for line in open(path)]


def plot_reliability(centers, accs, confs, title, out_path):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=200, help="test queries to evaluate")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    corpus = load_jsonl(DATA_DIR / "corpus.jsonl")
    test = load_jsonl(DATA_DIR / "test.jsonl")[: args.n_test]
    corpus_by_id = {d["doc_id"]: d for d in corpus}

    doc_ids = [d["doc_id"] for d in corpus]
    texts = [d["text"] for d in corpus]
    bm25 = BM25Retriever(doc_ids, texts)

    print(f"Loading {args.model} ...")
    gen = HFLocalLLMGenerator(model_name=args.model)
    print(f"Loaded on device={gen.device}. Evaluating {len(test)} test queries.")

    conditions = ["no_context", "gold_context", "bm25_context"]
    summary_rows = []

    for condition in conditions:
        confidences, corrects, y_true_idx, prob_matrix, supports = [], [], [], [], []
        t0 = time.perf_counter()
        for i, q in enumerate(test):
            gold_doc = corpus_by_id[q["doc_id"]]
            if condition == "no_context":
                doc_id, ev_text, ev_sents = "none", "", []
            elif condition == "gold_context":
                doc_id, ev_text, ev_sents = q["doc_id"], gold_doc["text"], gold_doc["sentences"]
            else:
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
            if ev_text:
                supports.append(support_ratio(out.answer_text, ev_text))
            if (i + 1) % 25 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  [{condition}] {i + 1}/{len(test)} ({elapsed:.1f}s elapsed)")

        confidences = np.array(confidences)
        corrects = np.array(corrects, dtype=float)
        prob_matrix = np.array(prob_matrix)
        y_true_idx = np.array(y_true_idx)

        acc = accuracy(corrects.tolist(), [1.0] * len(corrects))
        ece = expected_calibration_error(confidences, corrects)
        brier = brier_score_multiclass(prob_matrix, y_true_idx, len(CLASSES))
        mean_support = float(np.mean(supports)) if supports else float("nan")

        centers, accs_bin, confs_bin, _ = reliability_bins(confidences, corrects)
        RESULTS_DIR.joinpath("figures").mkdir(parents=True, exist_ok=True)
        plot_reliability(
            centers, accs_bin, confs_bin,
            title=f"Reliability (LLM): {condition}",
            out_path=RESULTS_DIR / "figures" / f"reliability_llm_{condition}.png",
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
                "model": args.model,
            }
        )
        print(f"[{condition}] acc={acc:.3f} ece={ece:.3f} brier={brier:.3f}")

    df = pd.DataFrame(summary_rows)
    RESULTS_DIR.joinpath("tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "tables" / "generation_results_llm.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
