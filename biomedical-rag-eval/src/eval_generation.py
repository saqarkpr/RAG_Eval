"""
Shared generation-evaluation loop, used by both run_generation_eval.py
(classical baseline) and run_generation_eval_llm.py (real LLM). Both
scripts call `evaluate_generator` with a different `generator` object but
identical evidence-construction, metric, and plotting code -- so a
baseline-vs-LLM comparison is guaranteed apples-to-apples: same test
queries, same BM25 retriever, same three evidence conditions, same metric
implementations. Nothing about the evaluation methodology can silently
drift between the two scripts because there is only one copy of it.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import (
    bootstrap_ci,
    brier_score_multiclass,
    expected_calibration_error,
    reliability_bins,
    support_ratio,
)

CLASSES = ["maybe", "no", "yes"]  # fixed order shared by every generator's class_probs
CONDITIONS = ["no_context", "gold_context", "bm25_context"]


def _plot_reliability(centers, accs, confs, title, out_path):
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


def evaluate_generator(
    generator,
    test: list[dict],
    corpus_by_id: dict,
    bm25_retriever,
    results_dir: Path,
    figure_prefix: str = "reliability",
    progress_every: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run `generator` over `test` under all three evidence conditions.

    Returns (summary_df, per_query_df):
    - summary_df: one row per condition -- accuracy (+ bootstrapped 95%
      CI), ECE, Brier, mean confidence, mean support ratio, n.
    - per_query_df: one row per (condition, query) with the gold and
      predicted decision, full class_probs, support ratio, and which
      document was actually cited -- enough to diagnose *where* a
      generator goes wrong, not just its aggregate accuracy.

    Also writes one reliability-diagram PNG per condition under
    `results_dir/figures/{figure_prefix}_{condition}.png`.
    """
    summary_rows = []
    per_query_rows = []

    for condition in CONDITIONS:
        confidences, corrects, y_true_idx, prob_matrix, supports = [], [], [], [], []
        for i, q in enumerate(test):
            gold_doc = corpus_by_id[q["doc_id"]]
            if condition == "no_context":
                doc_id, ev_text, ev_sents = "none", "", []
            elif condition == "gold_context":
                doc_id, ev_text, ev_sents = q["doc_id"], gold_doc["text"], gold_doc["sentences"]
            else:  # bm25_context
                result = bm25_retriever.retrieve(q["question"], k=1)
                doc_id = result.doc_ids[0]
                retrieved_doc = corpus_by_id[doc_id]
                ev_text, ev_sents = retrieved_doc["text"], retrieved_doc["sentences"]

            out = generator.generate(q["question"], doc_id, ev_text, ev_sents)
            is_correct = out.decision == q["final_decision"]
            confidences.append(out.confidence)
            corrects.append(is_correct)
            y_true_idx.append(CLASSES.index(q["final_decision"]))
            row_probs = [out.class_probs.get(c, 0.0) for c in CLASSES]
            prob_matrix.append(row_probs)
            # Faithfulness is checked against the CITED SENTENCE, not the
            # full templated answer_text ("Yes. Evidence: \"...\" (source:
            # doc123)") -- the wrapper words ("Evidence", "source", the
            # doc id) are not claims the generator made and shouldn't
            # count toward or against its lexical groundedness.
            support = support_ratio(out.cited_sentence, ev_text) if ev_text else float("nan")
            if ev_text:
                supports.append(support)

            per_query_rows.append(
                {
                    "condition": condition,
                    "qid": q["qid"],
                    "gold_doc_id": q["doc_id"],
                    "cited_doc_id": doc_id,
                    "gold_decision": q["final_decision"],
                    "pred_decision": out.decision,
                    "correct": is_correct,
                    "confidence": out.confidence,
                    "p_maybe": row_probs[0],
                    "p_no": row_probs[1],
                    "p_yes": row_probs[2],
                    "support_ratio": support,
                    "cited_sentence": out.cited_sentence,
                }
            )

            if progress_every and (i + 1) % progress_every == 0:
                print(f"  [{condition}] {i + 1}/{len(test)}")

        confidences = np.array(confidences)
        corrects = np.array(corrects, dtype=float)
        prob_matrix = np.array(prob_matrix)
        y_true_idx = np.array(y_true_idx)

        # Defensive check: every generator's class_probs must be a real
        # probability vector over the three decision classes, or ECE/Brier
        # below would silently compute nonsense.
        if not np.allclose(prob_matrix.sum(axis=1), 1.0, atol=1e-3):
            bad = np.where(~np.isclose(prob_matrix.sum(axis=1), 1.0, atol=1e-3))[0]
            raise ValueError(
                f"[{condition}] class_probs do not sum to 1 for {len(bad)} "
                f"quer(y/ies) (first offending row sums to "
                f"{prob_matrix[bad[0]].sum():.4f}) -- generator bug."
            )
        if (prob_matrix < 0).any():
            raise ValueError(f"[{condition}] negative class_probs -- generator bug.")

        acc_mean, acc_lo, acc_hi = bootstrap_ci(corrects)
        ece = expected_calibration_error(confidences, corrects)
        brier = brier_score_multiclass(prob_matrix, y_true_idx, len(CLASSES))
        mean_support = float(np.nanmean(supports)) if supports else float("nan")

        centers, accs_bin, confs_bin, _ = reliability_bins(confidences, corrects)
        figures_dir = results_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        _plot_reliability(
            centers, accs_bin, confs_bin,
            title=f"Reliability: {condition}",
            out_path=figures_dir / f"{figure_prefix}_{condition}.png",
        )

        summary_rows.append(
            {
                "condition": condition,
                "accuracy": acc_mean,
                "accuracy_ci_lo": acc_lo,
                "accuracy_ci_hi": acc_hi,
                "ece": ece,
                "brier": brier,
                "mean_confidence": float(confidences.mean()),
                "mean_support_ratio": mean_support,
                "n": len(test),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(per_query_rows)
