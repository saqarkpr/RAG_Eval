"""
Build the classical-vs-LLM summary figure used in the README, from the
two already-committed result tables (results/tables/generation_results.csv
and generation_results_llm.csv). Re-run this any time either table
changes so the figure never drifts out of sync with the numbers in the
text.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CONDITIONS = ["no_context", "gold_context", "bm25_context"]
LABELS = {"no_context": "No context", "gold_context": "Gold context", "bm25_context": "BM25 context"}


def main():
    classical = pd.read_csv(RESULTS_DIR / "tables" / "generation_results.csv").set_index("condition")
    llm = pd.read_csv(RESULTS_DIR / "tables" / "generation_results_llm.csv").set_index("condition")

    x = np.arange(len(CONDITIONS))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # --- Panel 1: accuracy with 95% CI error bars ---
    ax = axes[0]
    c_acc = classical.loc[CONDITIONS, "accuracy"].values
    c_lo = classical.loc[CONDITIONS, "accuracy_ci_lo"].values
    c_hi = classical.loc[CONDITIONS, "accuracy_ci_hi"].values
    l_acc = llm.loc[CONDITIONS, "accuracy"].values
    l_lo = llm.loc[CONDITIONS, "accuracy_ci_lo"].values
    l_hi = llm.loc[CONDITIONS, "accuracy_ci_hi"].values

    ax.bar(
        x - width / 2, c_acc, width, yerr=[c_acc - c_lo, c_hi - c_acc],
        capsize=4, label="Classical (TF-IDF+LogReg)", color="#6b8fbf",
    )
    ax.bar(
        x + width / 2, l_acc, width, yerr=[l_acc - l_lo, l_hi - l_acc],
        capsize=4, label="Qwen2.5-1.5B-Instruct", color="#d98c4a",
    )
    ax.axhline(0.505, color="gray", linestyle="--", linewidth=1, label="Majority-class baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in CONDITIONS])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 0.75)
    ax.set_title("Decision accuracy (95% CI)")
    ax.legend(fontsize=7, loc="upper left")

    # --- Panel 2: confidence vs. accuracy gap (overconfidence) ---
    ax = axes[1]
    c_conf = classical.loc[CONDITIONS, "mean_confidence"].values
    l_conf = llm.loc[CONDITIONS, "mean_confidence"].values

    ax.plot(x, c_acc, "o-", color="#6b8fbf", label="Classical: accuracy")
    ax.plot(x, c_conf, "o--", color="#6b8fbf", alpha=0.6, label="Classical: confidence")
    ax.plot(x, l_acc, "s-", color="#d98c4a", label="Qwen: accuracy")
    ax.plot(x, l_conf, "s--", color="#d98c4a", alpha=0.6, label="Qwen: confidence")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in CONDITIONS])
    ax.set_ylabel("Value")
    ax.set_ylim(0, 1.0)
    ax.set_title("Confidence vs. accuracy gap\n(bigger gap = more overconfident)")
    ax.legend(fontsize=6.5, loc="upper left")

    fig.tight_layout()
    out_path = RESULTS_DIR / "figures" / "classical_vs_llm_summary.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
