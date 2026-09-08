"""Trustworthy-evaluation metrics: calibration (ECE + reliability diagram),
hallucination rate on unanswerable questions, and an evidence-grounding
("faithfulness") proxy."""
import numpy as np


def expected_calibration_error(confidences, corrects, n_bins=10):
    confidences = np.asarray(confidences, dtype=float)
    corrects = np.asarray(corrects, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_stats = []
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        count = int(mask.sum())
        if count == 0:
            bin_stats.append({"lo": lo, "hi": hi, "count": 0, "acc": None, "conf": None})
            continue
        acc = corrects[mask].mean()
        conf = confidences[mask].mean()
        ece += (count / n) * abs(acc - conf)
        bin_stats.append({"lo": lo, "hi": hi, "count": count, "acc": float(acc), "conf": float(conf)})
    return float(ece), bin_stats


def brier_score(confidences, corrects):
    confidences = np.asarray(confidences, dtype=float)
    corrects = np.asarray(corrects, dtype=float)
    return float(np.mean((confidences - corrects) ** 2))


def hallucination_rate(records):
    """records: list of dicts with gold_answer_type and predicted_answer_type.
    Hallucination = system answers (does not predict 'unanswerable') on a
    question whose gold label IS 'unanswerable'."""
    gold_unanswerable = [r for r in records if r["gold_answer_type"] == "unanswerable"]
    if not gold_unanswerable:
        return None, 0, 0
    n_hallucinated = sum(1 for r in gold_unanswerable if r["predicted_answer_type"] != "unanswerable")
    rate = n_hallucinated / len(gold_unanswerable)
    return rate, n_hallucinated, len(gold_unanswerable)


def false_abstention_rate(records):
    """The complementary error: system says 'unanswerable' on a question
    that actually IS answerable."""
    gold_answerable = [r for r in records if r["gold_answer_type"] != "unanswerable"]
    if not gold_answerable:
        return None, 0, 0
    n_false_abstain = sum(1 for r in gold_answerable if r["predicted_answer_type"] == "unanswerable")
    rate = n_false_abstain / len(gold_answerable)
    return rate, n_false_abstain, len(gold_answerable)


def evidence_grounding_rate(records):
    """Among questions the system actually answered (didn't abstain) AND
    that have a defined gold document, fraction where the evidence document
    the generator relied on (top1_id) is the correct gold document. This is
    the proxy for 'faithfulness to the right evidence' -- distinct from
    lexical faithfulness (whether the answer text appears in retrieved
    text), which is 1.0 by construction for this extractive generator and
    is reported separately in the README as a design-driven ceiling."""
    answered = [
        r for r in records
        if r["predicted_answer_type"] != "unanswerable" and r["gold_arxiv_id"] is not None
    ]
    if not answered:
        return None, 0, 0
    n_grounded = sum(1 for r in answered if r["top1_id"] == r["gold_arxiv_id"])
    rate = n_grounded / len(answered)
    return rate, n_grounded, len(answered)
