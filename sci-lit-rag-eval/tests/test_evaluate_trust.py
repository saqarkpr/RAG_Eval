import numpy as np

from src.evaluate_trust import (
    brier_score,
    evidence_grounding_rate,
    expected_calibration_error,
    false_abstention_rate,
    hallucination_rate,
)


def test_ece_zero_for_perfect_calibration():
    conf = np.array([0.9] * 10 + [0.1] * 10)
    correct = np.array([1.0] * 9 + [0.0] + [0.0] * 9 + [1.0])
    ece, bins = expected_calibration_error(conf, correct, n_bins=10)
    assert ece == 0.0
    assert len(bins) == 10
    assert sum(b["count"] for b in bins) == 20
    assert all(b["acc"] is None for b in bins if b["count"] == 0)


def test_ece_large_for_confident_but_wrong_and_top_edge_inclusive():
    ece, bins = expected_calibration_error([1.0] * 5, [0.0] * 5, n_bins=10)
    assert ece == 1.0
    assert bins[-1]["count"] == 5  # confidence exactly 1.0 lands in the last bin


def test_brier_score_known_values():
    assert brier_score([1.0, 0.0], [1.0, 0.0]) == 0.0
    assert brier_score([0.5, 0.5], [1.0, 0.0]) == 0.25


def _rec(gold_type, pred_type, gold_id="g", top1="g"):
    return {"gold_answer_type": gold_type, "predicted_answer_type": pred_type,
            "gold_arxiv_id": gold_id, "top1_id": top1}


def test_hallucination_rate_counts_answers_on_unanswerable_items():
    records = [_rec("unanswerable", "yes_no", None), _rec("unanswerable", "unanswerable", None),
               _rec("yes_no", "yes_no")]
    assert hallucination_rate(records) == (0.5, 1, 2)
    assert hallucination_rate([_rec("yes_no", "yes_no")]) == (None, 0, 0)


def test_false_abstention_rate_counts_abstentions_on_answerable_items():
    records = [_rec("yes_no", "unanswerable"), _rec("extractive", "extractive"),
               _rec("unanswerable", "unanswerable", None)]
    assert false_abstention_rate(records) == (0.5, 1, 2)
    assert false_abstention_rate([_rec("unanswerable", "unanswerable", None)]) == (None, 0, 0)


def test_evidence_grounding_rate_only_over_answered_items_with_gold_doc():
    records = [
        _rec("yes_no", "yes_no", "g", "g"),          # answered, grounded
        _rec("yes_no", "yes_no", "g", "other"),      # answered, wrong document
        _rec("yes_no", "unanswerable", "g", "g"),    # abstained: excluded
        _rec("unanswerable", "yes_no", None, "x"),   # no gold document: excluded
    ]
    assert evidence_grounding_rate(records) == (0.5, 1, 2)
    assert evidence_grounding_rate([_rec("yes_no", "unanswerable")]) == (None, 0, 0)
