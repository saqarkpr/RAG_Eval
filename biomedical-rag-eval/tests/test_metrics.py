"""Every number in results/tables/*.csv comes out of these functions, so a
silent bug here would be invisible in the README tables -- these tests
pin down known-answer cases (perfect calibration -> ECE 0, etc.) rather
than just checking "it runs".
"""
import numpy as np

from src.metrics import (
    accuracy,
    bootstrap_ci,
    brier_score_multiclass,
    expected_calibration_error,
    reliability_bins,
    support_ratio,
)


def test_recall_at_k_hit_and_miss():
    from src.metrics import recall_at_k

    assert recall_at_k("d1", ["d1", "d2", "d3"], k=1) == 1
    assert recall_at_k("d3", ["d1", "d2", "d3"], k=2) == 0
    assert recall_at_k("d3", ["d1", "d2", "d3"], k=3) == 1
    assert recall_at_k("missing", ["d1", "d2"], k=10) == 0


def test_reciprocal_rank():
    from src.metrics import reciprocal_rank

    assert reciprocal_rank("d1", ["d1", "d2", "d3"]) == 1.0
    assert reciprocal_rank("d2", ["d1", "d2", "d3"]) == 0.5
    assert reciprocal_rank("missing", ["d1", "d2"]) == 0.0


def test_accuracy_basic():
    assert accuracy(["yes", "no", "yes"], ["yes", "no", "no"]) == 2 / 3
    assert accuracy(["a"], ["a"]) == 1.0


def test_bootstrap_ci_mean_matches_plain_mean():
    values = [1, 0, 1, 1, 0, 1, 1, 0]
    mean, lo, hi = bootstrap_ci(values, n_boot=500, seed=42)
    assert mean == np.mean(values)
    assert lo <= mean <= hi


def test_bootstrap_ci_is_deterministic_given_seed():
    values = [0.2, 0.9, 0.4, 0.7, 1.0, 0.0, 0.5]
    a = bootstrap_ci(values, n_boot=200, seed=7)
    b = bootstrap_ci(values, n_boot=200, seed=7)
    assert a == b


def test_bootstrap_ci_degenerate_all_same_value_has_zero_width():
    mean, lo, hi = bootstrap_ci([1.0, 1.0, 1.0, 1.0])
    assert mean == lo == hi == 1.0


def test_ece_is_zero_for_perfect_calibration():
    # confidence exactly matches empirical accuracy in every occupied bin
    confidences = np.array([0.9] * 10 + [0.1] * 10)
    correct = np.array([1.0] * 9 + [0.0] * 1 + [0.0] * 9 + [1.0] * 1)
    ece = expected_calibration_error(confidences, correct, n_bins=10)
    assert ece == 0.0


def test_ece_is_large_for_confident_but_wrong():
    confidences = np.array([0.99] * 20)
    correct = np.array([0.0] * 20)  # always wrong despite near-certainty
    ece = expected_calibration_error(confidences, correct, n_bins=10)
    assert ece > 0.9


def test_brier_score_zero_for_perfect_one_hot_prediction():
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y_true_idx = np.array([0, 1])
    assert brier_score_multiclass(probs, y_true_idx, n_classes=3) == 0.0


def test_brier_score_positive_for_uniform_prediction():
    probs = np.array([[1 / 3, 1 / 3, 1 / 3]])
    y_true_idx = np.array([0])
    score = brier_score_multiclass(probs, y_true_idx, n_classes=3)
    assert score > 0.0


def test_reliability_bins_shapes_and_empty_bins_are_nan():
    confidences = np.array([0.05, 0.95])
    correct = np.array([1.0, 0.0])
    centers, accs, confs, counts = reliability_bins(confidences, correct, n_bins=10)
    assert len(centers) == len(accs) == len(confs) == len(counts) == 10
    assert counts.sum() == 2
    # bins with zero items must report NaN, not a fabricated 0
    empty_bins = counts == 0
    assert np.all(np.isnan(accs[empty_bins]))


def test_support_ratio_full_overlap_is_one():
    assert support_ratio("glucose lowers blood pressure", "glucose lowers blood pressure") == 1.0


def test_support_ratio_zero_overlap_is_zero():
    assert support_ratio("completely unrelated statement", "totally different evidence text") == 0.0


def test_support_ratio_empty_generated_text_is_trivially_one():
    # no content words claimed -> nothing unsupported -> vacuously grounded
    assert support_ratio("", "any evidence") == 1.0


def test_support_ratio_ignores_case_and_short_tokens():
    assert support_ratio("Glucose", "glucose is a sugar") == 1.0


def test_support_ratio_keeps_negation_word_not_as_content():
    # "not effective" vs "effective" -- the word "not" must count as a
    # content word so that support_ratio can penalize a flipped claim.
    generated = "not effective"
    evidence = "the drug was effective in all patients"
    # "effective" is shared, but "not" is absent from evidence -> ratio < 1
    assert support_ratio(generated, evidence) < 1.0
