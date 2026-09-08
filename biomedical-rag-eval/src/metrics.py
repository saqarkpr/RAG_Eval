"""Evaluation metrics: retrieval quality, classification accuracy,
calibration (ECE/Brier), and a lexical faithfulness/support check used as
a hallucination-risk proxy.
"""
import numpy as np


def recall_at_k(gold_doc_id: str, retrieved_doc_ids: list[str], k: int) -> int:
    return int(gold_doc_id in retrieved_doc_ids[:k])


def reciprocal_rank(gold_doc_id: str, retrieved_doc_ids: list[str]) -> float:
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id == gold_doc_id:
            return 1.0 / rank
    return 0.0


def accuracy(y_true: list, y_pred: list) -> float:
    assert len(y_true) == len(y_pred)
    return float(np.mean([a == b for a, b in zip(y_true, y_pred)]))


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> float:
    """Standard ECE: bin predictions by confidence, compare mean confidence
    to empirical accuracy in each bin, weight by bin occupancy."""
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (
            confidences >= lo
        ) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def brier_score_multiclass(probs: np.ndarray, y_true_idx: np.ndarray, n_classes: int) -> float:
    """Multiclass Brier score: mean squared error between the predicted
    probability simplex and the one-hot true label."""
    onehot = np.zeros((len(y_true_idx), n_classes))
    onehot[np.arange(len(y_true_idx)), y_true_idx] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def reliability_bins(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    """Return (bin_centers, bin_accuracy, bin_confidence, bin_counts) for
    plotting a reliability diagram."""
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    centers, accs, confs, counts = [], [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (
            confidences >= lo
        ) & (confidences <= hi)
        centers.append((lo + hi) / 2)
        counts.append(int(mask.sum()))
        if mask.sum() == 0:
            accs.append(np.nan)
            confs.append(np.nan)
        else:
            accs.append(float(correct[mask].mean()))
            confs.append(float(confidences[mask].mean()))
    return np.array(centers), np.array(accs), np.array(confs), np.array(counts)


_WORD_RE = None


def _words(text: str) -> set[str]:
    global _WORD_RE
    if _WORD_RE is None:
        import re

        _WORD_RE = re.compile(r"[a-zA-Z]{3,}")
    return set(w.lower() for w in _WORD_RE.findall(text))

_STOPWORDS = {
    "the", "and", "for", "was", "were", "with", "that", "this", "from", "have",
    "has", "are", "not", "but", "our", "these", "those", "than", "into", "such",
    "may", "can", "also", "been", "did", "does", "each", "any", "all", "more",
}


def support_ratio(generated_text: str, evidence_text: str) -> float:
    """Fraction of the generated answer's (non-stopword) content words that
    also appear in the cited evidence passage. A cheap, model-free proxy
    for lexical groundedness / hallucination risk: it catches claims that
    introduce vocabulary absent from any retrieved evidence, but it is NOT
    an entailment check (a low-overlap paraphrase would be unfairly
    penalized, and a high-overlap contradiction would be missed) -- see
    README limitations.
    """
    gen_words = _words(generated_text) - _STOPWORDS
    if not gen_words:
        return 1.0
    ev_words = _words(evidence_text) - _STOPWORDS
    supported = gen_words & ev_words
    return len(supported) / len(gen_words)
