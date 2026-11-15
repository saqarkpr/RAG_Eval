import pytest

from src.data_utils import corpus_index_by_id
from src.generator import (
    FEATURE_NAMES,
    extract_candidate_spans,
    extract_features,
    fuzzy_match,
    predict_extractive_span,
)
from src.retrieval import BM25Retriever


def test_extract_candidate_spans_ranks_by_frequency_then_position():
    text = "We introduce DPR. Compared with BM25, DPR is stronger. BM25 is lexical. ALSO GPU."
    ranked = extract_candidate_spans(text)
    assert ranked[:2] == ["DPR", "BM25"]  # both twice; DPR appears first
    assert "GPU" not in ranked  # stoplisted acronym


def test_extract_candidate_spans_detects_mixed_case_method_names():
    assert "fDPO" in extract_candidate_spans("We propose fDPO for alignment.")
    assert "HalluCheck" in extract_candidate_spans("We describe HalluCheck here.")
    assert extract_candidate_spans("plain lowercase words only") == []


def test_predict_extractive_span_returns_top_candidate_or_empty():
    assert predict_extractive_span("We present QuantX, a scheme.") == "QuantX"
    assert predict_extractive_span("nothing to extract here") == ""


@pytest.mark.parametrize(
    "pred,gold,expected",
    [("DPR", "DPR", True), ("dpr", "DPR", True), ("DPR", "the DPR retriever", True),
     ("BM25", "DPR", False), ("", "DPR", False), ("DPR", "", False)],
)
def test_fuzzy_match(pred, gold, expected):
    assert fuzzy_match(pred, gold) is expected


def test_extract_features_shape_and_semantics(toy_docs):
    docs_by_id = corpus_index_by_id(toy_docs)
    retriever = BM25Retriever(toy_docs)
    q = "Does DPR not use contrastive learning?"
    retrieved = retriever.retrieve(q, k=4)
    feats, top1_id = extract_features(q, retrieved, docs_by_id)
    assert top1_id == "2401.00001"
    assert len(feats) == len(FEATURE_NAMES)
    f = dict(zip(FEATURE_NAMES, feats))
    assert f["top1_score"] == retrieved[0][1]
    assert f["margin"] == pytest.approx(retrieved[0][1] - retrieved[1][1])
    assert f["has_negation"] == 1
    assert f["starts_aux"] == 1
    assert f["q_len"] == len(q.split())
    assert 0.0 <= f["overlap"] <= 1.0
    assert f["n_candidates"] >= 1


def test_extract_features_single_result_and_empty_input(toy_docs):
    docs_by_id = corpus_index_by_id(toy_docs)
    feats, _ = extract_features("What is CALM?", [("2401.00002", 1.0)], docs_by_id)
    assert dict(zip(FEATURE_NAMES, feats))["margin"] == 1.0  # top2 score defaults to 0
    with pytest.raises(ValueError):
        extract_features("q", [], docs_by_id)
