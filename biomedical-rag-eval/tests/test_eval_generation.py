"""Integration test for the shared evaluate_generator loop (src/eval_generation.py):
this is the function both run_generation_eval.py (classical) and
run_generation_eval_llm.py (real LLM, run on Colab) call, so a bug here
would silently affect every number in the README. Exercised here with the
classical generator on a tiny synthetic dataset -- fast, no GPU, no
downloaded weights.
"""
import numpy as np
import pytest

from src.eval_generation import CLASSES, CONDITIONS, evaluate_generator
from src.generator import ExtractiveClassifierGenerator, GenerationResult
from src.retrieval import BM25Retriever

CORPUS = [
    {
        "doc_id": "d1",
        "text": "Metformin significantly lowered fasting glucose in the treatment group.",
        "sentences": ["Metformin significantly lowered fasting glucose in the treatment group."],
    },
    {
        "doc_id": "d2",
        "text": "Ibuprofen is a nonsteroidal anti inflammatory drug, not an antibiotic.",
        "sentences": ["Ibuprofen is a nonsteroidal anti inflammatory drug, not an antibiotic."],
    },
    {
        "doc_id": "d3",
        "text": "Regular aerobic exercise improved cardiovascular outcomes in all cohorts.",
        "sentences": ["Regular aerobic exercise improved cardiovascular outcomes in all cohorts."],
    },
]
TEST = [
    {"qid": "q1", "doc_id": "d1", "question": "Does metformin lower glucose?", "final_decision": "yes"},
    {"qid": "q2", "doc_id": "d2", "question": "Is ibuprofen an antibiotic?", "final_decision": "no"},
    {"qid": "q3", "doc_id": "d3", "question": "Does exercise help the heart?", "final_decision": "yes"},
]


def _make_generator():
    gen = ExtractiveClassifierGenerator()
    questions = [q["question"] for q in TEST] * 2
    evidence = [d["text"] for d in CORPUS] * 2
    labels = [q["final_decision"] for q in TEST] * 2
    gen.fit(questions, evidence, labels)
    return gen


def test_evaluate_generator_returns_expected_shapes(tmp_path):
    corpus_by_id = {d["doc_id"]: d for d in CORPUS}
    bm25 = BM25Retriever([d["doc_id"] for d in CORPUS], [d["text"] for d in CORPUS])
    gen = _make_generator()

    summary_df, per_query_df = evaluate_generator(gen, TEST, corpus_by_id, bm25, tmp_path)

    assert set(summary_df["condition"]) == set(CONDITIONS)
    assert len(summary_df) == len(CONDITIONS)
    assert len(per_query_df) == len(CONDITIONS) * len(TEST)
    for col in ["accuracy", "accuracy_ci_lo", "accuracy_ci_hi", "ece", "brier", "mean_confidence"]:
        assert col in summary_df.columns
        assert summary_df[col].notna().all()


def test_evaluate_generator_no_context_condition_has_no_evidence(tmp_path):
    corpus_by_id = {d["doc_id"]: d for d in CORPUS}
    bm25 = BM25Retriever([d["doc_id"] for d in CORPUS], [d["text"] for d in CORPUS])
    gen = _make_generator()

    _, per_query_df = evaluate_generator(gen, TEST, corpus_by_id, bm25, tmp_path)
    no_ctx = per_query_df[per_query_df["condition"] == "no_context"]
    assert (no_ctx["cited_doc_id"] == "none").all()
    # support_ratio is undefined (NaN) with no evidence, by construction
    assert no_ctx["support_ratio"].isna().all()


def test_evaluate_generator_gold_context_support_ratio_is_defined(tmp_path):
    corpus_by_id = {d["doc_id"]: d for d in CORPUS}
    bm25 = BM25Retriever([d["doc_id"] for d in CORPUS], [d["text"] for d in CORPUS])
    gen = _make_generator()

    _, per_query_df = evaluate_generator(gen, TEST, corpus_by_id, bm25, tmp_path)
    gold_ctx = per_query_df[per_query_df["condition"] == "gold_context"]
    assert gold_ctx["support_ratio"].notna().all()
    assert gold_ctx["cited_doc_id"].tolist() == [q["doc_id"] for q in TEST]


def test_evaluate_generator_writes_one_reliability_figure_per_condition(tmp_path):
    corpus_by_id = {d["doc_id"]: d for d in CORPUS}
    bm25 = BM25Retriever([d["doc_id"] for d in CORPUS], [d["text"] for d in CORPUS])
    gen = _make_generator()

    evaluate_generator(gen, TEST, corpus_by_id, bm25, tmp_path, figure_prefix="unittest")
    for condition in CONDITIONS:
        assert (tmp_path / "figures" / f"unittest_{condition}.png").exists()


class _BadProbsGenerator:
    """A deliberately broken generator whose class_probs do not sum to
    1 -- evaluate_generator must catch this loudly (ValueError) rather
    than silently computing a meaningless ECE/Brier score on garbage."""

    name = "bad"

    def generate(self, question, doc_id, evidence_text, evidence_sentences):
        return GenerationResult(
            answer_text="yes",
            decision="yes",
            confidence=0.9,
            class_probs={"yes": 0.9, "no": 0.05, "maybe": 0.0},  # sums to 0.95
            cited_doc_id=doc_id,
            cited_sentence="",
        )


def test_evaluate_generator_raises_on_invalid_probability_vector(tmp_path):
    corpus_by_id = {d["doc_id"]: d for d in CORPUS}
    bm25 = BM25Retriever([d["doc_id"] for d in CORPUS], [d["text"] for d in CORPUS])

    with pytest.raises(ValueError, match="class_probs do not sum to 1"):
        evaluate_generator(_BadProbsGenerator(), TEST, corpus_by_id, bm25, tmp_path)
