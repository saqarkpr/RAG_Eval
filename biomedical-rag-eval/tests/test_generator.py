"""Tests for the classical (no downloaded weights) generator path, which
is the part of generator.py that actually runs in CI. HFLocalLLMGenerator
needs torch/transformers + a GPU-friendly environment and is exercised on
Colab instead (see README) -- it is intentionally not imported here so
these tests never require installing torch just to check the classical
baseline.
"""
from src.generator import ExtractiveClassifierGenerator, GenerationResult, best_matching_sentence


def test_best_matching_sentence_picks_highest_overlap():
    sentences = [
        "Patients received a placebo tablet daily.",
        "Metformin reduced fasting glucose significantly.",
        "No adverse events were reported.",
    ]
    query_words = {"metformin", "glucose"}
    assert best_matching_sentence(query_words, sentences) == sentences[1]


def test_best_matching_sentence_empty_list_returns_empty_string():
    assert best_matching_sentence({"anything"}, []) == ""


def test_best_matching_sentence_no_overlap_returns_first_sentence():
    sentences = ["Alpha beta gamma.", "Delta epsilon zeta."]
    assert best_matching_sentence({"unrelated"}, sentences) == sentences[0]


TRAIN_QUESTIONS = [
    "Does metformin lower glucose?",
    "Does metformin lower glucose?",
    "Is ibuprofen an antibiotic?",
    "Is ibuprofen an antibiotic?",
    "Does exercise help the heart?",
    "Does exercise help the heart?",
]
TRAIN_EVIDENCE = [
    "Metformin significantly lowered fasting glucose in the treatment group.",
    "Metformin significantly lowered fasting glucose in the treatment group.",
    "Ibuprofen is a nonsteroidal anti inflammatory drug, not an antibiotic.",
    "Ibuprofen is a nonsteroidal anti inflammatory drug, not an antibiotic.",
    "Regular aerobic exercise improved cardiovascular outcomes in all cohorts.",
    "Regular aerobic exercise improved cardiovascular outcomes in all cohorts.",
]
TRAIN_LABELS = ["yes", "yes", "no", "no", "yes", "yes"]


def _fitted_generator():
    gen = ExtractiveClassifierGenerator()
    gen.fit(TRAIN_QUESTIONS, TRAIN_EVIDENCE, TRAIN_LABELS)
    return gen


def test_extractive_generator_fit_sets_classes():
    gen = _fitted_generator()
    assert set(gen.classes_) == {"yes", "no"}


def test_extractive_generator_generate_returns_generation_result():
    gen = _fitted_generator()
    out = gen.generate(
        question="Does metformin lower glucose?",
        doc_id="d1",
        evidence_text=TRAIN_EVIDENCE[0],
        evidence_sentences=[TRAIN_EVIDENCE[0]],
    )
    assert isinstance(out, GenerationResult)
    assert out.decision in {"yes", "no"}
    assert out.cited_doc_id == "d1"


def test_extractive_generator_class_probs_sum_to_one():
    gen = _fitted_generator()
    out = gen.generate("Is ibuprofen an antibiotic?", "d2", TRAIN_EVIDENCE[2], [TRAIN_EVIDENCE[2]])
    assert abs(sum(out.class_probs.values()) - 1.0) < 1e-6


def test_extractive_generator_confidence_matches_top_class_prob():
    gen = _fitted_generator()
    out = gen.generate("Does exercise help the heart?", "d3", TRAIN_EVIDENCE[4], [TRAIN_EVIDENCE[4]])
    assert out.confidence == max(out.class_probs.values())


def test_extractive_generator_citation_is_verbatim_from_evidence_sentences():
    # the whole point of the constrained-citation design: the cited
    # sentence must be one of the exact strings offered, never a
    # paraphrase or hallucinated quote.
    gen = _fitted_generator()
    sentences = [TRAIN_EVIDENCE[0], "An unrelated filler sentence about weather."]
    out = gen.generate("Does metformin lower glucose?", "d1", TRAIN_EVIDENCE[0], sentences)
    assert out.cited_sentence in sentences


def test_extractive_generator_handles_no_evidence_gracefully():
    gen = _fitted_generator()
    out = gen.generate("Does metformin lower glucose?", "none", "", [])
    assert out.cited_sentence == ""
    assert out.decision in {"yes", "no"}
