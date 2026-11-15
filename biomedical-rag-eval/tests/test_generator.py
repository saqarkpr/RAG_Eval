"""Tests for generator.py. The classical generator runs end to end here.
HFLocalLLMGenerator needs torch/transformers and a GPU-friendly
environment (exercised on Colab, see README), so only its torch-free
helpers -- evidence truncation, prompt construction, citation parsing and
candidate-token enumeration -- are tested, with a fake tokenizer where one
is needed. Nothing in this file imports torch."""
import pytest

from src.generator import (
    DECISION_WORDS,
    SYSTEM_NO_EVIDENCE,
    SYSTEM_WITH_EVIDENCE,
    ExtractiveClassifierGenerator,
    GenerationResult,
    best_matching_sentence,
    build_messages,
    candidate_token_sequences,
    citation_messages,
    parse_sentence_choice,
    truncate_evidence,
)


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


def test_extractive_generator_generate_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        ExtractiveClassifierGenerator().generate("q", "d1", "evidence", ["evidence"])


def test_extractive_generator_fit_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        ExtractiveClassifierGenerator().fit(["q1", "q2"], ["e1"], ["yes", "no"])


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
    assert out.answer_text.startswith(out.decision.capitalize() + ".")
    assert "(source: d1)" in out.answer_text


def test_extractive_generator_class_probs_sum_to_one():
    gen = _fitted_generator()
    out = gen.generate(
        "Is ibuprofen an antibiotic?", "d2", TRAIN_EVIDENCE[2], [TRAIN_EVIDENCE[2]]
    )
    assert abs(sum(out.class_probs.values()) - 1.0) < 1e-6


def test_extractive_generator_confidence_matches_top_class_prob():
    gen = _fitted_generator()
    out = gen.generate(
        "Does exercise help the heart?", "d3", TRAIN_EVIDENCE[4], [TRAIN_EVIDENCE[4]]
    )
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


# --- torch-free helpers of HFLocalLLMGenerator -----------------------------


def test_truncate_evidence_short_text_is_unchanged():
    assert truncate_evidence("Short. Text.", 100) == "Short. Text."


def test_truncate_evidence_never_cuts_a_sentence_in_half():
    text = "First sentence here. Second sentence here. Third sentence here."
    out = truncate_evidence(text, 45)
    assert out == "First sentence here. Second sentence here."
    assert len(out) <= 45


def test_truncate_evidence_hard_cuts_when_first_sentence_exceeds_budget():
    text = "A" * 50 + ". Second."
    out = truncate_evidence(text, 20)
    assert out == "A" * 20


def test_build_messages_with_evidence_uses_evidence_prompt():
    msgs = build_messages("Does X work?", "X worked in all patients.")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == SYSTEM_WITH_EVIDENCE
    assert "Evidence:\nX worked in all patients." in msgs[1]["content"]
    assert "Question: Does X work?" in msgs[1]["content"]


def test_build_messages_without_evidence_has_no_insufficiency_cue():
    # Regression test for the bug described in the README: the no-context
    # prompt must not mention missing/insufficient evidence, or the model
    # collapses to "maybe".
    msgs = build_messages("Does X work?", "")
    assert msgs[0]["content"] == SYSTEM_NO_EVIDENCE
    joined = " ".join(m["content"] for m in msgs).lower()
    assert "insufficient" not in joined
    assert "evidence" not in joined


def test_build_messages_truncates_long_evidence():
    long_evidence = " ".join(f"Sentence number {i}." for i in range(200))
    msgs = build_messages("Q?", long_evidence, max_evidence_chars=100)
    user = msgs[1]["content"]
    evidence_block = user.split("Evidence:\n", 1)[1].split("\n\nQuestion:", 1)[0]
    assert len(evidence_block) <= 100
    assert evidence_block.endswith(".")


def test_citation_messages_replays_decision_and_numbers_sentences():
    base = build_messages("Q?", "S one. S two.")
    msgs = citation_messages(base, "yes", ["S one.", "S two."])
    assert msgs[:2] == base
    assert msgs[2] == {"role": "assistant", "content": "yes"}
    assert msgs[3]["role"] == "user"
    assert "1. S one.\n2. S two." in msgs[3]["content"]
    assert "only the number" in msgs[3]["content"]


@pytest.mark.parametrize(
    "reply,n,expected",
    [
        ("2", 3, 1),
        ("Sentence 3.", 3, 2),
        ("0", 3, None),  # 1-based numbering: 0 is out of range
        ("4", 3, None),
        ("none of them", 3, None),
        ("", 3, None),
    ],
)
def test_parse_sentence_choice(reply, n, expected):
    assert parse_sentence_choice(reply, n) == expected


class _FakeTokenizer:
    """Mimics a BPE tokenizer where a leading space changes the token id
    and 'maybe' is split into two pieces."""

    VOCAB = {
        "yes": [1], " yes": [2], "Yes": [3], " Yes": [4],
        "no": [5], " no": [6], "No": [7], " No": [8],
        "maybe": [9, 10], " maybe": [11, 10], "Maybe": [12, 10], " Maybe": [11, 10],
    }

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return {"input_ids": list(self.VOCAB[text])}


def test_candidate_token_sequences_enumerates_distinct_surface_forms():
    tok = _FakeTokenizer()
    assert candidate_token_sequences(tok, "yes") == [[1], [2], [3], [4]]
    # " maybe" and " Maybe" share a token sequence -> deduplicated
    assert candidate_token_sequences(tok, "maybe") == [[9, 10], [11, 10], [12, 10]]


def test_decision_words_order_is_stable():
    assert DECISION_WORDS == ["yes", "no", "maybe"]
