"""Retrieval backends are the base of the whole pipeline -- if they can't
reliably find an exact lexical match on a toy corpus, nothing downstream
(accuracy, calibration, faithfulness) means anything. These tests build a
tiny synthetic corpus with unambiguous, non-overlapping vocabulary per
document, so the "correct" answer is unambiguous by construction.
"""
from src.retrieval import BM25Retriever, LsaRetriever, tokenize

DOC_IDS = ["d1", "d2", "d3", "d4"]
TEXTS = [
    "Metformin lowers blood glucose in patients with type two diabetes mellitus.",
    "Ibuprofen is a nonsteroidal anti inflammatory drug used for pain relief.",
    "Amoxicillin is a beta lactam antibiotic used to treat bacterial infections.",
    "Regular aerobic exercise improves cardiovascular fitness and lowers resting heart rate.",
]


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Metformin, DIABETES!") == ["metformin", "diabetes"]


def test_tokenize_drops_non_alpha_tokens():
    # numbers and standalone punctuation are not word tokens
    assert "2" not in tokenize("type 2 diabetes")
    assert tokenize("type 2 diabetes") == ["type", "diabetes"]


def test_bm25_retrieves_exact_lexical_match_top1():
    retriever = BM25Retriever(DOC_IDS, TEXTS)
    result = retriever.retrieve("metformin diabetes glucose", k=4)
    assert result.doc_ids[0] == "d1"
    assert len(result.doc_ids) == 4
    assert len(result.scores) == 4
    assert result.latency_s >= 0


def test_bm25_ranks_unrelated_query_below_related_one():
    retriever = BM25Retriever(DOC_IDS, TEXTS)
    top1_exercise = retriever.retrieve("aerobic exercise heart rate", k=1).doc_ids[0]
    assert top1_exercise == "d4"


def test_bm25_respects_k():
    retriever = BM25Retriever(DOC_IDS, TEXTS)
    assert len(retriever.retrieve("antibiotic infection", k=2).doc_ids) == 2


# LsaRetriever's TfidfVectorizer uses min_df=2 (a word must appear in at
# least 2 documents to enter the vocabulary at all) -- a sensible default
# on the real ~1000-document corpus, but it means a 4-document, one-topic-
# per-document toy corpus like TEXTS above has almost no vocabulary left
# after filtering (most content words then occur in exactly one document
# and get dropped), which would make an LSA-specific test flaky for
# reasons that have nothing to do with LsaRetriever's own logic. So the
# LSA tests below use a corpus where each topic is repeated across two
# documents, which is enough for its distinctive vocabulary to clear
# min_df=2 while still being distinguishable from the other topics.
LSA_DOC_IDS = ["a1", "a2", "b1", "b2", "c1", "c2", "d1", "d2"]
LSA_TEXTS = [
    "Metformin lowers blood glucose in patients with type two diabetes.",
    "Metformin therapy reduced blood glucose levels in diabetes patients.",
    "Ibuprofen is a nonsteroidal anti inflammatory drug used for pain relief.",
    "Ibuprofen relieves pain through nonsteroidal anti inflammatory action.",
    "Amoxicillin is a beta lactam antibiotic used to treat bacterial infections.",
    "Amoxicillin treats bacterial infections as a beta lactam antibiotic.",
    "Regular aerobic exercise improves cardiovascular fitness and heart rate.",
    "Aerobic exercise regularly improves cardiovascular fitness and heart rate.",
]


def test_lsa_retrieves_matching_topic_top1():
    retriever = LsaRetriever(LSA_DOC_IDS, LSA_TEXTS, n_components=4)
    result = retriever.retrieve("amoxicillin bacterial antibiotic infections", k=8)
    assert result.doc_ids[0] in {"c1", "c2"}


def test_lsa_scores_are_bounded_cosine_similarities():
    retriever = LsaRetriever(LSA_DOC_IDS, LSA_TEXTS, n_components=4)
    result = retriever.retrieve("ibuprofen pain relief", k=8)
    assert all(-1.0001 <= s <= 1.0001 for s in result.scores)


def test_lsa_caps_n_components_below_corpus_size():
    # requesting more components than documents-1 must not raise
    retriever = LsaRetriever(DOC_IDS, TEXTS, n_components=999)
    result = retriever.retrieve("diabetes", k=1)
    assert len(result.doc_ids) == 1
