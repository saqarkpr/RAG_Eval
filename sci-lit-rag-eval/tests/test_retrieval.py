import pytest

from src.efficiency import benchmark_retriever
from src.retrieval import RETRIEVERS, BM25Retriever, TfidfLsaRetriever, build_retriever
from src.retrieval_eval import evaluate_retriever


@pytest.mark.parametrize("name", sorted(RETRIEVERS))
def test_retriever_finds_gold_document_top1(name, toy_docs):
    retriever = build_retriever(name, toy_docs)
    for query, gold in [
        ("dense passage retriever contrastive", "2401.00001"),
        ("calibration method language models", "2401.00002"),
        ("post training quantization four bits", "2401.00003"),
        ("hallucination detector sampling agreement", "2401.00004"),
    ]:
        ranked = retriever.retrieve(query, k=4)
        assert ranked[0][0] == gold, (name, query, ranked)
        assert len(ranked) == 4
        assert all(isinstance(s, float) for _, s in ranked)
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)


def test_retrieve_respects_k(toy_docs):
    assert len(BM25Retriever(toy_docs).retrieve("dense retriever", k=2)) == 2


def test_bm25_empty_query_returns_zero_scores(toy_docs):
    ranked = BM25Retriever(toy_docs).retrieve("the of and", k=4)  # all stopwords
    assert len(ranked) == 4
    assert all(s == 0.0 for _, s in ranked)


def test_lsa_caps_components_and_scores_are_cosines(toy_docs):
    retriever = TfidfLsaRetriever(toy_docs, n_components=999)
    assert retriever.n_components <= len(toy_docs) - 1
    ranked = retriever.retrieve("dense passage retriever", k=4)
    assert all(-1.0001 <= s <= 1.0001 for _, s in ranked)


def test_build_retriever_passes_kwargs_and_rejects_unknown(toy_docs):
    retriever = build_retriever("tfidf_lsa", toy_docs, n_components=2)
    assert retriever.n_components == 2
    with pytest.raises(ValueError, match="unknown retriever"):
        build_retriever("dense_neural", toy_docs)


def test_retrievers_expose_build_time_and_memory(toy_docs):
    for name in RETRIEVERS:
        r = build_retriever(name, toy_docs)
        assert r.build_time_s >= 0
        assert r.approx_memory_bytes() > 0


def test_evaluate_retriever_ignores_unanswerable_and_computes_recall_mrr(toy_docs, toy_qa):
    metrics, per_query = evaluate_retriever(BM25Retriever(toy_docs), toy_qa)
    assert metrics["n_queries"] == 8  # the 4 unanswerable items have no gold document
    assert len(per_query) == 8
    assert metrics["recall@1"] == 1.0
    assert metrics["recall@5"] == 1.0
    assert metrics["mrr"] == 1.0


def test_evaluate_retriever_handles_missing_gold_document():
    class _Never:
        def retrieve(self, query, k=5):
            return [("other", 1.0)]

    qa = [{"qid": "q", "question": "x", "answer_type": "yes_no", "gold_arxiv_id": "gold"}]
    metrics, per_query = evaluate_retriever(_Never(), qa)
    assert metrics["recall@1"] == 0.0 and metrics["mrr"] == 0.0
    assert per_query[0]["rank"] is None


def test_benchmark_retriever_reports_latency_fields(toy_docs):
    stats = benchmark_retriever(BM25Retriever(toy_docs), ["dense", "calibration"], n_repeats=2)
    assert stats["n_queries_timed"] == 4
    assert stats["mean_latency_ms"] >= 0
    assert stats["p95_latency_ms"] >= 0
    assert stats["approx_memory_kb"] > 0
