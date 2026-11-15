"""End-to-end: train.py's cascade under CV and evaluate.py's summary,
report and negation probe on the toy corpus, written to a temp directory
(the committed results/ are never touched)."""
import json

import evaluate
import train
from src.data_utils import corpus_index_by_id
from src.retrieval import BM25Retriever


def test_run_cv_covers_every_item_with_valid_predictions(toy_docs, toy_qa):
    docs_by_id = corpus_index_by_id(toy_docs)
    rows = train.build_dataset(toy_qa, BM25Retriever(toy_docs), docs_by_id, k=4)
    preds = train.run_cv(rows, seed=0, n_splits=2)
    assert len(preds) == len(toy_qa) and all(p is not None for p in preds)
    for p in preds:
        assert p["predicted_answer_type"] in {"unanswerable", "yes_no", "extractive"}
        assert 0.0 <= p["confidence"] <= 1.0
        if p["predicted_answer_type"] == "yes_no":
            assert p["predicted_value"] in {"Yes", "No"}
        elif p["predicted_answer_type"] == "extractive":
            # the span is a literal substring of the retrieved evidence, by construction
            assert p["predicted_value"] in p["top1_text"]
        else:
            assert p["predicted_value"] is None


def test_run_cv_is_deterministic_for_a_seed(toy_docs, toy_qa):
    docs_by_id = corpus_index_by_id(toy_docs)
    rows = train.build_dataset(toy_qa, BM25Retriever(toy_docs), docs_by_id, k=4)
    a = [(p["predicted_answer_type"], p["confidence"]) for p in train.run_cv(rows, 3, 2)]
    b = [(p["predicted_answer_type"], p["confidence"]) for p in train.run_cv(rows, 3, 2)]
    assert a == b


def test_negate_question_strips_aux_and_lowercases():
    # the leading auxiliary is dropped and only the first character is lowercased
    assert evaluate.negate_question("Does DPR use contrastive learning?") == (
        "Is it NOT the case that dPR use contrastive learning?"
    )
    assert evaluate.negate_question("Are LLMs calibrated?") == (
        "Is it NOT the case that lLMs calibrated?"
    )
    assert evaluate.negate_question("?") == "?"


def test_is_correct_rules():
    assert evaluate.is_correct({"gold_answer_type": "unanswerable",
                                "predicted_answer_type": "unanswerable"})
    assert not evaluate.is_correct({"gold_answer_type": "unanswerable",
                                    "predicted_answer_type": "yes_no"})
    assert evaluate.is_correct({"gold_answer_type": "yes_no", "predicted_answer_type": "yes_no",
                                "predicted_value": "No", "gold_answer": "No"})
    assert not evaluate.is_correct({"gold_answer_type": "yes_no",
                                    "predicted_answer_type": "extractive"})
    assert evaluate.is_correct({"gold_answer_type": "extractive",
                                "predicted_answer_type": "extractive",
                                "predicted_value": "dpr", "gold_answer": "DPR"})


def test_train_and_evaluate_end_to_end(tmp_path, toy_docs, toy_qa):
    (tmp_path / "corpus.json").write_text(json.dumps(
        [{k: d[k] for k in ("arxiv_id", "title", "abstract")} for d in toy_docs]))
    (tmp_path / "qa.json").write_text(json.dumps(toy_qa))
    config = tmp_path / "config.yaml"
    config.write_text(
        "seed: 1\ntop_k: 4\nn_splits: 2\nece_bins: 5\ntfidf_lsa:\n  n_components: 2\n"
        f"data:\n  corpus: {tmp_path / 'corpus.json'}\n  qa_benchmark: {tmp_path / 'qa.json'}\n"
    )
    out = tmp_path / "results"
    for name in ("bm25", "tfidf_lsa"):
        train.main(["--config", str(config), "--retriever", name, "--results-dir", str(out)])
        assert (out / f"records_{name}.json").exists()
        assert (out / f"retrieval_metrics_{name}.json").exists()
        assert (out / f"efficiency_{name}.json").exists()
        run_cfg = json.loads((out / f"run_config_{name}.json").read_text())
        assert run_cfg["seed"] == 1 and run_cfg["top_k"] == 4 and run_cfg["n_splits"] == 2

    evaluate.main(["--config", str(config), "--results-dir", str(out)])
    summary = json.loads((out / "summary.json").read_text())
    assert [s["retriever"] for s in summary] == ["bm25", "tfidf_lsa"]
    for s in summary:
        assert s["n_items"] == len(toy_qa)
        assert 0.0 <= s["accuracy"] <= 1.0 and 0.0 <= s["ece"] <= 1.0
        assert s["hallucination_count"].endswith("/4")
        assert s["false_abstention_count"].endswith("/8")
        assert (out / f"reliability_{s['retriever']}.png").exists()
    robustness = json.loads((out / "robustness.json").read_text())
    assert robustness["n_probed"] == 4
    assert 0.0 <= robustness["negation_flip_rate"] <= 1.0
    report = (out / "report.md").read_text()
    assert "| bm25 |" in report and "| tfidf_lsa |" in report and "Negation flip rate" in report
