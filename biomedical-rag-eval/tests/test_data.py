"""Tests for src/data.py: the sentence splitter that decides what a
generator can cite, and the JSONL/processed-data helpers."""
import pytest

from src.data import load_jsonl, load_processed, split_sentences, write_jsonl


def test_split_sentences_basic():
    text = "Metformin lowers glucose. It is widely used. Is it safe?"
    assert split_sentences(text) == [
        "Metformin lowers glucose.",
        "It is widely used.",
        "Is it safe?",
    ]


def test_split_sentences_keeps_common_abbreviations_together():
    # "vs. control" and "e.g. the" are not sentence boundaries (lowercase follows)
    text = "Drug A vs. control improved outcomes, e.g. the pain score. Results were mixed."
    assert split_sentences(text) == [
        "Drug A vs. control improved outcomes, e.g. the pain score.",
        "Results were mixed.",
    ]


def test_split_sentences_handles_numbers_and_brackets_after_boundary():
    text = "Patients were enrolled. 25 completed follow-up. (Two withdrew.)"
    assert split_sentences(text) == [
        "Patients were enrolled.",
        "25 completed follow-up.",
        "(Two withdrew.)",
    ]


def test_split_sentences_empty_and_single_sentence():
    assert split_sentences("") == []
    assert split_sentences("   ") == []
    assert split_sentences("No boundary here") == ["No boundary here"]


def test_jsonl_roundtrip(tmp_path):
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    path = tmp_path / "nested" / "rows.jsonl"
    write_jsonl(path, rows)  # creates the parent directory
    assert load_jsonl(path) == rows


def test_load_processed_reports_missing_files_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_data.py"):
        load_processed("test", data_dir=tmp_path)


def test_load_processed_returns_corpus_index_and_splits(tmp_path):
    corpus = [{"doc_id": "d1", "text": "t1"}, {"doc_id": "d2", "text": "t2"}]
    write_jsonl(tmp_path / "corpus.jsonl", corpus)
    write_jsonl(tmp_path / "train.jsonl", [{"qid": "q1"}])
    write_jsonl(tmp_path / "test.jsonl", [{"qid": "q2"}, {"qid": "q3"}])
    got_corpus, by_id, train, test = load_processed("train", "test", data_dir=tmp_path)
    assert got_corpus == corpus
    assert set(by_id) == {"d1", "d2"}
    assert [q["qid"] for q in train] == ["q1"]
    assert [q["qid"] for q in test] == ["q2", "q3"]
