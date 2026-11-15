import json

from src.config import DEFAULTS, load_config, resolve_data_paths
from src.data_utils import STOPWORDS, corpus_index_by_id, load_corpus, load_qa, tokenize


def test_tokenize_lowercases_and_drops_stopwords():
    toks = tokenize("The Model IS Trained on Data")
    assert toks == ["model", "trained", "data"]
    assert not (set(toks) & STOPWORDS)


def test_tokenize_keeps_hyphenated_and_alphanumeric_tokens():
    assert tokenize("GPT-4 and BM25", drop_stopwords=False) == ["gpt-4", "and", "bm25"]


def test_tokenize_can_keep_stopwords():
    assert "the" in tokenize("the model", drop_stopwords=False)


def test_load_corpus_builds_text_and_index(tmp_path):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps([{"arxiv_id": "1", "title": " T ", "abstract": " A. "}]))
    docs = load_corpus(path)
    assert docs[0]["text"] == "T. A."
    assert corpus_index_by_id(docs) == {"1": docs[0]}


def test_load_qa_roundtrip(tmp_path):
    path = tmp_path / "qa.json"
    items = [{"qid": "q1", "question": "Q?", "answer_type": "yes_no"}]
    path.write_text(json.dumps(items))
    assert load_qa(path) == items


def test_load_config_defaults_when_file_missing(tmp_path):
    cfg = load_config(tmp_path / "nope.yaml")
    assert cfg == DEFAULTS
    assert cfg is not DEFAULTS  # a copy, never the shared default dict


def test_load_config_merges_nested_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("seed: 7\ntfidf_lsa:\n  n_components: 3\n")
    cfg = load_config(path)
    assert cfg["seed"] == 7
    assert cfg["tfidf_lsa"] == {"n_components": 3}
    assert cfg["top_k"] == DEFAULTS["top_k"]
    assert DEFAULTS["tfidf_lsa"]["n_components"] == 16  # defaults untouched


def test_project_default_config_matches_reported_settings():
    cfg = load_config()
    assert (cfg["retriever"], cfg["seed"], cfg["top_k"], cfg["n_splits"]) == ("bm25", 42, 5, 5)
    corpus, qa = resolve_data_paths(cfg)
    assert corpus.exists() and qa.exists()
