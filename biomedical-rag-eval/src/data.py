"""
Shared data helpers for scripts/ and tests/: project paths, JSONL I/O, a
sentence splitter, and a loader that gives a clear error when the
processed data has not been built yet (instead of a bare FileNotFoundError
from deep inside a script).
"""
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "ori_pqal.json"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# Split after sentence-final punctuation that is followed by whitespace and
# an uppercase letter, digit or opening bracket. Abbreviations such as
# "e.g. the" or "vs. control" are therefore left intact, while "...
# patients. The" is split. Good enough for PubMed abstracts; a full
# sentence tokenizer would need an extra dependency.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'])")


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences. Never returns empty strings; a
    paragraph with no sentence boundary comes back as a one-element list."""
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def load_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def load_processed(*splits: str, data_dir: Path = DATA_DIR):
    """Return (corpus, corpus_by_id, *split_rows) from data/processed.

    Raises a FileNotFoundError that says how to build the data if any of
    the requested files is missing.
    """
    needed = [data_dir / "corpus.jsonl"] + [data_dir / f"{s}.jsonl" for s in splits]
    missing = [p for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing processed data: {', '.join(str(p) for p in missing)}. "
            "Run `python scripts/download_data.py && python scripts/prepare_data.py` "
            "(or `make data`) first."
        )
    corpus = load_jsonl(needed[0])
    corpus_by_id = {d["doc_id"]: d for d in corpus}
    split_rows = [load_jsonl(p) for p in needed[1:]]
    return (corpus, corpus_by_id, *split_rows)
