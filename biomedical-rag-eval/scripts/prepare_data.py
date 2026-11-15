"""
Build the retrieval corpus and train/dev/test splits from the PubMedQA
labeled subset (PQA-L, 1000 expert-annotated items).

Source: ori_pqal.json (PubMedQA, https://github.com/pubmedqa/pubmedqa,
MIT license). Each item is one PubMed abstract with a question written by
the abstract's own title, multi-paragraph CONTEXTS with section LABELS
(BACKGROUND/METHODS/RESULTS/...), a long free-text answer (the abstract's
conclusion), and an expert yes/no/maybe decision.

We deliberately build our own random document-level split (seed 42,
60/20/20) rather than reusing the benchmark's official
reasoning-required/reasoning-free splits, because those were designed for
a different (classification-only, no-retrieval) task. Our task
additionally needs a retrieval corpus of *distractor* documents, so every
item's abstract doubles as one corpus document and splits are drawn at the
document level to avoid leaking a document's own text across
train/dev/test. The split is not stratified by decision label; the
resulting test-set label distribution is reported in the README.

Each corpus document keeps three views of the abstract:
- `text`:      the full abstract (all CONTEXTS paragraphs joined), used as
               the evidence string for retrieval and generation;
- `contexts`:  the original section paragraphs, aligned with `sections`;
- `sentences`: the abstract split into individual sentences -- this is the
               list a generator cites from, so a citation is a real
               sentence rather than a whole section paragraph.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import DATA_DIR, RAW_PATH, split_sentences, write_jsonl  # noqa: E402

SEED = 42
SPLIT = {"train": 0.6, "dev": 0.2, "test": 0.2}


def build_corpus_and_queries(raw: dict):
    corpus, queries = [], []
    for pmid, item in raw.items():
        contexts = [c.strip() for c in item["CONTEXTS"]]
        sentences = [s for paragraph in contexts for s in split_sentences(paragraph)]
        corpus.append(
            {
                "doc_id": pmid,
                "text": " ".join(contexts),
                "sections": item.get("LABELS", []),
                "contexts": contexts,
                "sentences": sentences,
            }
        )
        queries.append(
            {
                "qid": pmid,
                "doc_id": pmid,  # the query's own abstract is the single gold document
                "question": item["QUESTION"],
                "long_answer": item["LONG_ANSWER"],
                "final_decision": item["final_decision"],
                "meshes": item.get("MESHES", []),
                "year": item.get("YEAR"),
            }
        )
    return corpus, queries


def split_ids(ids: list[str], seed: int = SEED) -> dict[str, list[str]]:
    ids = list(ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(n * SPLIT["train"])
    n_dev = int(n * SPLIT["dev"])
    return {
        "train": ids[:n_train],
        "dev": ids[n_train : n_train + n_dev],
        "test": ids[n_train + n_dev :],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if not args.raw.exists():
        raise FileNotFoundError(f"{args.raw} not found; run scripts/download_data.py first")
    raw = json.loads(args.raw.read_text(encoding="utf-8"))

    corpus, queries = build_corpus_and_queries(raw)
    by_id = {q["qid"]: q for q in queries}
    splits = split_ids([q["qid"] for q in queries], seed=args.seed)

    write_jsonl(args.out_dir / "corpus.jsonl", corpus)
    for split_name, id_list in splits.items():
        write_jsonl(args.out_dir / f"{split_name}.jsonl", (by_id[qid] for qid in id_list))

    print(f"corpus: {len(corpus)} documents")
    for split_name, id_list in splits.items():
        labels = [by_id[qid]["final_decision"] for qid in id_list]
        dist = {lab: labels.count(lab) for lab in ("yes", "no", "maybe")}
        print(f"{split_name}: {len(id_list)} queries  {dist}")


if __name__ == "__main__":
    main()
