"""
Build the retrieval corpus and train/dev/test splits from the PubMedQA
labeled subset (PQA-L, 1000 expert-annotated items).

Source: ori_pqal.json (PubMedQA, https://github.com/pubmedqa/pubmedqa,
MIT license). Each item is one PubMed abstract with a question written by
the abstract's own title, multi-sentence CONTEXTS with section LABELS
(BACKGROUND/METHODS/RESULTS/...), a long free-text answer (the abstract's
conclusion), and an expert yes/no/maybe decision.

We deliberately build our own stratified split rather than reusing the
benchmark's official reasoning-required/reasoning-free splits, because
those were designed for a different (classification-only, no-retrieval)
task. Our task additionally needs a retrieval corpus of *distractor*
documents, so every item's abstract doubles as one corpus document and
splits are drawn at the document level to avoid leaking a document's own
text across train/dev/test.
"""
import json
import random
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "ori_pqal.json"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
SEED = 42
SPLIT = {"train": 0.6, "dev": 0.2, "test": 0.2}


def main():
    random.seed(SEED)
    raw = json.loads(RAW_PATH.read_text())

    corpus, queries = [], []
    for pmid, item in raw.items():
        context_text = " ".join(item["CONTEXTS"])
        corpus.append(
            {
                "doc_id": pmid,
                "text": context_text,
                "sections": item.get("LABELS", []),
                "sentences": item["CONTEXTS"],
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

    ids = [q["qid"] for q in queries]
    random.shuffle(ids)
    n = len(ids)
    n_train = int(n * SPLIT["train"])
    n_dev = int(n * SPLIT["dev"])
    split_ids = {
        "train": set(ids[:n_train]),
        "dev": set(ids[n_train : n_train + n_dev]),
        "test": set(ids[n_train + n_dev :]),
    }

    by_id = {q["qid"]: q for q in queries}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "corpus.jsonl", "w") as f:
        for doc in corpus:
            f.write(json.dumps(doc) + "\n")

    for split_name, id_set in split_ids.items():
        with open(OUT_DIR / f"{split_name}.jsonl", "w") as f:
            for qid in ids:
                if qid in id_set:
                    f.write(json.dumps(by_id[qid]) + "\n")

    print(f"corpus: {len(corpus)} documents")
    for split_name, id_set in split_ids.items():
        print(f"{split_name}: {len(id_set)} queries")


if __name__ == "__main__":
    main()
