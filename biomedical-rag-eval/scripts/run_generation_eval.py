"""
Evaluate the classical (no downloaded weights needed) generator baseline:
`ExtractiveClassifierGenerator` under three evidence conditions --
no_context / gold_context / bm25_context -- to isolate what retrieval
quality buys (or costs) end-to-end. See src/eval_generation.py for the
shared evaluation loop: run_generation_eval_llm.py calls the exact same
function, so this baseline and the real-LLM run in that script are
apples-to-apples (same test set, same conditions, same metrics).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.eval_generation import evaluate_generator
from src.generator import ExtractiveClassifierGenerator
from src.retrieval import BM25Retriever

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_jsonl(path):
    return [json.loads(line) for line in open(path)]


def main():
    corpus = load_jsonl(DATA_DIR / "corpus.jsonl")
    train = load_jsonl(DATA_DIR / "train.jsonl")
    test = load_jsonl(DATA_DIR / "test.jsonl")
    corpus_by_id = {d["doc_id"]: d for d in corpus}

    doc_ids = [d["doc_id"] for d in corpus]
    texts = [d["text"] for d in corpus]
    bm25 = BM25Retriever(doc_ids, texts)

    # Train the classifier-generator using each training query's OWN
    # (gold) abstract as evidence -- at train time we always have the
    # correct evidence; the point of the eval below is what happens at
    # test time when evidence instead comes from a retriever.
    gen = ExtractiveClassifierGenerator()
    train_questions = [q["question"] for q in train]
    train_evidence = [corpus_by_id[q["doc_id"]]["text"] for q in train]
    train_labels = [q["final_decision"] for q in train]
    gen.fit(train_questions, train_evidence, train_labels)

    summary_df, per_query_df = evaluate_generator(
        gen, test, corpus_by_id, bm25, RESULTS_DIR, figure_prefix="reliability"
    )
    RESULTS_DIR.joinpath("tables").mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(RESULTS_DIR / "tables" / "generation_results.csv", index=False)
    per_query_df.to_csv(RESULTS_DIR / "tables" / "generation_per_query.csv", index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
