"""
Evaluate the classical (no downloaded weights needed) generator baseline:
`ExtractiveClassifierGenerator` under three evidence conditions --
no_context / gold_context / bm25_context -- to isolate what retrieval
quality buys (or costs) end-to-end. See src/eval_generation.py for the
shared evaluation loop: run_generation_eval_llm.py calls the exact same
function, so this baseline and the real-LLM run in that script are
apples-to-apples (same test set, same conditions, same metrics).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import DATA_DIR, RESULTS_DIR, load_processed  # noqa: E402
from src.eval_generation import evaluate_generator  # noqa: E402
from src.generator import ExtractiveClassifierGenerator  # noqa: E402
from src.retrieval import BM25Retriever  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--n-test", type=int, default=None, help="evaluate only the first N")
    args = parser.parse_args()

    corpus, corpus_by_id, train, test = load_processed("train", "test", data_dir=args.data_dir)
    if args.n_test is not None:
        test = test[: args.n_test]

    bm25 = BM25Retriever([d["doc_id"] for d in corpus], [d["text"] for d in corpus])

    # Train the classifier-generator using each training query's OWN
    # (gold) abstract as evidence -- at train time we always have the
    # correct evidence; the point of the eval below is what happens at
    # test time when evidence instead comes from a retriever.
    gen = ExtractiveClassifierGenerator()
    gen.fit(
        [q["question"] for q in train],
        [corpus_by_id[q["doc_id"]]["text"] for q in train],
        [q["final_decision"] for q in train],
    )

    summary_df, per_query_df = evaluate_generator(
        gen, test, corpus_by_id, bm25, args.results_dir, figure_prefix="reliability"
    )
    tables = args.results_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(tables / "generation_results.csv", index=False)
    per_query_df.to_csv(tables / "generation_per_query.csv", index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
