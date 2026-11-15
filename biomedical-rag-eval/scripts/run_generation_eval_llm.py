"""
Same evaluation as run_generation_eval.py, with `HFLocalLLMGenerator` in
place of the classical baseline -- both scripts call the identical
`evaluate_generator` function in src/eval_generation.py, so this is
apples-to-apples with the baseline by construction, not by convention.

Run on Colab or any GPU machine (the model is scored once per query for
the decision distribution, plus one extra pass per multi-token surface
form and one short generation call per query for the citation, so CPU
will be slow):

    pip install -r requirements-llm.txt
    python scripts/run_generation_eval_llm.py --n-test 20   # smoke test first
    python scripts/run_generation_eval_llm.py                # full 200

Writes results/tables/generation_results_llm.csv,
results/tables/generation_per_query_llm.csv and
results/figures/reliability_llm_{condition}.png.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import DATA_DIR, RESULTS_DIR, load_processed  # noqa: E402
from src.eval_generation import evaluate_generator  # noqa: E402
from src.generator import HFLocalLLMGenerator  # noqa: E402
from src.retrieval import BM25Retriever  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=200, help="test queries to evaluate")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu (auto if omitted)")
    parser.add_argument("--max-evidence-chars", type=int, default=2000)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    corpus, corpus_by_id, test = load_processed("test", data_dir=args.data_dir)
    test = test[: args.n_test]
    bm25 = BM25Retriever([d["doc_id"] for d in corpus], [d["text"] for d in corpus])

    print(f"Loading {args.model} ...")
    gen = HFLocalLLMGenerator(
        model_name=args.model, device=args.device, max_evidence_chars=args.max_evidence_chars
    )
    print(f"Loaded on device={gen.device}. Evaluating {len(test)} test queries.")

    summary_df, per_query_df = evaluate_generator(
        gen, test, corpus_by_id, bm25, args.results_dir,
        figure_prefix="reliability_llm", progress_every=25,
    )
    summary_df["model"] = args.model
    tables = args.results_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(tables / "generation_results_llm.csv", index=False)
    per_query_df.to_csv(tables / "generation_per_query_llm.csv", index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
