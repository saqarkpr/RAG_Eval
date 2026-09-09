"""
Same evaluation as run_generation_eval.py, with `HFLocalLLMGenerator` in
place of the classical baseline -- both scripts call the identical
`evaluate_generator` function in src/eval_generation.py, so this is
apples-to-apples with the baseline by construction, not by convention.

Run on Colab (GPU strongly recommended -- the model is scored 3x per
query for the decision distribution plus one short generation call per
query, so CPU will be slow):

    pip install torch transformers accelerate
    python scripts/run_generation_eval_llm.py --n-test 20   # smoke test first
    python scripts/run_generation_eval_llm.py                # full 200

Writes results/tables/generation_results_llm.csv and
results/figures/reliability_llm_{condition}.png.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.eval_generation import evaluate_generator
from src.generator import HFLocalLLMGenerator
from src.retrieval import BM25Retriever

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_jsonl(path):
    return [json.loads(line) for line in open(path)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=200, help="test queries to evaluate")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    corpus = load_jsonl(DATA_DIR / "corpus.jsonl")
    test = load_jsonl(DATA_DIR / "test.jsonl")[: args.n_test]
    corpus_by_id = {d["doc_id"]: d for d in corpus}

    doc_ids = [d["doc_id"] for d in corpus]
    texts = [d["text"] for d in corpus]
    bm25 = BM25Retriever(doc_ids, texts)

    print(f"Loading {args.model} ...")
    gen = HFLocalLLMGenerator(model_name=args.model)
    print(f"Loaded on device={gen.device}. Evaluating {len(test)} test queries.")

    summary_df, per_query_df = evaluate_generator(
        gen, test, corpus_by_id, bm25, RESULTS_DIR,
        figure_prefix="reliability_llm", progress_every=25,
    )
    summary_df["model"] = args.model
    RESULTS_DIR.joinpath("tables").mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(RESULTS_DIR / "tables" / "generation_results_llm.csv", index=False)
    per_query_df.to_csv(RESULTS_DIR / "tables" / "generation_per_query_llm.csv", index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
