# RAG_Eval

Two self-contained projects on **trustworthy evaluation of
retrieval-augmented question answering**: not just "does it answer
correctly," but does it retrieve the right evidence, is its confidence
calibrated, does it abstain when it should, and is the answer grounded in
what was actually retrieved.

| Project | Domain / data | Retrievers | Generators | Headline metrics |
|---|---|---|---|---|
| [`biomedical-rag-eval/`](biomedical-rag-eval/) | PubMedQA (1000 PubMed abstracts, yes/no/maybe) | BM25, TF-IDF+LSA | TF-IDF + logistic regression (offline) and Qwen2.5-1.5B-Instruct (GPU) through one shared evaluation loop | Recall@k / MRR with bootstrapped CIs, accuracy, ECE, Brier, reliability diagrams, lexical support ratio |
| [`sci-lit-rag-eval/`](sci-lit-rag-eval/) | 34 arXiv NLP/ML abstracts, 77 hand-written yes/no, extractive and unanswerable questions | BM25, TF-IDF+LSA | Cascaded logistic-regression classifiers + regex span extraction (no LLM, no GPU, cannot fabricate text) | Recall@k / MRR, accuracy, ECE, Brier, hallucination rate, false-abstention rate, evidence-grounding rate, negation-robustness probe, CPU efficiency |

Each project has its own README with the pipeline, the committed results,
and an honest reading of them (including where the benchmarks make
retrieval look easier than it would be on real user queries).

## Quick start

Both projects run fully offline on CPU in well under a minute:

```bash
cd biomedical-rag-eval
pip install -r requirements.txt
bash scripts/run_all.sh          # downloads PubMedQA, builds splits, evaluates
```

```bash
cd sci-lit-rag-eval
pip install -r requirements.txt
bash scripts/train.sh            # both retrievers + evaluation report
```

## Tests

Each project ships a `pytest` suite and a `ruff` configuration; the
workflow in `.github/workflows/tests.yml` runs both on every push against
Python 3.10 and 3.12.

```bash
pip install -r biomedical-rag-eval/requirements-dev.txt && (cd biomedical-rag-eval && pytest)
pip install -r sci-lit-rag-eval/requirements-dev.txt && (cd sci-lit-rag-eval && pytest)
```

## License

MIT (see `biomedical-rag-eval/LICENSE`). PubMedQA is used under its MIT
license; the arXiv abstracts in `sci-lit-rag-eval/data/` are quoted
verbatim from their public abstract pages, with IDs and provenance listed
in `sci-lit-rag-eval/data/PROVENANCE.md`.
