# sci-lit-rag-eval

Trustworthy retrieval-augmented QA over a real corpus of arXiv NLP/ML
papers, evaluated the way a deployment-facing RAG system should be:
not just "does it answer," but "does it know what it doesn't know,"
"is its confidence calibrated," "is the answer grounded in the right
evidence," and "what does correctness cost in retrieval choice and
compute."

```
question
   |
   v
retriever (BM25  or  TF-IDF + LSA)  --top-k evidence-->
   |
   v
cascaded classifier generator
   answerable? --no--> abstain ("unanswerable")
        |yes
        v
   yes/no or extractive? --extractive--> regex span extraction over
        |yes/no                          retrieved evidence
        v
   yes vs no
   |
   v
answer + evidence id + calibrated confidence
```

No LLM and no GPU are used anywhere in this pipeline. The "generator" is a
small cascade of `scikit-learn` logistic-regression classifiers plus a
regex-based span extractor, deliberately chosen so every produced answer
is either "abstain" or a literal substring of the retrieved evidence text
-- there is no free-text generation step that could fabricate content
outside what retrieval actually surfaced. This is a real trade-off
(discussed in Limitations) between a system that *cannot* hallucinate
free text and one that also cannot phrase an answer beyond copy-paste.

## Why this exists

This is a portfolio project built to demonstrate the same methodological
skill set as `biomedical-rag-eval` (retrieval-augmented generation,
trustworthy-LLM evaluation, and reproducible ML tooling) on a
general NLP/ML literature domain instead of a clinical one: BM25 vs.
dense-ish retrieval, calibration (ECE, reliability diagrams), faithfulness
/ evidence-grounding, hallucination-rate measurement, a lightweight
robustness probe, and a CPU-only efficiency comparison in place of GPU
quantization (see Limitations for why).

## Data

- **Corpus**: 34 real arXiv papers (title + full abstract, fetched
  verbatim from each paper's abstract page) across nine NLP/ML clusters --
  RAG, uncertainty/calibration, LLM quantization, hallucination detection,
  in-context learning, preference optimization (DPO/GRPO), dense
  retrieval, instruction tuning, and CoT distillation. See
  `data/PROVENANCE.md` for exactly how it was assembled.
- **QA benchmark**: 77 self-constructed questions (34 yes/no, 27
  extractive, 16 deliberately unanswerable). **This is not a published
  academic benchmark** -- it was hand-written against the real abstracts
  above, and that is stated plainly rather than implied otherwise. See
  `data/PROVENANCE.md` for the full methodology and its known bias
  (questions are lexically close to source text, which inflates retrieval
  scores relative to organic user queries).

## Pipeline

1. **Retrieval** (`src/retrieval.py`): `BM25Okapi` (sparse/lexical) vs.
   TF-IDF + truncated SVD / LSA (`sklearn`), a CPU-only "dense-ish"
   alternative that needs no neural encoder.
2. **Generation** (`src/generator.py`): a 3-stage cascade of
   `LogisticRegression` classifiers --
   (A) answerable vs. unanswerable,
   (B) yes/no vs. extractive,
   (C) yes vs. no --
   each using retrieval-derived features (top score, score margin,
   z-scored score, question/evidence lexical overlap, negation cues,
   question length, candidate-span count). Extractive answers are pulled
   via a regex acronym/method-name detector run over the top-retrieved
   passage, not generated freely.
3. **Trustworthy evaluation** (`src/evaluate_trust.py`): Expected
   Calibration Error + reliability diagrams, Brier score, hallucination
   rate (system answers when it should abstain), false-abstention rate
   (system abstains when it shouldn't), and an evidence-grounding rate
   (did the system's answer actually rely on the *correct* document, not
   just *some* retrieved document).
4. **Retrieval evaluation** (`src/retrieval_eval.py`): Recall@1/3/5 and
   MRR against gold arXiv IDs.
5. **Efficiency** (`src/efficiency.py`): index build time, per-query
   latency (mean/p95), and approximate memory footprint, CPU-only.
6. **Robustness probe** (in `evaluate.py`): a diagnostic negation-flip
   test -- does the yes/no classifier's answer flip when a question's
   logical polarity is negated? (It should, almost always; see results.)

All classifier metrics use 5-fold **stratified cross-validation with a
single fold split shared across all three cascade stages**, so a
Stage-A error (wrongly calling something answerable) can propagate into
Stage B/C exactly as it would at inference time -- this is an end-to-end,
error-propagating evaluation, not three independently-scored components.

## Running it

```bash
pip install -r requirements.txt
bash scripts/train.sh          # trains + evaluates both retrievers, writes results/
# or, for a single retriever:
python train.py --retriever bm25 --seed 42
python train.py --retriever tfidf_lsa --seed 42
python evaluate.py
```

`scripts/slurm.sh` is an `sbatch` wrapper around the same commands. The
job is deliberately submitted to a CPU partition with a 2-minute-scale
time budget -- see the comment in that file for why (this pipeline has no
GPU-bound step).

## Results

*(from `results/report.md`, regenerated by `evaluate.py`; seed 42, 5-fold CV, k=5)*

| Retriever | Recall@1 | Recall@3 | Recall@5 | MRR | QA acc. | ECE | Brier | Hallucination rate | False-abstention rate | Evidence-grounding rate |
|---|---|---|---|---|---|---|---|---|---|---|
| BM25 | 0.984 | 0.984 | 0.984 | 0.984 | 0.571 | 0.173 | 0.268 | 0.125 (2/16) | 0.180 (11/61) | 1.000 (50/50) |
| TF-IDF+LSA | 0.869 | 0.967 | 0.984 | 0.919 | 0.494 | 0.162 | 0.262 | 0.312 (5/16) | 0.295 (18/61) | 0.907 (39/43) |

**Efficiency (CPU-only):**

| Retriever | Build time (s) | Mean latency (ms) | P95 latency (ms) | Approx. memory (KB) |
|---|---|---|---|---|
| BM25 | 0.0057 | 0.131 | 0.193 | 384.5 |
| TF-IDF+LSA | 0.0204 | 0.682 | 0.782 | 197.6 |

**Robustness probe (negation flip test):** of 34 yes/no questions, negating
a question's polarity flipped the classifier's answer only **38.2%**
(13/34) of the time, trained on all data as a diagnostic (not a held-out
generalization estimate). A logically sound yes/no classifier should flip
on nearly all of these -- negating the question negates the correct
answer by construction. This low rate is a genuine, reproducible finding:
the classifier is picking up surface negation *cue words* (e.g. "not",
"only", "without") rather than tracking logical polarity, and the
diagnostic prompt used here (prepending "Is it NOT the case that...")
doesn't always trigger those same cue words. This is exactly the kind of
brittleness a trustworthy-evaluation report is supposed to surface, and it
is reported here rather than smoothed over.

### Reading the results honestly

- **Retrieval looks unusually strong** (BM25 Recall@1 = 0.984) because the
  benchmark questions were hand-written against the same abstracts they
  are supposed to retrieve, so several reuse the abstract's own
  terminology. This is disclosed in `data/PROVENANCE.md`; it is *not*
  representative of retrieval quality against organic user queries, and a
  real deployment would need a benchmark built independently of the
  corpus (e.g. QASPER-style, written by someone who only sees the paper
  title).
- **QA accuracy (57.1% / 49.4%) is modest, on purpose.** Because the
  generator only extracts literal substrings and answers yes/no from
  surface features (no semantic understanding), it fails on multi-word or
  paraphrased gold answers (e.g. "inter-problem, intra-problem, and
  intra-task generalization") and on yes/no questions whose negation isn't
  lexically marked. This is a ceiling effect of the deliberately
  LLM-free, non-fabricating design, not a bug.
- **Evidence-grounding rate (1.000 for BM25) is a genuinely useful
  number**: it shows that whenever BM25-backed retrieval was used, the
  document the generator actually leaned on for an answer was always the
  correct one on this benchmark -- but note this is measured only among
  *answered* (non-abstained) questions, so a system that abstains more is
  not automatically penalized here; hallucination rate and false-abstention
  rate are the metrics that catch over- and under-abstention respectively.
- **Retrieval choice measurably changes trustworthiness, not just
  accuracy**: swapping BM25 for TF-IDF-LSA raises the hallucination rate
  from 12.5% to 31.2% and the false-abstention rate from 18.0% to 29.5%,
  even though Recall@5 is identical (0.984) for both retrievers -- it's
  the *rank-1* retrieval quality (Recall@1: 0.984 vs 0.869) that the
  downstream classifier's confidence features are most sensitive to. This
  is the single clearest empirical result in this project: retrieval
  quality and trustworthy-generation quality are coupled in a way that a
  Recall@5-only retrieval eval would completely miss.

## Limitations

- **No neural retriever or LLM generator.** This sandbox had no GPU and
  no access to Hugging Face Hub / model downloads, so "dense retrieval" is
  approximated with classical TF-IDF+LSA rather than a sentence-transformer
  encoder, and "generation" is a classifier + regex-extraction cascade
  rather than an actual language model. The documented upgrade path (see
  `biomedical-rag-eval`'s equivalent note) is to swap in a
  sentence-transformer retriever and a small local LLM (e.g. Qwen2.5) via
  Colab, without changing the evaluation harness in `src/evaluate_trust.py`
  or `src/retrieval_eval.py` -- those operate on the same `(question,
  predicted_answer, confidence, top1_id)` record shape regardless of how
  it was produced.
- **The QA benchmark is small (77 items) and self-constructed**, not a
  peer-reviewed dataset -- see `data/PROVENANCE.md`. Numbers here
  characterize this pipeline's behavior, not the state of the art.
- **The corpus is small (34 papers)**, chosen for breadth across nine
  clusters rather than depth within one -- appropriate for a portfolio
  demonstration, not for a production literature-QA system.
- **The negation-robustness probe is a diagnostic, not a generalization
  estimate**: it trains on all 77 items with no held-out split, because
  its purpose is to interrogate what the fitted classifier is (not)
  sensitive to, not to estimate accuracy on new data.
