# Biomedical RAG: Trustworthy Evaluation

A retrieval-augmented question-answering pipeline over PubMed abstracts,
evaluated the way a *deployment* decision requires evidence to be
evaluated — not just "does it answer correctly," but does it retrieve the
right evidence, is its confidence trustworthy, and is its answer actually
grounded in what it retrieved.

This is a from-scratch, fully offline, fully reproducible pipeline: no
pretrained embedding or language model weights are downloaded anywhere in
the default path. The retriever is built from TF-IDF/BM25/LSA primitives
and the generator is a classifier trained on the same data it's evaluated
on. That's a deliberate scope choice (see [Limitations](#limitations)),
not an oversight — the point of this repo is the *evaluation harness* and
what it reveals, which is designed to keep working unmodified once a real
instruction-tuned LLM is dropped in as the generator (interface documented
in `src/generator.py`).

## Task and data

[PubMedQA](https://github.com/pubmedqa/pubmedqa) (Jin et al., 2019, MIT
license), labeled subset (PQA-L): 1000 PubMed abstracts, each reframed as
a yes/no/maybe research question (derived from the paper's own title) with
expert-annotated answers and the abstract's own conclusion as a long-form
reference answer. We build our own document-level 60/20/20 train/dev/test
split (seed 42) rather than reusing the benchmark's official split, since
that split was designed for classification-only evaluation; retrieval
evaluation additionally needs the other 999 abstracts in the corpus to
serve as distractors for whichever one is the true source document per
question.

```
scripts/download_data.py   # fetches ori_pqal.json from the public GitHub repo
scripts/prepare_data.py    # builds corpus.jsonl + train/dev/test.jsonl
```

## Pipeline

```
question ─▶ retriever ─▶ top-k documents ─▶ generator ─▶ answer + confidence + citation
```

**Retrieval** (`src/retrieval.py`): two independent methods, so retrieval
quality itself is a measured variable rather than assumed:
- `BM25Retriever` — classic lexical ranking.
- `LsaRetriever` — TF-IDF projected through truncated SVD (latent semantic
  analysis), fit on the corpus itself. This is the from-scratch stand-in
  for a neural dense/bi-encoder retriever.

**Generation** (`src/generator.py`): `ExtractiveClassifierGenerator` is a
TF-IDF + logistic-regression classifier over `question [SEP] evidence`,
trained on the train split, producing a probability over {yes, no, maybe}
plus an extracted evidence sentence and an explicit source citation. It
needs no downloaded weights, so every metric below is a real, run-today
number rather than a projection.

**Evaluation** (`src/metrics.py`, `scripts/run_*_eval.py`):
- Retrieval: Recall@{1,3,5,10}, MRR, indexing time, per-query latency —
  each with a bootstrapped 95% CI (1000 resamples), because a point
  estimate on 200 test queries isn't enough to call a winner.
- Generation, under three evidence conditions (no context / gold context /
  BM25-retrieved context, to isolate what retrieval errors cost
  downstream): decision accuracy, Expected Calibration Error, multiclass
  Brier score, and a reliability diagram per condition.
- Faithfulness: a lexical *support ratio* — the fraction of the generated
  answer's content words that also appear in the cited evidence — as a
  cheap, model-free hallucination-risk proxy (see limitations: it is not
  an entailment check).

## Results

Reproduced by `bash scripts/run_all.sh`. n=200 test questions, 1000-document
corpus, seed 42.

**Retrieval** (`results/tables/retrieval_results.csv`):

| retriever | Recall@1 | Recall@5 | MRR | mean latency |
|---|---|---|---|---|
| BM25 | 0.960 [0.930, 0.985] | 0.990 [0.975, 1.00] | 0.974 [0.953, 0.991] | 2.8 ms |
| TF-IDF+LSA (128d) | 0.720 [0.655, 0.785] | 0.900 [0.860, 0.940] | 0.797 [0.750, 0.847] | 1.0 ms |

BM25 wins decisively and with non-overlapping CIs. This is expected, not a
bug: each question is generated from its own abstract's title, so
question and gold document share exact vocabulary — the setting lexical
retrieval is built for. It's still worth measuring rather than assuming:
a semantic/dense retriever's real advantage shows up on paraphrased or
vocabulary-mismatched queries, which this benchmark's question-generation
process doesn't produce. That's a property of the *benchmark*, and a
concrete reason a production system should also be evaluated on queries
written independently of the source text (see Limitations).

**Generation** (`results/tables/generation_results.csv`):

| condition | accuracy | ECE | Brier | mean confidence |
|---|---|---|---|---|
| no context | 0.520 | 0.079 | 0.602 | 0.565 |
| gold context | 0.510 | 0.102 | 0.609 | 0.612 |
| BM25-retrieved context | 0.520 | 0.105 | 0.607 | 0.615 |

The test set's majority-class rate (always predicting "yes") is 0.505 —
so this classifier is, within noise, not beating the majority baseline
*in any condition, including with perfect (gold) retrieval*. That is the
headline finding, and it's a real negative result, not a bug: PubMedQA's
yes/no/maybe task requires weighing quantitative findings against a
question's framing, which a bag-of-words classifier structurally cannot
do — more/better-retrieved evidence doesn't help a model that cannot use
it. It also gets *more* confident with more context (0.565 → 0.615 mean
confidence) while accuracy is flat, which is exactly the overconfidence
pattern a calibration check exists to catch: ECE roughly doubles from
0.079 to ~0.10 once any evidence is added. This is the central argument
for the `HFLocalLLMGenerator` upgrade path below — the retrieval side of
this pipeline is already solved (BM25 MRR 0.97), so the accuracy ceiling
here is entirely a generator-capability problem, and this harness is
built to keep measuring exactly the same way once the generator can
actually reason over its evidence.

Mean support ratio (generated answer vs. cited evidence) is ~0.90 in both
context conditions. For an extractive generator this is close to true by
construction (the answer *is* a sentence pulled from the evidence) — the
metric is included because it's the same one that becomes a real,
non-trivial faithfulness/hallucination check once the generator is
abstractive.

## Reproducing

```bash
pip install -r requirements.txt
bash scripts/run_all.sh
```

Runs in well under a minute on CPU. Outputs land in `results/tables/*.csv`
and `results/figures/reliability_*.png`.

## Upgrade path: a real LLM generator

`src/generator.py` defines `HFLocalLLMGenerator` with the same
`(question, doc_id, evidence_text, evidence_sentences) -> GenerationResult`
interface as the extractive baseline, so every script in `scripts/` works
unchanged once it's implemented against real weights. It isn't runnable in
the environment this repo was built in (no network access to download
model weights), so it's left as a documented stub rather than faked. To
finish it (e.g. on Colab with a T4 GPU, using an open instruct model such
as Qwen2.5-1.5B-Instruct):

This is implemented — `HFLocalLLMGenerator` scores the model's actual
next-token log-probability for each of "yes"/"no"/"maybe" as the
immediate continuation of the prompt and softmaxes across just those
three (rather than trusting a verbalized "confidence: 0.8", which is
known to be unreliable), then asks for a short separate generation to
quote the supporting evidence. Run it with:

```bash
pip install torch transformers accelerate
python scripts/run_generation_eval_llm.py --n-test 20   # smoke test
python scripts/run_generation_eval_llm.py                # full 200
```

It calls the exact same `evaluate_generator` function
(`src/eval_generation.py`) that the classical baseline above does — same
test set, same three conditions, same metric code — so the two result
tables are comparable apples-to-apples by construction, not by
convention.

This is also where genuine hallucination detection becomes possible: an
abstractive model can introduce claims with no support in the evidence at
all, which the support-ratio metric is designed to catch but the
extractive baseline can't exercise.

## Limitations

- **Generator is a classical, non-neural baseline.** This is the repo's
  central, stated limitation, not a hidden one — see Results above for
  why swapping in a real LLM is the natural next step and exactly what
  should change when you do.
- **Retrieval task is easier than production retrieval.** Questions are
  generated from their own source abstract's title, so lexical overlap
  with the gold document is unusually high; BM25's 0.96 Recall@1 should
  not be read as "retrieval is a solved problem" for queries phrased
  independently of their source text (e.g. a clinician's actual question).
- **Support ratio is a lexical proxy, not an entailment check.** It
  penalizes faithful paraphrases that don't reuse the evidence's exact
  wording, and would miss a high-overlap sentence that actually
  contradicts its evidence. A real faithfulness metric for the LLM
  generator should add NLI-based entailment checking.
- **Fairness/subgroup evaluation is out of scope here.** The corpus has no
  patient-level demographic metadata to stratify by (it's abstract-level
  literature, not patient records), so this axis of "trustworthy
  evaluation" is deliberately left to a dataset where it's answerable
  rather than reported against data that can't support it.
- **1000-document corpus is small** relative to a real literature or
  clinical-note retrieval setting; latency numbers here are not
  informative about scaling to millions of documents.

## Repository layout

```
scripts/
  download_data.py       fetch the public PubMedQA labeled subset
  prepare_data.py         build corpus + train/dev/test splits
  run_retrieval_eval.py   BM25 vs. TF-IDF+LSA retrieval evaluation
  run_generation_eval.py  generation accuracy/calibration/faithfulness eval
  run_all.sh              run the full pipeline end-to-end
src/
  retrieval.py       BM25Retriever, LsaRetriever
  generator.py       ExtractiveClassifierGenerator, HFLocalLLMGenerator
  metrics.py         recall@k, MRR, ECE, Brier, reliability bins, support ratio
  eval_generation.py shared no/gold/bm25-context eval loop used by BOTH
                     run_generation_eval.py and run_generation_eval_llm.py,
                     so a baseline-vs-LLM comparison is apples-to-apples by
                     construction (one evaluation implementation, not two
                     that could silently drift apart)
results/
  tables/     retrieval_results.csv, generation_results.csv (committed)
  figures/    reliability_{condition}.png (committed)
data/         gitignored; regenerate with scripts/download_data.py + prepare_data.py
```

## Data and license

Data: [PubMedQA](https://pubmed.ncbi.nlm.nih.gov/) labeled subset, via the
[pubmedqa/pubmedqa](https://github.com/pubmedqa/pubmedqa) GitHub
repository (MIT license). If you use PubMedQA, please cite:
Jin, Q., Dhingra, B., Liu, Z., Cohen, W., & Lu, X. (2019). *PubMedQA: A
Dataset for Biomedical Research Question Answering.* EMNLP-IJCNLP 2019.

Code in this repository: MIT license.
