# Data provenance

## Corpus (`corpus.json`)

34 real arXiv papers (titles + full abstracts, fetched verbatim from each
paper's `arxiv.org/abs/<id>` page) spanning nine NLP/ML topic clusters:
retrieval-augmented generation, uncertainty/calibration, LLM quantization
and inference efficiency, hallucination detection, in-context learning,
preference optimization (DPO/GRPO/RLHF), dense retrieval, instruction
tuning, and chain-of-thought distillation. No medical, clinical, or
biomedical content is included by design.

Papers were located via targeted topic searches and abstracts were pulled
directly from each paper's official arXiv abstract page. arXiv IDs and
titles are real and independently verifiable; this is a genuine (if small)
snapshot of the literature, not a synthetic corpus.

## QA benchmark (`qa_benchmark.json`)

**This benchmark is self-constructed, not a published academic dataset.**
Unlike PubMedQA (used in the companion `biomedical-rag-eval` project),
there is no equivalent public, license-clear QA dataset over an
arbitrary arXiv abstract subset assembled specifically for this project,
so all 77 questions were written by hand against the real abstracts above,
following the same evidence-grounded design as QASPER / PubMedQA:

- 34 yes/no questions, each tied to one specific paper's abstract, with an
  unambiguous ground-truth answer derivable from the text (roughly balanced:
  15 "Yes" / 19 "No").
- 27 extractive questions asking for the specific method/framework name (or
  a short factual span) introduced in a paper; gold answers are short spans
  drawn directly from that paper's title/abstract. Only non-survey papers
  (27 of 34) received an extractive question, since a survey does not
  "introduce a method."
- 16 deliberately **unanswerable** questions: plausible-sounding, specific
  factual questions (exact metrics, hyperparameters, hardware, licenses,
  venues, human-annotation statistics) that are *not* stated in any
  abstract in the corpus. These test whether the system abstains instead
  of confabulating a confident answer -- the core "hallucination rate"
  metric in `results/report.md`.

Because several yes/no and extractive questions reuse phrasing close to the
source abstract (a natural consequence of hand-writing single-document
factual questions quickly), lexical retrievers do unusually well on this
benchmark compared to open-domain QA benchmarks built from independently
authored user queries. This is flagged explicitly as a limitation in the
top-level README rather than hidden.
