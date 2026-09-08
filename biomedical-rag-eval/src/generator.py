"""
Answer generation backends.

The default `ExtractiveClassifierGenerator` needs no downloaded model
weights, so it runs anywhere (including this offline sandbox) and still
produces a real predictive distribution to evaluate calibration honestly:
a logistic regression over TF-IDF question+evidence features, trained on
the train split, predicts P(yes/no/maybe). The extractive "answer" text
is the single evidence sentence with highest lexical overlap with the
question, plus an explicit citation to its source document and section.

`HFLocalLLMGenerator` is the documented upgrade path: a real instruction-
tuned LLM (e.g. Qwen2.5-1.5B-Instruct) reads the same retrieved evidence
and produces an abstractive answer with the same
(answer, confidence, evidence, doc_id) interface, so every evaluation
script in this repo works unchanged once it's plugged in. It requires
`transformers`/`torch` and network access to download weights, which this
environment does not have -- see README for how to run it on Colab.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .retrieval import tokenize


@dataclass
class GenerationResult:
    answer_text: str
    decision: str
    confidence: float
    class_probs: dict
    cited_doc_id: str
    cited_sentence: str


class ExtractiveClassifierGenerator:
    name = "extractive_tfidf_logreg"

    def __init__(self):
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False, min_df=2)
        self.clf = LogisticRegression(max_iter=2000, C=2.0)
        self.classes_ = None

    def fit(self, questions: list[str], evidences: list[str], labels: list[str]):
        X_text = [f"{q} [SEP] {e}" for q, e in zip(questions, evidences)]
        X = self.vectorizer.fit_transform(X_text)
        self.clf.fit(X, labels)
        self.classes_ = list(self.clf.classes_)

    def _best_sentence(self, question: str, sentences: list[str]) -> str:
        q_words = set(tokenize(question))
        if not sentences:
            return ""
        best, best_score = sentences[0], -1
        for s in sentences:
            overlap = len(q_words & set(tokenize(s)))
            if overlap > best_score:
                best, best_score = s, overlap
        return best

    def generate(
        self, question: str, doc_id: str, evidence_text: str, evidence_sentences: list[str]
    ) -> GenerationResult:
        X = self.vectorizer.transform([f"{question} [SEP] {evidence_text}"])
        probs = self.clf.predict_proba(X)[0]
        pred_idx = int(np.argmax(probs))
        decision = self.classes_[pred_idx]
        confidence = float(probs[pred_idx])
        best_sentence = self._best_sentence(question, evidence_sentences)
        answer_text = (
            f"{decision.capitalize()}. Evidence: \"{best_sentence}\" (source: {doc_id})"
        )
        return GenerationResult(
            answer_text=answer_text,
            decision=decision,
            confidence=confidence,
            class_probs=dict(zip(self.classes_, probs.tolist())),
            cited_doc_id=doc_id,
            cited_sentence=best_sentence,
        )


class HFLocalLLMGenerator:
    """Not runnable in this offline environment -- requires `pip install
    transformers torch` plus network access to download model weights
    (e.g. on Colab). Included so the pipeline's interface is upgrade-ready;
    every evaluation script consumes `GenerationResult`, not this class
    directly.
    """

    name = "hf_local_llm"

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "HFLocalLLMGenerator requires `pip install torch transformers` "
                "and network access to Hugging Face Hub. Run this on Colab; "
                "the offline sandbox this repo was drafted in has neither."
            ) from e
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    PROMPT_TEMPLATE = (
        "You are a careful clinical-literature assistant. Answer the question "
        "using ONLY the evidence below. Answer with exactly one of "
        "yes/no/maybe, a confidence between 0 and 1, and quote the sentence "
        "you relied on.\n\nEvidence (source {doc_id}):\n{evidence}\n\n"
        "Question: {question}\nAnswer:"
    )

    def generate(
        self, question: str, doc_id: str, evidence_text: str, evidence_sentences: list[str]
    ) -> GenerationResult:
        raise NotImplementedError(
            "Implement parsing of the model's structured output into a "
            "GenerationResult once running with real weights on Colab."
        )
