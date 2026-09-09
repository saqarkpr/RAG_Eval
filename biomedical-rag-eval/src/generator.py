"""
Answer generation backends.

The default `ExtractiveClassifierGenerator` needs no downloaded model
weights, so it runs anywhere (including this offline sandbox) and still
produces a real predictive distribution to evaluate calibration honestly:
a logistic regression over TF-IDF question+evidence features, trained on
the train split, predicts P(yes/no/maybe). The extractive "answer" text
is the single evidence sentence with highest lexical overlap with the
question, plus an explicit citation to its source document and section.

`HFLocalLLMGenerator` is the real-LLM upgrade: a genuine instruction-tuned
model (e.g. Qwen2.5-1.5B-Instruct) reads the same retrieved evidence and
produces an abstractive answer through the same
(answer, confidence, evidence, doc_id) `GenerationResult` interface, so
`scripts/run_generation_eval_llm.py` reuses every metric in `metrics.py`
unchanged. It requires `transformers`/`torch` and network access to
download weights, which this repo's offline sandbox does not have --
run it on Colab (see README).
"""
import re
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


def best_matching_sentence(query_words: set, sentences: list[str]) -> str:
    """Lexical-overlap fallback: the sentence sharing the most tokenized
    words with `query_words`. Guarantees a verbatim sentence from
    `sentences` is returned (never a model-hallucinated paraphrase),
    which is what makes citations checkable at all.
    """
    if not sentences:
        return ""
    best, best_score = sentences[0], -1
    for s in sentences:
        overlap = len(query_words & set(tokenize(s)))
        if overlap > best_score:
            best, best_score = s, overlap
    return best


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

    def generate(
        self, question: str, doc_id: str, evidence_text: str, evidence_sentences: list[str]
    ) -> GenerationResult:
        X = self.vectorizer.transform([f"{question} [SEP] {evidence_text}"])
        probs = self.clf.predict_proba(X)[0]
        pred_idx = int(np.argmax(probs))
        decision = self.classes_[pred_idx]
        confidence = float(probs[pred_idx])
        best_sentence = best_matching_sentence(set(tokenize(question)), evidence_sentences)
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
    """Real instruction-tuned LLM generator. Not runnable in the offline
    sandbox this repo was drafted in (no GPU, no network access to
    Hugging Face Hub to download weights) -- run it on Colab instead:
    `pip install torch transformers accelerate`, then
    `python scripts/run_generation_eval_llm.py`.

    Two design choices exist specifically to keep this generator
    trustworthy-by-construction rather than just prompting an LLM and
    hoping:

    1. Confidence is NOT the model's own verbalized "confidence: 0.8"
       (asking an LLM to introspect on its own calibration is known to be
       unreliable). Instead we score the model's actual next-token
       log-probability for each of the three candidate decision words
       ("yes"/"no"/"maybe") as the immediate continuation of the prompt,
       and softmax across just those three. This is best described as a
       *renormalized candidate-token probability*, not a calibrated
       probability in the general sense -- it says how the model weighs
       these three specific continuations against each other, not that
       the resulting number is a well-calibrated estimate of correctness
       (that calibration is exactly what ECE/Brier in metrics.py then
       measure empirically).
    2. The citation is NEVER free-form LLM generation. An earlier version
       of this class asked the model to generate its supporting quote
       directly, which risks a fluent quote that doesn't actually appear
       in the evidence -- exactly the failure mode a "trustworthy
       evaluation" project shouldn't wave through. Instead the model
       picks a sentence NUMBER from a numbered list of the real evidence
       sentences, which is mapped back to that exact string; if parsing
       the number fails, a lexical-overlap fallback (never the model's
       own free text) is used instead. The cited sentence is therefore
       always verbatim from the evidence or empty, by construction.
    """

    name = "hf_local_llm"
    DECISION_WORDS = ["yes", "no", "maybe"]

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str | None = None,
        max_evidence_chars: int = 2000,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "HFLocalLLMGenerator requires `pip install torch transformers "
                "accelerate` and network access to Hugging Face Hub. Run this "
                "on Colab; the offline sandbox this repo was drafted in has "
                "neither."
            ) from e
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_evidence_chars = max_evidence_chars
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # bfloat16 needs Ampere-or-newer hardware (A100, RTX30xx+); a T4
        # (Turing, the free Colab GPU) has no bf16 tensor cores and either
        # errors or silently runs it much slower via emulation. float16
        # is what T4 actually supports well, so pick it dynamically rather
        # than hardcoding bf16 and assuming an A100.
        if self.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype  # `dtype=` (transformers renamed torch_dtype=)
        ).to(self.device)
        self.model.eval()
        # Precompute each decision word's token ids once; most tokenizers
        # encode a leading-space word differently from a bare word, and the
        # chat template always leaves a space before the answer, so we
        # tokenize " yes" / " no" / " maybe" to match how it actually
        # appears mid-generation.
        self._decision_token_ids = {
            w: self.tokenizer(" " + w, add_special_tokens=False)["input_ids"]
            for w in self.DECISION_WORDS
        }

    def _truncate_evidence(self, evidence_text: str) -> str:
        """Truncate to max_evidence_chars WITHOUT cutting a sentence in
        half: accumulate whole sentences (split on '. ') until the next
        one would exceed the budget. Falls back to a hard character cut
        only if a single sentence alone exceeds the whole budget.
        """
        if len(evidence_text) <= self.max_evidence_chars:
            return evidence_text
        sentences = re.split(r"(?<=[.!?])\s+", evidence_text)
        kept, length = [], 0
        for s in sentences:
            if length + len(s) + 1 > self.max_evidence_chars and kept:
                break
            kept.append(s)
            length += len(s) + 1
        truncated = " ".join(kept)
        return truncated if truncated else evidence_text[: self.max_evidence_chars]

    def _build_prompt(self, question: str, evidence_text: str) -> str:
        has_evidence = bool(evidence_text)
        if has_evidence:
            evidence = self._truncate_evidence(evidence_text)
            system = (
                "You are a careful biomedical-literature assistant. Base "
                "your answer only on the evidence given. If the evidence "
                "is insufficient or mixed, answer 'maybe'."
            )
            user = (
                f"Evidence:\n{evidence}\n\nQuestion: {question}\n"
                "Answer with exactly one word: yes, no, or maybe."
            )
        else:
            # No evidence at all is a different question ("what does the
            # model's own prior say?") from "the evidence you were given
            # is ambiguous" -- reusing the evidence-present system prompt
            # here would tell the model evidence is missing/insufficient
            # and then also instruct it to answer 'maybe' whenever
            # evidence is insufficient, collapsing every no-evidence
            # answer to a forced 'maybe' regardless of what the model
            # actually believes. Keep this branch free of any
            # insufficient-evidence cue.
            system = (
                "You are a careful biomedical-literature assistant. Answer "
                "using your own biomedical knowledge."
            )
            user = (
                f"Question: {question}\n"
                "Answer with exactly one word: yes, no, or maybe."
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _decision_distribution(self, prompt: str) -> dict:
        torch = self._torch
        prefix_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        scores = []
        for word in self.DECISION_WORDS:
            cont_ids = self._decision_token_ids[word]
            full_ids = torch.cat(
                [prefix_ids[0], torch.tensor(cont_ids, device=self.device)]
            ).unsqueeze(0)
            with torch.inference_mode():
                logits = self.model(full_ids).logits[0]
            logprob = 0.0
            prefix_len = prefix_ids.shape[1]
            for i, tok_id in enumerate(cont_ids):
                pos = prefix_len - 1 + i
                logprob += torch.log_softmax(logits[pos].float(), dim=-1)[tok_id].item()
            scores.append(logprob)
        probs = torch.softmax(torch.tensor(scores), dim=0).tolist()
        return dict(zip(self.DECISION_WORDS, probs))

    def _select_cited_sentence(
        self, prompt: str, decision: str, evidence_sentences: list[str], question: str
    ) -> str:
        """Ask the model to pick a sentence NUMBER (never free text) from
        the real evidence sentences, and map that number back to the
        exact string. Falls back to lexical overlap if the model doesn't
        return a valid in-range number, so the cited sentence is always
        verbatim -- the model can pick badly, but it cannot fabricate a
        quote that doesn't exist in the evidence.
        """
        if not evidence_sentences:
            return ""
        torch = self._torch
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(evidence_sentences))
        select_prompt = (
            prompt
            + f" {decision}\n\nEvidence sentences:\n{numbered}\n\n"
            "Which numbered sentence above most directly supports this "
            "answer? Reply with only the number."
        )
        inputs = self.tokenizer(select_prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out_ids = self.model.generate(
                **inputs, max_new_tokens=6, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out_ids[0][inputs["input_ids"].shape[1] :]
        reply = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        match = re.search(r"\d+", reply)
        if match:
            idx = int(match.group()) - 1
            if 0 <= idx < len(evidence_sentences):
                return evidence_sentences[idx]
        return best_matching_sentence(set(tokenize(question)), evidence_sentences)

    def generate(
        self, question: str, doc_id: str, evidence_text: str, evidence_sentences: list[str]
    ) -> GenerationResult:
        prompt = self._build_prompt(question, evidence_text)
        probs = self._decision_distribution(prompt)
        decision = max(probs, key=probs.get)
        confidence = probs[decision]
        cited_sentence = self._select_cited_sentence(
            prompt, decision, evidence_sentences, question
        )
        answer_text = (
            f"{decision.capitalize()}. Evidence: \"{cited_sentence}\" (source: {doc_id})"
        )
        return GenerationResult(
            answer_text=answer_text,
            decision=decision,
            confidence=confidence,
            class_probs=probs,
            cited_doc_id=doc_id,
            cited_sentence=cited_sentence,
        )
