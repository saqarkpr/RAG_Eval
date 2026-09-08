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
    """Real instruction-tuned LLM generator. Not runnable in the offline
    sandbox this repo was drafted in (no GPU, no network access to
    Hugging Face Hub to download weights) -- run it on Colab instead:
    `pip install torch transformers accelerate`, then
    `python scripts/run_generation_eval_llm.py`.

    Confidence is NOT taken from the model's own verbalized "confidence: 0.8"
    (asking an LLM to introspect on its own calibration is known to be
    unreliable and gives no comparability to the classifier baseline's
    `predict_proba`). Instead we score the model's actual next-token
    log-probability for each of the three candidate decision words
    ("yes"/"no"/"maybe") as the immediate continuation of the prompt, and
    softmax across just those three to get a real predictive distribution
    -- directly comparable to `ExtractiveClassifierGenerator`'s
    `class_probs`, and to what the ECE/Brier code in `metrics.py` expects.
    A short separate generation call then asks the model to quote its
    supporting sentence, decoupling "what did it decide" (scored) from
    "what does it say it relied on" (generated), so a low-confidence
    decision can't hide behind a confident-sounding quote.
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
            model_name, torch_dtype=dtype
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

    def _build_prompt(self, question: str, evidence_text: str) -> str:
        evidence = (evidence_text or "(no evidence retrieved)")[: self.max_evidence_chars]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful biomedical-literature assistant. Base "
                    "your answer only on the evidence given. If the evidence "
                    "is insufficient or mixed, answer 'maybe'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Evidence:\n{evidence}\n\nQuestion: {question}\n"
                    "Answer with exactly one word: yes, no, or maybe."
                ),
            },
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
            with torch.no_grad():
                logits = self.model(full_ids).logits[0]
            logprob = 0.0
            prefix_len = prefix_ids.shape[1]
            for i, tok_id in enumerate(cont_ids):
                pos = prefix_len - 1 + i
                logprob += torch.log_softmax(logits[pos], dim=-1)[tok_id].item()
            scores.append(logprob)
        probs = torch.softmax(torch.tensor(scores), dim=0).tolist()
        return dict(zip(self.DECISION_WORDS, probs))

    def _quote_evidence(self, prompt: str, decision: str) -> str:
        torch = self._torch
        quote_prompt = (
            prompt
            + f" {decision}\nQuote the single sentence from the evidence above "
            "that most directly supports this answer:"
        )
        inputs = self.tokenizer(quote_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs, max_new_tokens=60, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate(
        self, question: str, doc_id: str, evidence_text: str, evidence_sentences: list[str]
    ) -> GenerationResult:
        prompt = self._build_prompt(question, evidence_text)
        probs = self._decision_distribution(prompt)
        decision = max(probs, key=probs.get)
        confidence = probs[decision]
        quoted = self._quote_evidence(prompt, decision) if evidence_text else ""
        answer_text = f"{decision.capitalize()}. Evidence: \"{quoted}\" (source: {doc_id})"
        return GenerationResult(
            answer_text=answer_text,
            decision=decision,
            confidence=confidence,
            class_probs=probs,
            cited_doc_id=doc_id,
            cited_sentence=quoted,
        )
