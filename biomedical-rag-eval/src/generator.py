"""
Answer generation backends.

The default `ExtractiveClassifierGenerator` needs no downloaded model
weights, so it runs anywhere (including an offline sandbox) and still
produces a real predictive distribution to evaluate calibration honestly:
a logistic regression over TF-IDF question+evidence features, trained on
the train split, predicts P(yes/no/maybe). The extractive "answer" text
is the single evidence sentence with highest lexical overlap with the
question, plus an explicit citation to its source document.

`HFLocalLLMGenerator` is the real-LLM upgrade: a genuine instruction-tuned
model (e.g. Qwen2.5-1.5B-Instruct) reads the same retrieved evidence and
answers through the same `GenerationResult` interface, so
`scripts/run_generation_eval_llm.py` reuses every metric in `metrics.py`
unchanged. It requires `transformers`/`torch` and network access to
download weights -- run it on Colab or any GPU machine (see README).

Everything that does not need torch (prompt construction, evidence
truncation, candidate-token enumeration) lives in module-level functions
so it can be unit-tested without installing torch.
"""
import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .retrieval import tokenize

DECISION_WORDS = ["yes", "no", "maybe"]


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


def format_answer(decision: str, cited_sentence: str, doc_id: str) -> str:
    return f'{decision.capitalize()}. Evidence: "{cited_sentence}" (source: {doc_id})'


class ExtractiveClassifierGenerator:
    name = "extractive_tfidf_logreg"

    def __init__(self, C: float = 2.0, min_df: int = 2):
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize, lowercase=False, token_pattern=None, min_df=min_df
        )
        self.clf = LogisticRegression(max_iter=2000, C=C)
        self.classes_ = None

    def fit(self, questions: list[str], evidences: list[str], labels: list[str]):
        if not (len(questions) == len(evidences) == len(labels)):
            raise ValueError("questions, evidences and labels must have the same length")
        X_text = [f"{q} [SEP] {e}" for q, e in zip(questions, evidences)]
        X = self.vectorizer.fit_transform(X_text)
        self.clf.fit(X, labels)
        self.classes_ = list(self.clf.classes_)
        return self

    def generate(
        self, question: str, doc_id: str, evidence_text: str, evidence_sentences: list[str]
    ) -> GenerationResult:
        if self.classes_ is None:
            raise RuntimeError("ExtractiveClassifierGenerator.generate() called before fit()")
        X = self.vectorizer.transform([f"{question} [SEP] {evidence_text}"])
        probs = self.clf.predict_proba(X)[0]
        pred_idx = int(np.argmax(probs))
        decision = self.classes_[pred_idx]
        confidence = float(probs[pred_idx])
        best_sentence = best_matching_sentence(set(tokenize(question)), evidence_sentences)
        return GenerationResult(
            answer_text=format_answer(decision, best_sentence, doc_id),
            decision=decision,
            confidence=confidence,
            class_probs=dict(zip(self.classes_, probs.tolist())),
            cited_doc_id=doc_id,
            cited_sentence=best_sentence,
        )


# ---------------------------------------------------------------------------
# LLM generator: torch-free helpers (unit-tested) ---------------------------
# ---------------------------------------------------------------------------

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")

SYSTEM_WITH_EVIDENCE = (
    "You are a careful biomedical-literature assistant. Base your answer "
    "only on the evidence given. If the evidence is insufficient or mixed, "
    "answer 'maybe'."
)
# No evidence at all is a different question ("what does the model's own
# prior say?") from "the evidence you were given is ambiguous". Reusing the
# evidence-present system prompt here would tell the model evidence is
# missing and then also instruct it to answer 'maybe' whenever evidence is
# insufficient, collapsing every no-evidence answer to a forced 'maybe'
# regardless of what the model actually believes (this happened in the
# first LLM run; see README). Keep this branch free of any
# insufficient-evidence cue.
SYSTEM_NO_EVIDENCE = (
    "You are a careful biomedical-literature assistant. Answer using your "
    "own biomedical knowledge."
)
ANSWER_INSTRUCTION = "Answer with exactly one word: yes, no, or maybe."


def truncate_evidence(evidence_text: str, max_chars: int) -> str:
    """Truncate to `max_chars` WITHOUT cutting a sentence in half:
    accumulate whole sentences until the next one would exceed the budget.
    Falls back to a hard character cut only if the first sentence alone
    exceeds the whole budget.
    """
    if len(evidence_text) <= max_chars:
        return evidence_text
    kept, length = [], 0
    for s in _SENTENCE_BOUNDARY_RE.split(evidence_text):
        if length + len(s) + 1 > max_chars and kept:
            break
        kept.append(s)
        length += len(s) + 1
    truncated = " ".join(kept)
    return truncated if len(truncated) <= max_chars else evidence_text[:max_chars]


def build_messages(question: str, evidence_text: str, max_evidence_chars: int = 2000) -> list:
    """Chat messages for the decision prompt. Two different system prompts
    are used depending on whether any evidence is present (see
    SYSTEM_NO_EVIDENCE above for why)."""
    if evidence_text:
        evidence = truncate_evidence(evidence_text, max_evidence_chars)
        system = SYSTEM_WITH_EVIDENCE
        user = f"Evidence:\n{evidence}\n\nQuestion: {question}\n{ANSWER_INSTRUCTION}"
    else:
        system = SYSTEM_NO_EVIDENCE
        user = f"Question: {question}\n{ANSWER_INSTRUCTION}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def citation_messages(decision_messages: list, decision: str, evidence_sentences: list) -> list:
    """Follow-up chat turn asking the model to pick a sentence NUMBER (never
    free text) from the real evidence sentences. The model's own decision
    is replayed as its previous assistant turn so the follow-up is a
    well-formed conversation rather than text pasted after the generation
    prompt."""
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(evidence_sentences))
    return decision_messages + [
        {"role": "assistant", "content": decision},
        {
            "role": "user",
            "content": (
                f"Evidence sentences:\n{numbered}\n\n"
                "Which numbered sentence above most directly supports your "
                "answer? Reply with only the number."
            ),
        },
    ]


def parse_sentence_choice(reply: str, n_sentences: int) -> int | None:
    """Return the 0-based index chosen in `reply`, or None if the reply does
    not contain a valid in-range number."""
    match = re.search(r"\d+", reply)
    if not match:
        return None
    idx = int(match.group()) - 1
    return idx if 0 <= idx < n_sentences else None


def candidate_token_sequences(tokenizer, word: str) -> list[list[int]]:
    """Distinct token-id sequences for the surface forms of `word` the model
    may emit as its answer: lowercase and capitalized, with and without a
    leading space. Which form is "natural" depends on the chat template
    (ChatML ends the generation prompt with a newline, so the first answer
    token has no leading space; other templates leave a space) and on
    whether the model prefers "Yes" to "yes", so all forms are scored and
    their probabilities summed per word."""
    seqs: list[list[int]] = []
    for form in (word, word.capitalize()):
        for text in (form, " " + form):
            ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
            if ids and ids not in seqs:
                seqs.append(ids)
    return seqs


class HFLocalLLMGenerator:
    """Real instruction-tuned LLM generator. Needs `pip install -r
    requirements-llm.txt` and Hugging Face Hub access to download weights;
    run it on Colab or any GPU machine with
    `python scripts/run_generation_eval_llm.py`.

    Two design choices exist specifically to keep this generator
    trustworthy-by-construction rather than just prompting an LLM and
    hoping:

    1. Confidence is NOT the model's own verbalized "confidence: 0.8"
       (asking an LLM to introspect on its own calibration is known to be
       unreliable). Instead we score the model's actual next-token
       log-probability for each of the three candidate decision words
       ("yes"/"no"/"maybe", summed over their lowercase/capitalized and
       space/no-space surface forms) as the immediate continuation of the
       prompt, and softmax across just those three. This is best described
       as a *renormalized candidate-token probability*, not a calibrated
       probability in the general sense -- it says how the model weighs
       these three continuations against each other, not that the number
       is a well-calibrated estimate of correctness (that calibration is
       exactly what ECE/Brier in metrics.py then measure empirically).
    2. The citation is NEVER free-form LLM generation. The model picks a
       sentence NUMBER from a numbered list of the real evidence sentences,
       which is mapped back to that exact string; if parsing the number
       fails, a lexical-overlap fallback (never the model's own free text)
       is used instead. The cited sentence is therefore always verbatim
       from the evidence or empty, by construction.
    """

    name = "hf_local_llm"
    DECISION_WORDS = DECISION_WORDS

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
                "HFLocalLLMGenerator requires `pip install -r requirements-llm.txt` "
                "and network access to Hugging Face Hub (run it on Colab or a GPU machine)."
            ) from e
        self._torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_evidence_chars = max_evidence_chars
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # bfloat16 needs Ampere-or-newer hardware (A100, RTX30xx+); a T4
        # (Turing, the free Colab GPU) has no bf16 tensor cores and either
        # errors or silently runs it much slower via emulation. float16 is
        # what T4 actually supports well, so pick it dynamically.
        if self.device.startswith("cuda"):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        except TypeError:  # transformers < 4.56 only knows the old keyword
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model = model.to(self.device)
        self.model.eval()
        self._decision_variants = {
            w: candidate_token_sequences(self.tokenizer, w) for w in self.DECISION_WORDS
        }

    # -- prompting ---------------------------------------------------------
    def _render(self, messages: list) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _encode(self, prompt: str):
        # The rendered chat template already contains any special tokens
        # (BOS, role markers), so do not let the tokenizer add a second set.
        return self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(
            self.device
        )

    # -- decision ----------------------------------------------------------
    def _decision_distribution(self, prompt: str) -> dict:
        torch = self._torch
        prefix_ids = self._encode(prompt)
        prefix_len = prefix_ids.shape[1]
        with torch.inference_mode():
            next_logprobs = torch.log_softmax(
                self.model(prefix_ids).logits[0, -1].float(), dim=-1
            )
        word_scores = []
        for word in self.DECISION_WORDS:
            variant_logprobs = []
            for seq in self._decision_variants[word]:
                logprob = next_logprobs[seq[0]].item()
                if len(seq) > 1:  # multi-token surface form: one extra pass
                    full_ids = torch.cat(
                        [prefix_ids[0], torch.tensor(seq, device=self.device)]
                    ).unsqueeze(0)
                    with torch.inference_mode():
                        logits = self.model(full_ids).logits[0]
                    for i, tok_id in enumerate(seq[1:], start=1):
                        pos = prefix_len - 1 + i
                        logprob += torch.log_softmax(logits[pos].float(), dim=-1)[tok_id].item()
                variant_logprobs.append(logprob)
            word_scores.append(torch.logsumexp(torch.tensor(variant_logprobs), dim=0).item())
        probs = torch.softmax(torch.tensor(word_scores), dim=0).tolist()
        return dict(zip(self.DECISION_WORDS, probs))

    # -- citation ----------------------------------------------------------
    def _select_cited_sentence(
        self, decision_messages: list, decision: str, evidence_sentences: list[str], question: str
    ) -> str:
        if not evidence_sentences:
            return ""
        torch = self._torch
        prompt = self._render(citation_messages(decision_messages, decision, evidence_sentences))
        input_ids = self._encode(prompt)
        with torch.inference_mode():
            out_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=6,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        reply = self.tokenizer.decode(out_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        idx = parse_sentence_choice(reply, len(evidence_sentences))
        if idx is not None:
            return evidence_sentences[idx]
        return best_matching_sentence(set(tokenize(question)), evidence_sentences)

    def generate(
        self, question: str, doc_id: str, evidence_text: str, evidence_sentences: list[str]
    ) -> GenerationResult:
        messages = build_messages(question, evidence_text, self.max_evidence_chars)
        probs = self._decision_distribution(self._render(messages))
        decision = max(probs, key=probs.get)
        cited_sentence = self._select_cited_sentence(
            messages, decision, evidence_sentences, question
        )
        return GenerationResult(
            answer_text=format_answer(decision, cited_sentence, doc_id),
            decision=decision,
            confidence=probs[decision],
            class_probs=probs,
            cited_doc_id=doc_id,
            cited_sentence=cited_sentence,
        )
