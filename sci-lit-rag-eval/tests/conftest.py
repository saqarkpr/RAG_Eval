"""Make `src`, `train` and `evaluate` importable when tests run from any
directory, and share a tiny synthetic corpus + QA benchmark."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def toy_docs():
    """Four abstracts with disjoint vocabulary and one acronym each, so the
    gold retrieval target and the extractable span are unambiguous."""
    raw = [
        ("2401.00001", "Dense Passage Retrieval for Open Domain Questions",
         "We introduce DPR, a dense passage retriever trained with contrastive "
         "learning on question passage pairs. DPR outperforms lexical baselines."),
        ("2401.00002", "Calibrating Language Model Confidence",
         "We propose CALM, a calibration method for language models. CALM reduces "
         "expected calibration error on held out benchmarks."),
        ("2401.00003", "Quantizing Transformers to Four Bits",
         "We present QuantX, a post training quantization scheme for transformers "
         "that keeps accuracy at four bits."),
        ("2401.00004", "Detecting Hallucinations with Self Consistency",
         "We describe HalluCheck, a hallucination detector based on sampling "
         "multiple answers and measuring agreement."),
    ]
    docs = [{"arxiv_id": a, "title": t, "abstract": ab} for a, t, ab in raw]
    for d in docs:
        d["text"] = d["title"] + ". " + d["abstract"]
    return docs


@pytest.fixture
def toy_qa():
    return [
        {"qid": "q1", "question": "Does DPR use contrastive learning on question passage pairs?",
         "answer_type": "yes_no", "gold_answer": "Yes", "gold_arxiv_id": "2401.00001"},
        {"qid": "q2", "question": "Does CALM increase expected calibration error?",
         "answer_type": "yes_no", "gold_answer": "No", "gold_arxiv_id": "2401.00002"},
        {"qid": "q3", "question": "Is QuantX a scheme that keeps accuracy at four bits?",
         "answer_type": "yes_no", "gold_answer": "Yes", "gold_arxiv_id": "2401.00003"},
        {"qid": "q4", "question": "Does HalluCheck avoid sampling multiple answers?",
         "answer_type": "yes_no", "gold_answer": "No", "gold_arxiv_id": "2401.00004"},
        {"qid": "q5", "question": "What dense passage retriever is introduced?",
         "answer_type": "extractive", "gold_answer": "DPR", "gold_arxiv_id": "2401.00001"},
        {"qid": "q6", "question": "What calibration method for language models is proposed?",
         "answer_type": "extractive", "gold_answer": "CALM", "gold_arxiv_id": "2401.00002"},
        {"qid": "q7", "question": "What post training quantization scheme is presented?",
         "answer_type": "extractive", "gold_answer": "QuantX", "gold_arxiv_id": "2401.00003"},
        {"qid": "q8", "question": "What hallucination detector is described?",
         "answer_type": "extractive", "gold_answer": "HalluCheck", "gold_arxiv_id": "2401.00004"},
        {"qid": "q9", "question": "What GPU count was used to train the dense retriever?",
         "answer_type": "unanswerable", "gold_answer": None, "gold_arxiv_id": None},
        {"qid": "q10", "question": "Which venue published the calibration paper?",
         "answer_type": "unanswerable", "gold_answer": None, "gold_arxiv_id": None},
        {"qid": "q11", "question": "What is the licence of the quantization code?",
         "answer_type": "unanswerable", "gold_answer": None, "gold_arxiv_id": None},
        {"qid": "q12", "question": "How many annotators labelled the hallucination data?",
         "answer_type": "unanswerable", "gold_answer": None, "gold_arxiv_id": None},
    ]
