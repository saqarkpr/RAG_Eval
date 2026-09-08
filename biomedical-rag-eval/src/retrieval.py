"""
Retrieval backends for the biomedical RAG pipeline.

Two independent retrievers are implemented, both from lightweight,
locally-computable primitives (no pretrained embedding models are
downloaded, so the whole pipeline runs offline):

- BM25Retriever: classic lexical ranking (rank_bm25).
- LsaRetriever: a "dense" retriever built from TF-IDF + truncated SVD
  (latent semantic analysis) fit directly on the corpus. This is the
  from-scratch stand-in for a neural bi-encoder; see README for the
  documented upgrade path to a real sentence-embedding or trained
  dual-encoder model once GPU/internet access to model weights is
  available (e.g. on Colab).
"""
import re
import time
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

_TOKEN_RE = re.compile(r"[a-zA-Z]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class RetrievalResult:
    doc_ids: list[str]
    scores: list[float]
    latency_s: float


class BM25Retriever:
    name = "bm25"

    def __init__(self, doc_ids: list[str], texts: list[str]):
        self.doc_ids = doc_ids
        tokenized = [tokenize(t) for t in texts]
        self._index = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int = 10) -> RetrievalResult:
        t0 = time.perf_counter()
        scores = self._index.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        dt = time.perf_counter() - t0
        return RetrievalResult(
            doc_ids=[self.doc_ids[i] for i in top],
            scores=[float(scores[i]) for i in top],
            latency_s=dt,
        )


class LsaRetriever:
    name = "tfidf_lsa"

    def __init__(self, doc_ids: list[str], texts: list[str], n_components: int = 128):
        self.doc_ids = doc_ids
        n_components = min(n_components, len(texts) - 1, 300)
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize, lowercase=False, min_df=2, max_df=0.9
        )
        tfidf = self.vectorizer.fit_transform(texts)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.doc_vecs = self.svd.fit_transform(tfidf)
        norms = np.linalg.norm(self.doc_vecs, axis=1, keepdims=True)
        self._doc_vecs_norm = self.doc_vecs / np.clip(norms, 1e-8, None)

    def retrieve(self, query: str, k: int = 10) -> RetrievalResult:
        t0 = time.perf_counter()
        q_tfidf = self.vectorizer.transform([query])
        q_vec = self.svd.transform(q_tfidf)[0]
        q_norm = q_vec / max(np.linalg.norm(q_vec), 1e-8)
        scores = self._doc_vecs_norm @ q_norm
        top = np.argsort(scores)[::-1][:k]
        dt = time.perf_counter() - t0
        return RetrievalResult(
            doc_ids=[self.doc_ids[i] for i in top],
            scores=[float(scores[i]) for i in top],
            latency_s=dt,
        )
