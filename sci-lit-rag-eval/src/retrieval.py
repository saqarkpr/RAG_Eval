"""Two retrievers over the arXiv-abstract corpus:

- BM25Retriever: classic sparse lexical ranking (rank_bm25.BM25Okapi).
- TfidfLsaRetriever: TF-IDF weighted bag-of-words projected into a low-rank
  latent space via truncated SVD (i.e. classic LSA / "dense-ish" retrieval
  without needing a neural encoder or GPU).

Both expose the same .retrieve(query, k) -> List[(arxiv_id, score)] interface
so downstream code (generator, evaluation) is retriever-agnostic.
"""
import time
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from src.data_utils import tokenize


class BM25Retriever:
    name = "bm25"

    def __init__(self, docs):
        self.docs = docs
        self.ids = [d["arxiv_id"] for d in docs]
        t0 = time.perf_counter()
        self.corpus_tokens = [tokenize(d["text"]) for d in docs]
        self.model = BM25Okapi(self.corpus_tokens)
        self.build_time_s = time.perf_counter() - t0

    def retrieve(self, query, k=5):
        q_tokens = tokenize(query)
        scores = self.model.get_scores(q_tokens)
        order = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in order]

    def approx_memory_bytes(self):
        # rank_bm25 keeps per-doc term frequency dicts + doc freqs; approximate
        # footprint via the size of those structures.
        import sys
        total = sys.getsizeof(self.model.doc_freqs)
        for d in self.model.doc_freqs:
            total += sys.getsizeof(d)
            for k_, v_ in d.items():
                total += sys.getsizeof(k_) + sys.getsizeof(v_)
        total += sys.getsizeof(self.model.idf)
        return total


class TfidfLsaRetriever:
    name = "tfidf_lsa"

    def __init__(self, docs, n_components=16, random_state=42):
        self.docs = docs
        self.ids = [d["arxiv_id"] for d in docs]
        texts = [d["text"] for d in docs]
        t0 = time.perf_counter()
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize, lowercase=False, token_pattern=None, min_df=1
        )
        tfidf = self.vectorizer.fit_transform(texts)
        n_comp = min(n_components, min(tfidf.shape) - 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
        self.doc_vecs = self.svd.fit_transform(tfidf)
        self.build_time_s = time.perf_counter() - t0

    def retrieve(self, query, k=5):
        q_tfidf = self.vectorizer.transform([query])
        q_vec = self.svd.transform(q_tfidf)
        sims = cosine_similarity(q_vec, self.doc_vecs)[0]
        order = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i])) for i in order]

    def approx_memory_bytes(self):
        return self.doc_vecs.nbytes + self.svd.components_.nbytes


RETRIEVERS = {
    "bm25": BM25Retriever,
    "tfidf_lsa": TfidfLsaRetriever,
}


def build_retriever(name, docs):
    return RETRIEVERS[name](docs)
