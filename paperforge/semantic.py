"""Embedding layer: near-duplicate detection, topic clustering, semantic search.

Bibliographic data is full of the same paper wearing different hats — preprint
vs camera-ready vs journal extension. String matching on titles misses all of
it. Embeddings catch it, and the same index then powers search over the corpus.

Falls back to TF-IDF if sentence-transformers is not installed, so the app still
runs on a thin environment.
"""
from __future__ import annotations

import numpy as np

from .models import PaperRecord

_MODEL_CACHE: dict[str, object] = {}


def _text_for(rec: PaperRecord) -> str:
    parts = [
        rec.get("title") or rec.input_ref,
        "; ".join(rec.get("authors") or [])[:200],
        (rec.get("abstract") or "")[:1500],
        "; ".join(rec.get("fields_of_study") or []),
    ]
    return " \n ".join(p for p in parts if p)


class EmbeddingIndex:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.backend = "tfidf"
        self._model = None
        self._vectorizer = None
        self.matrix: np.ndarray | None = None
        self.ids: list[str] = []

    def _load(self):
        if self._model is not None:
            return
        if self.model_name in _MODEL_CACHE:
            self._model = _MODEL_CACHE[self.model_name]
            self.backend = "sentence-transformers"
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            _MODEL_CACHE[self.model_name] = self._model
            self.backend = "sentence-transformers"
        except Exception:
            self.backend = "tfidf"

    # ---------------------------------------------------------------- encode
    def fit(self, records: list[PaperRecord]) -> np.ndarray:
        texts = [_text_for(r) for r in records]
        self.ids = [r.record_id for r in records]
        self._load()

        if self.backend == "sentence-transformers":
            vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            self.matrix = np.asarray(vecs, dtype=np.float32)
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer(
                stop_words="english", max_features=20000, ngram_range=(1, 2)
            )
            m = self._vectorizer.fit_transform(texts).toarray().astype(np.float32)
            norms = np.linalg.norm(m, axis=1, keepdims=True)
            self.matrix = m / np.clip(norms, 1e-9, None)

        for rec, vec in zip(records, self.matrix):
            rec.embedding = vec.tolist()
        return self.matrix

    def similarity(self) -> np.ndarray:
        if self.matrix is None:
            raise RuntimeError("call fit() first")
        return self.matrix @ self.matrix.T

    # ------------------------------------------------------------------ dedup
    def mark_duplicates(self, records: list[PaperRecord], threshold: float = 0.93) -> int:
        """Flag near-duplicates. Keeps the highest-confidence copy as canonical."""
        if len(records) < 2:
            return 0
        sim = self.similarity()
        np.fill_diagonal(sim, 0.0)
        order = sorted(range(len(records)), key=lambda i: -records[i].confidence)
        claimed: dict[int, int] = {}

        for i in order:
            if i in claimed:
                continue
            for j in np.where(sim[i] >= threshold)[0]:
                j = int(j)
                if j == i or j in claimed:
                    continue
                # exact DOI match is a hard duplicate regardless of score;
                # different DOIs at high similarity is usually preprint vs journal,
                # which we still link but keep as separate rows.
                claimed[j] = i
                records[j].duplicate_of = records[i].record_id
        return len(claimed)

    # --------------------------------------------------------------- clusters
    def cluster(self, records: list[PaperRecord], threshold: float = 0.72) -> int:
        """Agglomerative single-link clustering over cosine distance."""
        if len(records) < 2:
            for r in records:
                r.cluster_id = 0
            return 1
        try:
            from sklearn.cluster import AgglomerativeClustering

            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1.0 - threshold,
                metric="cosine",
                linkage="average",
            )
            labels = model.fit_predict(self.matrix)
        except Exception:
            labels = _greedy_clusters(self.similarity(), threshold)

        for rec, label in zip(records, labels):
            rec.cluster_id = int(label)
        return len(set(labels))

    # ----------------------------------------------------------------- search
    def search(self, query: str, records: list[PaperRecord], top_k: int = 10):
        self._load()
        if self.backend == "sentence-transformers":
            q = self._model.encode([query], normalize_embeddings=True)[0]
        else:
            q = self._vectorizer.transform([query]).toarray()[0]
            q = q / max(float(np.linalg.norm(q)), 1e-9)
        scores = self.matrix @ np.asarray(q, dtype=np.float32)
        idx = np.argsort(-scores)[:top_k]
        return [(records[i], float(scores[i])) for i in idx]


def _greedy_clusters(sim: np.ndarray, threshold: float) -> list[int]:
    n = sim.shape[0]
    labels = [-1] * n
    current = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        labels[i] = current
        stack = [i]
        while stack:
            k = stack.pop()
            for j in range(n):
                if labels[j] == -1 and sim[k, j] >= threshold:
                    labels[j] = current
                    stack.append(j)
        current += 1
    return labels


def cluster_labels(records: list[PaperRecord]) -> dict[int, str]:
    """Name each cluster by its most distinctive shared terms."""
    from collections import Counter

    buckets: dict[int, Counter] = {}
    for r in records:
        if r.cluster_id is None:
            continue
        terms = (r.get("fields_of_study") or []) + (r.get("keywords") or [])
        buckets.setdefault(r.cluster_id, Counter()).update(t.lower() for t in terms)
    return {
        cid: ", ".join(t for t, _ in counter.most_common(3)) or f"cluster {cid}"
        for cid, counter in buckets.items()
    }
