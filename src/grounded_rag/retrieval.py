"""Hybrid retrieval: BM25 and vector search, fused with Reciprocal Rank Fusion.

Why both: BM25 nails exact terms (plan names, error codes, numbers) and vectors catch
paraphrases. Why RRF instead of adding scores: BM25 scores and cosine similarities
live on different, query-dependent scales, so a weighted sum needs per-corpus tuning.
RRF only uses ranks, so it needs none, and a chunk both retrievers like wins.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

from grounded_rag.embedder import Embedder
from grounded_rag.envelopes import Chunk, Hit
from grounded_rag.text import content_tokens

RRF_K = 60


class BM25Index:
    """Okapi BM25 over pre-tokenized documents, with add and remove."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._terms: dict[str, Counter[str]] = {}
        self._df: Counter[str] = Counter()

    def __len__(self) -> int:
        return len(self._terms)

    def add(self, doc_id: str, terms: Sequence[str]) -> None:
        if doc_id in self._terms:
            self.remove(doc_id)
        counts = Counter(terms)
        self._terms[doc_id] = counts
        self._df.update(counts.keys())

    def remove(self, doc_id: str) -> None:
        counts = self._terms.pop(doc_id, None)
        if counts:
            self._df.subtract(counts.keys())
            self._df = +self._df  # drop zero counts

    def document_frequency(self, term: str) -> int:
        return self._df[term]

    def _idf(self, term: str) -> float:
        n, df = len(self._terms), self._df[term]
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query_terms: Sequence[str], top_n: int = 20) -> list[tuple[str, float]]:
        if not self._terms:
            return []
        avg_len = math.fsum(sum(c.values()) for c in self._terms.values()) / len(self._terms)
        query = set(query_terms)
        scored = []
        for doc_id, counts in self._terms.items():
            length = sum(counts.values())
            score = 0.0
            for term in query & counts.keys():
                tf = counts[term]
                norm = tf + self.k1 * (1 - self.b + self.b * length / avg_len)
                score += self._idf(term) * tf * (self.k1 + 1) / norm
            if score > 0:
                scored.append((doc_id, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_n]


def rrf(rankings: Iterable[Sequence[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank)."""
    scores: Counter[str] = Counter()
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


class HybridRetriever:
    def __init__(self, embedder: Embedder, *, candidates: int = 20, rrf_k: int = RRF_K) -> None:
        self.embedder = embedder
        self.candidates = candidates
        self.rrf_k = rrf_k
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, list[float]] = {}
        self._bm25 = BM25Index()

    def __len__(self) -> int:
        return len(self._chunks)

    def chunks_of(self, doc_id: str) -> list[Chunk]:
        """The indexed chunks of one document, in order."""
        return sorted(
            (c for c in self._chunks.values() if c.doc_id == doc_id), key=lambda c: c.index
        )

    def knows(self, term: str) -> bool:
        """Whether any indexed chunk contains this (normalized) term."""
        return self._bm25.document_frequency(term) > 0

    def add(self, chunks: Sequence[Chunk]) -> None:
        """Index chunks. Re-adding a document replaces all of its previous chunks."""
        for doc_id in {chunk.doc_id for chunk in chunks}:
            self.remove_document(doc_id)
        vectors = self.embedder.embed([chunk.text for chunk in chunks])
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._chunks[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = vector
            self._bm25.add(chunk.chunk_id, content_tokens(chunk.text))

    def remove_document(self, doc_id: str) -> None:
        for chunk_id in [cid for cid, c in self._chunks.items() if c.doc_id == doc_id]:
            del self._chunks[chunk_id], self._vectors[chunk_id]
            self._bm25.remove(chunk_id)

    def search(
        self, query: str, *, top_k: int = 4, filters: Mapping[str, str] | None = None
    ) -> list[Hit]:
        filters = filters or {}
        allowed = {
            cid
            for cid, chunk in self._chunks.items()
            if all(chunk.metadata.get(key) == value for key, value in filters.items())
        }
        if not allowed:
            return []

        lexical = [
            cid
            for cid, _ in self._bm25.search(content_tokens(query), top_n=len(self._chunks))
            if cid in allowed
        ][: self.candidates]
        query_vector = self.embedder.embed([query])[0]
        semantic = [
            cid
            for _, cid in sorted(
                ((_dot(query_vector, self._vectors[cid]), cid) for cid in allowed),
                key=lambda item: (-item[0], item[1]),
            )
        ][: self.candidates]

        hits = []
        for position, (cid, score) in enumerate(rrf([semantic, lexical], self.rrf_k)[:top_k], 1):
            hits.append(
                Hit(
                    source_id=f"S{position}",
                    chunk=self._chunks[cid],
                    score=round(score, 6),
                    vector_rank=semantic.index(cid) + 1 if cid in semantic else None,
                    bm25_rank=lexical.index(cid) + 1 if cid in lexical else None,
                )
            )
        return hits
