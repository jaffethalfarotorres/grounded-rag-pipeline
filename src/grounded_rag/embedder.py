"""Embedders behind one small protocol.

``HashingEmbedder`` is the default: feature hashing of words and word pairs into a
fixed-size, L2-normalized vector. It needs no model download and no network, so CI
and the eval run offline, and its results are reproducible. It captures lexical
similarity only; ``SentenceTransformerEmbedder`` (optional extra) plugs in a real
semantic model through the same interface.
"""

from __future__ import annotations

import hashlib
import importlib
import itertools
import math
from collections import Counter
from collections.abc import Sequence
from typing import Any, Protocol

from grounded_rag.text import content_tokens


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    name = "hashing"

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    @staticmethod
    def _features(text: str) -> list[str]:
        words = content_tokens(text)
        return words + [f"{a}_{b}" for a, b in itertools.pairwise(words)]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for feature, count in Counter(self._features(text)).items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(count))
        norm = math.sqrt(math.fsum(x * x for x in vector))
        return [x / norm for x in vector] if norm else vector

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]


class SentenceTransformerEmbedder:
    """Adapter for a ``sentence_transformers.SentenceTransformer`` model."""

    def __init__(self, model: Any, *, name: str) -> None:
        self._model = model
        self.name = name
        self.dim = model.get_sentence_embedding_dimension()

    @classmethod
    def from_name(
        cls, name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> SentenceTransformerEmbedder:
        try:
            module = importlib.import_module("sentence_transformers")
        except ImportError as exc:
            hint = "pip install 'grounded-rag-pipeline[embeddings]'"
            raise RuntimeError(f"sentence-transformers is not installed: {hint}") from exc
        return cls(module.SentenceTransformer(name), name=name)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        rows = self._model.encode(list(texts), normalize_embeddings=True)
        rows = rows.tolist() if hasattr(rows, "tolist") else rows
        return [[float(x) for x in row] for row in rows]
