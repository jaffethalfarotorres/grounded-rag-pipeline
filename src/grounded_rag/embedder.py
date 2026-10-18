"""Embedders behind one small protocol.

``HashingEmbedder`` is the default: feature hashing of words and word pairs into a
fixed-size, L2-normalized vector. It needs no model download and no network, so CI
and the offline eval run anywhere, and its results are reproducible. It captures lexical
similarity only. ``FastEmbedEmbedder`` (optional ``[embeddings]`` extra) runs a small
real model locally on CPU through ONNX, with no API key and no cost, behind the same
interface.

Queries and passages go through separate methods (``embed_query`` and ``embed``):
retrieval models are trained with that asymmetry and some of them add a prefix to
queries, so the retriever must not treat them as the same kind of text.
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

DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"


class EmbedderUnavailable(RuntimeError):
    """The embedder cannot be built here: its optional dependency is not installed."""


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


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

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def _unit(row: Any) -> list[float]:
    """A plain list of floats with length 1 (cosine similarity then equals a dot product)."""
    values = [float(x) for x in (row.tolist() if hasattr(row, "tolist") else row)]
    norm = math.sqrt(math.fsum(x * x for x in values))
    return [x / norm for x in values] if norm else values


class FastEmbedEmbedder:
    """A local ONNX model through ``fastembed.TextEmbedding`` (about 67 MB for bge-small)."""

    def __init__(self, model: Any, *, name: str) -> None:
        self._model = model
        self.name = name
        self.dim = model.embedding_size

    @classmethod
    def from_name(cls, name: str = DEFAULT_FASTEMBED_MODEL) -> FastEmbedEmbedder:
        try:
            module = importlib.import_module("fastembed")
        except ImportError as exc:
            hint = "pip install 'grounded-rag-pipeline[embeddings]'"
            raise EmbedderUnavailable(f"fastembed is not installed: {hint}") from exc
        return cls(module.TextEmbedding(name), name=name)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return [_unit(row) for row in self._model.passage_embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return _unit(next(iter(self._model.query_embed(text))))


EMBEDDERS = ("hashing", "fastembed")


def make_embedder(name: str) -> Embedder:
    """Build an embedder by name; ``fastembed`` downloads its model on first use."""
    if name == "hashing":
        return HashingEmbedder()
    if name == "fastembed":
        return FastEmbedEmbedder.from_name()
    raise ValueError(f"unknown embedder {name!r}; choose one of {list(EMBEDDERS)}")
