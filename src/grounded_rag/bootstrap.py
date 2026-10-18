"""Build a ready pipeline from a corpus directory and a backend name (shared by CLI and API)."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from pathlib import Path

from grounded_rag.embedder import EMBEDDERS, make_embedder
from grounded_rag.envelopes import IngestReport
from grounded_rag.generation import LLMGenerator
from grounded_rag.loaders import load_directory
from grounded_rag.pipeline import PipelineConfig, RAGPipeline
from grounded_rag.providers import BACKENDS, Completer, backend_from_env, build_chain

DEFAULT_CORPUS_DIR = Path(__file__).resolve().parents[2] / "corpus"
BACKEND_CHOICES = ["none", *BACKENDS]
EMBEDDER_CHOICES = list(EMBEDDERS)


def backend_spec(text: str) -> str:
    """argparse type: ``none``, one provider, or an ordered chain like ``groq,watsonx,ollama``."""
    names = text.split(",")
    unknown = [n for n in names if n not in BACKENDS and n != "none"]
    if unknown or ("none" in names and len(names) > 1):
        choices = ", ".join(["none", *BACKENDS])
        raise argparse.ArgumentTypeError(
            f"invalid backend {text!r}: use none, one of {choices}, or a comma-separated chain"
        )
    return text


def resolve_backend(name: str, env: Mapping[str, str] | None = None) -> Completer | None:
    """``none`` means extractive answers only: no LLM, no credentials, no cost.

    A comma-separated list is a fallback chain, tried in order.
    """
    env = os.environ if env is None else env
    if name == "none":
        return None
    if "," in name:
        return build_chain(name.split(","), env)
    return backend_from_env(name, env)


def build_pipeline(
    *,
    corpus_dir: Path | None,
    backend: str = "none",
    embedder: str = "hashing",
    env: Mapping[str, str] | None = None,
    config: PipelineConfig | None = None,
) -> tuple[RAGPipeline, IngestReport | None]:
    llm = resolve_backend(backend, env)
    pipeline = RAGPipeline(
        embedder=make_embedder(embedder),
        generator=LLMGenerator(llm) if llm else None,
        config=config,
    )
    report = None
    if corpus_dir is not None:
        documents, _ = load_directory(Path(corpus_dir))
        report = pipeline.ingest(documents)
    return pipeline, report
