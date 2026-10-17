"""Build a ready pipeline from a corpus directory and a backend name (shared by CLI and API)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from grounded_rag.envelopes import IngestReport
from grounded_rag.generation import LLMGenerator
from grounded_rag.loaders import load_directory
from grounded_rag.pipeline import PipelineConfig, RAGPipeline
from grounded_rag.providers import BACKENDS, LLMBackend, backend_from_env

DEFAULT_CORPUS_DIR = Path(__file__).resolve().parents[2] / "corpus"
BACKEND_CHOICES = ["none", *BACKENDS]


def resolve_backend(name: str, env: Mapping[str, str] | None = None) -> LLMBackend | None:
    """``none`` means extractive answers only: no LLM, no credentials, no cost."""
    return None if name == "none" else backend_from_env(name, os.environ if env is None else env)


def build_pipeline(
    *,
    corpus_dir: Path | None,
    backend: str = "none",
    env: Mapping[str, str] | None = None,
    config: PipelineConfig | None = None,
) -> tuple[RAGPipeline, IngestReport | None]:
    llm = resolve_backend(backend, env)
    pipeline = RAGPipeline(generator=LLMGenerator(llm) if llm else None, config=config)
    report = None
    if corpus_dir is not None:
        documents, _ = load_directory(Path(corpus_dir))
        report = pipeline.ingest(documents)
    return pipeline, report
