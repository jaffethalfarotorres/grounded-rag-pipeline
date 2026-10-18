"""HTTP API: POST /ingest, POST /query, GET /health, GET /metrics.

Request bodies are Pydantic models, so malformed input (empty question, top_k out of
range, a document id that is not a slug) is rejected with 422 before any code runs.

Run it with:  uvicorn grounded_rag.api:app_from_env --factory
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from grounded_rag import __version__
from grounded_rag.bootstrap import DEFAULT_CORPUS_DIR, build_pipeline
from grounded_rag.envelopes import Document, IngestReport, RAGAnswer
from grounded_rag.pipeline import RAGPipeline


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    documents: list[Document] = Field(min_length=1, max_length=100)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=10)
    filters: dict[str, str] = Field(default_factory=dict)


def create_app(pipeline: RAGPipeline) -> FastAPI:
    app = FastAPI(title="grounded-rag-pipeline", version=__version__)

    @app.post("/ingest")
    def ingest(request: IngestRequest) -> IngestReport:
        return pipeline.ingest(request.documents)

    @app.post("/query")
    def query(request: QueryRequest) -> RAGAnswer:
        return pipeline.ask(request.question, top_k=request.top_k, filters=request.filters)

    @app.get("/health")
    def health() -> dict[str, object]:
        backend = getattr(pipeline.llm, "backend", None)
        providers = {"providers": backend.status()} if hasattr(backend, "status") else {}
        return {
            **providers,
            "status": "ok",
            "documents": len(pipeline.documents),
            "chunks": len(pipeline.retriever),
            "index_version": pipeline.index_version,
            "generator": pipeline.generator_name,
            "embedder": pipeline.retriever.embedder.name,
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> PlainTextResponse:
        return PlainTextResponse(pipeline.metrics.render(), media_type="text/plain; version=0.0.4")

    return app


def app_from_env(env: Mapping[str, str] | None = None) -> FastAPI:
    """App factory: RAG_CORPUS_DIR (default: the bundled corpus) and RAG_BACKEND (default none)."""
    env = os.environ if env is None else env
    corpus = Path(env.get("RAG_CORPUS_DIR", DEFAULT_CORPUS_DIR))
    pipeline, _ = build_pipeline(
        corpus_dir=corpus if corpus.is_dir() else None,
        backend=env.get("RAG_BACKEND", "none"),
        embedder=env.get("RAG_EMBEDDER", "hashing"),
        env=env,
    )
    return create_app(pipeline)
