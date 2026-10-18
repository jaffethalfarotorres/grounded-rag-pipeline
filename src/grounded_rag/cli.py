"""Command line.

grounded-rag ask "How long do refunds take?"            # bundled corpus, no API key
grounded-rag ask "..." --backend groq --filter category=billing
grounded-rag ingest corpus/                             # shows quarantined passages
grounded-rag serve --port 8000                          # needs the [serve] extra
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from grounded_rag.bootstrap import (
    BACKEND_CHOICES,
    DEFAULT_CORPUS_DIR,
    EMBEDDER_CHOICES,
    build_pipeline,
)
from grounded_rag.embedder import EmbedderUnavailable
from grounded_rag.providers import BackendUnavailable


def _filter(text: str) -> tuple[str, str]:
    key, sep, value = text.partition("=")
    if not sep or not key or not value:
        raise argparse.ArgumentTypeError(f"expected key=value, got {text!r}")
    return key, value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grounded-rag", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    ask = commands.add_parser("ask", help="answer one question and print the RAGAnswer as JSON")
    ask.add_argument("question")
    ask.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_DIR)
    ask.add_argument("--backend", choices=BACKEND_CHOICES, default="none")
    ask.add_argument("--embedder", choices=EMBEDDER_CHOICES, default="hashing")
    ask.add_argument("--top-k", type=int, default=4)
    ask.add_argument("--filter", type=_filter, action="append", default=[], metavar="KEY=VALUE")

    ingest = commands.add_parser("ingest", help="load a directory and print the ingest report")
    ingest.add_argument("corpus", type=Path)

    serve = commands.add_parser("serve", help="run the HTTP API (needs the [serve] extra)")
    serve.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_DIR)
    serve.add_argument("--backend", choices=BACKEND_CHOICES, default="none")
    serve.add_argument("--embedder", choices=EMBEDDER_CHOICES, default="hashing")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "ingest":
            _, report = build_pipeline(corpus_dir=args.corpus)
            print(report.model_dump_json(indent=2))
            return 0

        pipeline, _ = build_pipeline(
            corpus_dir=args.corpus, backend=args.backend, embedder=args.embedder, env=env
        )
    except (BackendUnavailable, EmbedderUnavailable) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.command == "serve":
        import uvicorn

        from grounded_rag.api import create_app

        uvicorn.run(create_app(pipeline), host=args.host, port=args.port)
        return 0

    answer = pipeline.ask(args.question, top_k=args.top_k, filters=dict(args.filter))
    print(answer.model_dump_json(indent=2))
    return 0
