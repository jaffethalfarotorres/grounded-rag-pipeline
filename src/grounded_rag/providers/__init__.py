"""LLM providers behind one ``LLMBackend`` contract, plus a registry to build them by name."""

from __future__ import annotations

from collections.abc import Mapping

from grounded_rag.providers.base import BackendError, BackendUnavailable, LLMBackend
from grounded_rag.providers.groq import GroqBackend
from grounded_rag.providers.huggingface import HuggingFaceBackend
from grounded_rag.providers.mock import LLMCall, MockBackend
from grounded_rag.providers.ollama import OllamaBackend
from grounded_rag.providers.watsonx import WatsonxBackend

BACKENDS = {
    "watsonx": WatsonxBackend,
    "groq": GroqBackend,
    "huggingface": HuggingFaceBackend,
    "ollama": OllamaBackend,
}


def backend_from_env(name: str, env: Mapping[str, str] | None = None) -> LLMBackend:
    """Build a real backend from environment variables."""
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; choose one of {sorted(BACKENDS)}")
    return BACKENDS[name].from_env(env)


__all__ = [
    "BACKENDS",
    "BackendError",
    "BackendUnavailable",
    "GroqBackend",
    "HuggingFaceBackend",
    "LLMBackend",
    "LLMCall",
    "MockBackend",
    "OllamaBackend",
    "WatsonxBackend",
    "backend_from_env",
]
