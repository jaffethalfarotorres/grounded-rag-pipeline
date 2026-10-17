"""Minimal Prometheus-compatible metrics: labelled counters and a latency summary.

A few dozen lines instead of a client library: the API's /metrics endpoint serves
this text format, which Prometheus scrapes as is.
"""

from __future__ import annotations

from collections import defaultdict

_HELP = {
    "rag_queries_total": ("counter", "Questions answered, by final status."),
    "rag_cache_hits_total": ("counter", "Questions served from the answer cache."),
    "rag_llm_calls_total": ("counter", "LLM calls, by outcome."),
    "rag_grounding_rejections_total": ("counter", "LLM answers rejected by the grounding gate."),
    "rag_chunks_indexed_total": ("counter", "Chunks added to the index."),
    "rag_passages_quarantined_total": ("counter", "Passages kept out for prompt injection."),
    "rag_query_seconds": ("summary", "Time to answer a question."),
}


def _labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    return "{" + ",".join(f'{key}="{value}"' for key, value in sorted(labels.items())) + "}"


class Metrics:
    def __init__(self) -> None:
        self._counters: dict[str, dict[str, float]] = defaultdict(dict)
        self._summaries: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])

    def inc(self, name: str, amount: float = 1, **labels: str) -> None:
        key = _labels(labels)
        self._counters[name][key] = self._counters[name].get(key, 0) + amount

    def observe(self, name: str, value: float) -> None:
        total = self._summaries[name]
        total[0] += value
        total[1] += 1

    def value(self, name: str, **labels: str) -> float:
        return self._counters[name].get(_labels(labels), 0)

    def render(self) -> str:
        lines = []
        for name, (kind, text) in _HELP.items():
            lines += [f"# HELP {name} {text}", f"# TYPE {name} {kind}"]
            if kind == "summary":
                total, count = self._summaries[name]
                lines += [f"{name}_sum {total:.6f}", f"{name}_count {count:g}"]
                continue
            series = self._counters.get(name) or {"": 0}
            lines += [f"{name}{labels} {value:g}" for labels, value in sorted(series.items())]
        return "\n".join(lines) + "\n"
