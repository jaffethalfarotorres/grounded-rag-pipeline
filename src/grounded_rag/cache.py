"""In-process LRU cache for answers, with a TTL and an injectable clock.

The key includes the index version, so any ingest makes every cached answer stale
without having to track which documents an answer depended on.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable, Mapping

from grounded_rag.envelopes import RAGAnswer


def cache_key(
    question: str, *, top_k: int, filters: Mapping[str, str], index_version: int, generator: str
) -> str:
    payload = [
        " ".join(question.lower().split()),
        top_k,
        sorted(filters.items()),
        index_version,
        generator,
    ]
    return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()


class QueryCache:
    def __init__(self, *, max_entries: int, ttl_seconds: float, clock: Callable[[], float]) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self._entries: OrderedDict[str, tuple[float, RAGAnswer]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> RAGAnswer | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        stored_at, answer = entry
        if self.clock() - stored_at > self.ttl_seconds:
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return answer

    def put(self, key: str, answer: RAGAnswer) -> None:
        self._entries[key] = (self.clock(), answer)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()
