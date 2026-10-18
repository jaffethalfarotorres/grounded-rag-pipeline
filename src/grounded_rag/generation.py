"""Two answer writers behind the same call: an LLM, and an extractive fallback.

``ExtractiveGenerator`` quotes the source sentences that best match the question,
each with its label. It needs no model and is grounded by construction, so it serves
as the offline mode and as the fallback when an LLM answer fails the grounding gate.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from grounded_rag.defense import build_system_prompt, wrap_untrusted
from grounded_rag.envelopes import Hit, ModelTier
from grounded_rag.grounding import split_sentences
from grounded_rag.providers.base import LLMBackend
from grounded_rag.text import content_tokens

REFUSAL = "I can't answer that from the available documents."

ANSWER_TASK = f"""\
Answer the question using only the numbered sources.
Rules:
- End every sentence with the labels of the sources it uses, like [S1] or [S1, S2].
- Reuse the wording of the sources and copy numbers exactly; never round or convert them.
- If the sources do not contain the answer, reply with exactly: {REFUSAL}
- Plain text, at most 3 sentences, no markdown."""

_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_SPACES = re.compile(r"[^\S\n]+")  # any whitespace except newlines, Unicode included


def is_refusal(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return REFUSAL.rstrip(".").lower() in normalized


class ExtractiveGenerator:
    name = "extractive"

    def __init__(self, max_sentences: int = 2, min_words: int = 3) -> None:
        self.max_sentences = max_sentences
        self.min_words = min_words

    def _candidates(self, hits: Sequence[Hit]) -> list[tuple[int, int, str, str]]:
        found = []
        for rank, hit in enumerate(hits):
            for line in hit.chunk.text.splitlines():
                if line.lstrip().startswith("#"):
                    continue
                for position, sentence in enumerate(split_sentences(_BULLET.sub("", line))):
                    found.append((rank, position, sentence, hit.source_id))
        return found

    def generate(self, question: str, hits: Sequence[Hit]) -> str:
        wanted = set(content_tokens(question))
        scored = []
        for rank, position, sentence, label in self._candidates(hits):
            words = set(content_tokens(sentence))
            if len(words) < self.min_words or not wanted:
                continue
            overlap = len(wanted & words) / len(wanted)
            if overlap > 0:
                scored.append((-overlap, rank, position, sentence, label))
        if not scored:
            return REFUSAL
        best = sorted(scored)[: self.max_sentences]
        return " ".join(f"{sentence.rstrip('.!? ')} [{label}]." for *_, sentence, label in best)


class LLMGenerator:
    name = "llm"

    def __init__(self, backend: LLMBackend, *, tier: ModelTier = "small") -> None:
        self.backend = backend
        self.tier = tier
        self.system_prompt = build_system_prompt(ANSWER_TASK)

    def generate(self, question: str, hits: Sequence[Hit]) -> str:
        sources = "\n\n".join(f"[{h.source_id}] {h.chunk.title}\n{h.chunk.text}" for h in hits)
        prompt = f"Sources:\n{wrap_untrusted(sources)}\n\nQuestion:\n{wrap_untrusted(question)}"
        reply = self.backend.complete(prompt, system=self.system_prompt, tier=self.tier)
        # Models emit narrow no-break and other Unicode spaces ("49\u202fUSD", seen live on
        # Groq). Normalize them so the gate, the cache and every consumer see plain spaces.
        return _SPACES.sub(" ", reply).strip()
