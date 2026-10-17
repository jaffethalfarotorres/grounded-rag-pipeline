"""The grounding gate: a deterministic check between the model's answer and the user.

"Do not make things up" lives here as code, not as a line in a prompt. Every
sentence of an answer is checked against the sources it cites:

* ``fabricated_citation`` - it cites a source label that was never retrieved;
* ``uncited_claim``       - it states something and cites nothing;
* ``unsupported_claim``   - too few of its words appear in the sources it cites
                            (citing the right fact to the wrong source fails too);
* ``number_not_in_source`` - it contains a number its cited sources do not contain.

It is lexical on purpose: cheap, explainable and reproducible. The trade-off is that
it can reject a correct paraphrase; the pipeline then falls back to an extractive
answer quoted from the sources instead of releasing text it cannot verify.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from grounded_rag.envelopes import GroundingFinding, GroundingReport, Hit
from grounded_rag.text import content_tokens, tokens

DEFAULT_MIN_SUPPORT = 0.5

_MARKER = re.compile(r"\[\s*S\d+(?:\s*,\s*S\d+)*\s*\]")
_LABEL = re.compile(r"S\d+")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def cited_ids(text: str) -> list[str]:
    """Source labels in order of first use."""
    seen: list[str] = []
    for marker in _MARKER.findall(text):
        for label in _LABEL.findall(marker):
            if label not in seen:
                seen.append(label)
    return seen


def split_sentences(answer: str) -> list[str]:
    """Split on sentence ends; a fragment that is only citation markers joins the previous one."""
    sentences: list[str] = []
    for piece in (p.strip() for p in _SENTENCE_BREAK.split(answer.strip())):
        if not piece:
            continue
        if sentences and not tokens(_MARKER.sub(" ", piece)):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return sentences


def _numbers(text: str) -> set[str]:
    return {token for token in tokens(text) if _NUMBER.fullmatch(token)}


def verify(
    answer: str, hits: Sequence[Hit], *, min_support: float = DEFAULT_MIN_SUPPORT
) -> GroundingReport:
    if not answer.strip():
        finding = GroundingFinding(rule="empty_answer", sentence="", detail="no answer text")
        return GroundingReport(passed=False, sentences_checked=0, findings=(finding,))

    known = {hit.source_id: hit for hit in hits}
    findings: list[GroundingFinding] = []
    checked = 0
    for sentence in split_sentences(answer):
        claim = _MARKER.sub(" ", sentence)
        words = content_tokens(claim)
        if not words:
            continue
        checked += 1
        labels = cited_ids(sentence)

        def flag(rule: str, detail: str, sentence: str = sentence) -> None:
            findings.append(GroundingFinding(rule=rule, sentence=sentence, detail=detail))

        if unknown := [label for label in labels if label not in known]:
            flag("fabricated_citation", f"not retrieved: {', '.join(unknown)}")
        if not labels:
            if len(words) >= 2:
                flag("uncited_claim", "states something without citing a source")
            continue
        sources = " ".join(known[label].chunk.text for label in labels if label in known)
        if not sources:
            continue

        source_words = set(content_tokens(sources))
        support = sum(word in source_words for word in words) / len(words)
        if support < min_support:
            flag("unsupported_claim", f"support {support:.2f} < {min_support:.2f}")
        if missing := sorted(_numbers(claim) - _numbers(sources)):
            flag("number_not_in_source", f"not in cited sources: {', '.join(missing)}")

    return GroundingReport(passed=not findings, sentences_checked=checked, findings=tuple(findings))
