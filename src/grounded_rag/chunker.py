"""Paragraph-aware chunking.

Rules, each pinned by a property-based test:

* no chunk is longer than ``max_chars``;
* no word is lost or reordered;
* a paragraph that fits in one chunk is never split across two;
* a heading stays with the content that follows it whenever both fit, so a
  chunk does not end on a dangling title;
* a chunk that continues a section starts by repeating the last sentence of the
  previous chunk (``overlap``), so a fact split across the boundary keeps its
  context. Overlap never crosses into a new section and never breaks the size limit,
  and removing it gives back the document word for word.

Paragraphs longer than ``max_chars`` are split at sentence boundaries, and a
sentence longer than ``max_chars`` at word boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from grounded_rag.envelopes import Chunk, Document

DEFAULT_MAX_CHARS = 800

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_HEADING = re.compile(r"^#{1,6}\s+(.*)$")


@dataclass(frozen=True)
class _Unit:
    text: str
    is_heading: bool
    section: str


def _pack(pieces: list[str], max_chars: int, sep: str = " ") -> list[str]:
    """Greedily join pieces with ``sep`` into strings of at most ``max_chars``."""
    out: list[str] = []
    for piece in pieces:
        if out and len(out[-1]) + len(sep) + len(piece) <= max_chars:
            out[-1] = f"{out[-1]}{sep}{piece}"
        else:
            out.append(piece)
    return out


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    for sentence in _SENTENCE_BREAK.split(paragraph):
        if len(sentence) <= max_chars:
            pieces.append(sentence)
            continue
        for word in sentence.split():
            # A single word longer than the limit is cut; nothing else ever is.
            pieces.extend(word[i : i + max_chars] for i in range(0, len(word), max_chars))
    return _pack(pieces, max_chars)


def _units(text: str, max_chars: int) -> list[_Unit]:
    units: list[_Unit] = []
    section = ""
    for block in (b.strip() for b in _PARAGRAPH_BREAK.split(text)):
        if not block:
            continue
        heading = _HEADING.match(block)
        if heading:
            section = heading.group(1).strip()
        pieces = [block] if len(block) <= max_chars else _split_long(block, max_chars)
        units.extend(_Unit(piece, bool(heading), section) for piece in pieces)
    return units


def _size(units: list[_Unit]) -> int:
    return len("\n\n".join(unit.text for unit in units))


def _overlap(
    previous: list[_Unit], group: list[_Unit], text: str, max_chars: int, sentences: int
) -> str:
    """The last ``sentences`` of the previous chunk, if this chunk continues its section.

    A section change always leaves a heading at the end of one chunk or the start of the
    next, so the two heading checks are what keep overlap inside a section."""
    if sentences <= 0 or group[0].is_heading or previous[-1].is_heading:
        return ""
    tail = " ".join(_SENTENCE_BREAK.split(previous[-1].text)[-sentences:])
    return tail if len(tail) + 2 + len(text) <= max_chars else ""


def chunk_document(
    document: Document, *, max_chars: int = DEFAULT_MAX_CHARS, overlap_sentences: int = 1
) -> list[Chunk]:
    groups: list[list[_Unit]] = []
    current: list[_Unit] = []
    for unit in _units(document.text, max_chars):
        if current and _size([*current, unit]) > max_chars:
            carry: list[_Unit] = []
            # Move trailing headings forward while they still fit with the new content.
            while (
                current
                and current[-1].is_heading
                and _size([current[-1], *carry, unit]) <= max_chars
            ):
                carry.insert(0, current.pop())
            if current:
                groups.append(current)
            current = carry
        current.append(unit)
    if current:
        groups.append(current)

    chunks = []
    for index, group in enumerate(groups):
        content = next((unit for unit in group if not unit.is_heading), group[0])
        section = {"section": content.section} if content.section else {}
        text = "\n\n".join(unit.text for unit in group)
        repeated = (
            _overlap(groups[index - 1], group, text, max_chars, overlap_sentences) if index else ""
        )
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}#{index}",
                doc_id=document.doc_id,
                title=document.title,
                index=index,
                text=f"{repeated}\n\n{text}" if repeated else text,
                metadata={**document.metadata, **section},
                overlap=len(repeated) + 2 if repeated else 0,
            )
        )
    return chunks
