"""Typed envelopes for every hand-off in the pipeline.

Frozen Pydantic models that reject unknown fields. ``RAGAnswer`` checks its own
invariants on construction, so a result cannot claim "answered" without a passing
grounding report and at least one citation, and a blocked or refused result can
never carry answer text.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"

ModelTier = Literal["small", "large"]
Generator = Literal["llm", "extractive", "none"]
GroundingRule = Literal[
    "fabricated_citation",
    "uncited_claim",
    "unsupported_claim",
    "number_not_in_source",
    "empty_answer",
]


class _Envelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --- prompt defense ----------------------------------------------------------------------


class Finding(_Envelope):
    """One prompt-defense rule that matched."""

    rule: str
    excerpt: str


class DefenseReport(_Envelope):
    blocked: bool
    findings: tuple[Finding, ...] = ()


# --- corpus ------------------------------------------------------------------------------


class Document(_Envelope):
    """A source document as loaded, before chunking."""

    doc_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    title: str
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)


class Chunk(_Envelope):
    """A retrievable slice of a document. ``chunk_id`` is ``<doc_id>#<index>``."""

    chunk_id: str
    doc_id: str
    title: str
    index: int = Field(ge=0)
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_id(self) -> Self:
        if self.chunk_id != f"{self.doc_id}#{self.index}":
            raise ValueError("chunk_id must be '<doc_id>#<index>'")
        return self


class Hit(_Envelope):
    """A retrieved chunk with the label the model cites it by (S1, S2, ...)."""

    source_id: str = Field(pattern=r"^S[1-9]\d*$")
    chunk: Chunk
    score: float
    vector_rank: int | None = None
    bm25_rank: int | None = None


class QuarantinedChunk(_Envelope):
    chunk_id: str
    rules: tuple[str, ...]


class IngestReport(_Envelope):
    documents: int
    chunks_indexed: int
    quarantined: tuple[QuarantinedChunk, ...] = ()
    index_version: int


# --- answers -----------------------------------------------------------------------------


class Citation(_Envelope):
    source_id: str
    chunk_id: str
    doc_id: str
    title: str
    excerpt: str


class GroundingFinding(_Envelope):
    rule: GroundingRule
    sentence: str
    detail: str


class GroundingReport(_Envelope):
    """Outcome of the deterministic check between the model's answer and the sources."""

    passed: bool
    sentences_checked: int
    findings: tuple[GroundingFinding, ...] = ()


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    REFUSED = "refused"
    BLOCKED = "blocked"


class RAGAnswer(_Envelope):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    status: AnswerStatus
    question: str
    answer: str | None
    generator: Generator
    citations: tuple[Citation, ...] = ()
    grounding: GroundingReport | None = None
    rejected: GroundingReport | None = None
    defense: DefenseReport
    cached: bool = False
    trace: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        if self.defense.blocked:
            if self.status is not AnswerStatus.BLOCKED or self.answer or self.grounding:
                raise ValueError("a defense block means status 'blocked' and nothing else")
            return self
        if self.status is AnswerStatus.ANSWERED:
            if not self.answer:
                raise ValueError("an answered result needs answer text")
            if self.grounding is None or not self.grounding.passed:
                raise ValueError("an answered result needs a passing grounding report")
            if not self.citations:
                raise ValueError("an answered result needs at least one citation")
        elif self.status is AnswerStatus.REFUSED:
            if self.answer is not None or self.citations:
                raise ValueError("a refusal carries no answer and no citations")
        else:
            if self.answer is not None:
                raise ValueError("a blocked result never carries answer text")
            if self.grounding is None or self.grounding.passed:
                raise ValueError("a grounding block needs a failed grounding report")
        return self
