import json

import pytest
from pydantic import ValidationError

from grounded_rag.envelopes import (
    SCHEMA_VERSION,
    AnswerStatus,
    Chunk,
    Citation,
    DefenseReport,
    Document,
    Finding,
    GroundingFinding,
    GroundingReport,
    RAGAnswer,
)

CLEAN = DefenseReport(blocked=False)
PASSED = GroundingReport(passed=True, sentences_checked=1)
FAILED = GroundingReport(
    passed=False,
    sentences_checked=1,
    findings=(GroundingFinding(rule="fabricated_citation", sentence="x [S9]", detail="S9"),),
)
CITATION = Citation(source_id="S1", chunk_id="sla#0", doc_id="sla", title="SLA", excerpt="99.9%")


def answer(**overrides) -> RAGAnswer:
    fields = {
        "status": AnswerStatus.ANSWERED,
        "question": "What is the SLA?",
        "answer": "Uptime is 99.9% [S1].",
        "generator": "extractive",
        "citations": (CITATION,),
        "grounding": PASSED,
        "defense": CLEAN,
    }
    return RAGAnswer(**{**fields, **overrides})


def test_document_ids_are_slugs():
    Document(doc_id="refund-policy.v2", title="Refunds", text="...")
    with pytest.raises(ValidationError):
        Document(doc_id="Refund Policy!", title="Refunds", text="...")


def test_chunk_id_must_belong_to_its_document():
    Chunk(chunk_id="sla#0", doc_id="sla", title="SLA", index=0, text="t")
    with pytest.raises(ValidationError, match="chunk_id"):
        Chunk(chunk_id="pricing#0", doc_id="sla", title="SLA", index=0, text="t")


def test_answered_result_round_trips_through_json_with_schema_version():
    result = answer()
    payload = json.loads(result.model_dump_json())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert RAGAnswer.model_validate(payload) == result


def test_answered_result_requires_a_passing_grounding_report():
    with pytest.raises(ValidationError, match="grounding"):
        answer(grounding=FAILED)


def test_answered_result_requires_citations():
    with pytest.raises(ValidationError, match="citation"):
        answer(citations=())


def test_blocked_result_never_carries_an_answer():
    with pytest.raises(ValidationError, match="answer"):
        answer(status=AnswerStatus.BLOCKED, grounding=FAILED)


def test_grounding_block_requires_a_failed_report():
    with pytest.raises(ValidationError, match="grounding"):
        answer(status=AnswerStatus.BLOCKED, answer=None, citations=(), grounding=PASSED)


def test_defense_block_means_status_blocked_and_nothing_else():
    report = DefenseReport(blocked=True, findings=(Finding(rule="r", excerpt="x"),))
    with pytest.raises(ValidationError, match="defense"):
        answer(defense=report)
    blocked = answer(
        status=AnswerStatus.BLOCKED,
        answer=None,
        citations=(),
        grounding=None,
        generator="none",
        defense=report,
    )
    assert blocked.status is AnswerStatus.BLOCKED


def test_refusal_carries_no_answer_and_no_citations():
    refused = answer(
        status=AnswerStatus.REFUSED, answer=None, citations=(), grounding=None, generator="none"
    )
    assert refused.answer is None
    with pytest.raises(ValidationError, match="refus"):
        answer(status=AnswerStatus.REFUSED, grounding=None)
