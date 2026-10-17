import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from grounded_rag.envelopes import Chunk, Hit
from grounded_rag.grounding import cited_ids, split_sentences, verify


def hit(source_id: str, doc_id: str, text: str) -> Hit:
    chunk = Chunk(chunk_id=f"{doc_id}#0", doc_id=doc_id, title=doc_id, index=0, text=text)
    return Hit(source_id=source_id, chunk=chunk, score=0.03)


HITS = [
    hit("S1", "refunds", "Refunds are issued within 14 days of a written request."),
    hit("S2", "pricing", "The Pro plan costs 49 USD per month, billed monthly."),
    hit("S3", "sla", "Monthly uptime target is 99.9% for every paid plan."),
]


def rules(answer: str) -> set[str]:
    return {finding.rule for finding in verify(answer, HITS).findings}


# --- passes ------------------------------------------------------------------------------------


def test_a_faithful_cited_answer_passes():
    report = verify("Refunds are issued within 14 days of a written request [S1].", HITS)
    assert report.passed
    assert report.sentences_checked == 1


def test_a_sentence_may_combine_several_sources():
    answer = "The Pro plan costs 49 USD per month and targets 99.9% uptime [S2, S3]."
    assert verify(answer, HITS).passed


def test_adjacent_citation_markers_are_read_like_a_list():
    answer = "The Pro plan costs 49 USD per month and targets 99.9% uptime [S2][S3]."
    assert verify(answer, HITS).passed


def test_citation_labels_do_not_dilute_support():
    answer = "Monthly uptime target is 99.9% for every paid plan [S3]."
    assert verify(answer, HITS, min_support=1.0).passed


def test_citation_labels_are_not_mistaken_for_numbers():
    assert "number_not_in_source" not in rules("Uptime target is 99.9% for every paid plan [S3].")


# --- blocks ------------------------------------------------------------------------------------


def test_fabricated_citation_is_blocked():
    assert "fabricated_citation" in rules("Refunds are issued within 14 days [S7].")


def test_uncited_claim_is_blocked():
    assert "uncited_claim" in rules("Refunds are issued within 14 days of a written request.")


def test_changed_number_is_blocked_even_when_the_words_match():
    found = rules("Refunds are issued within 30 days of a written request [S1].")
    assert found == {"number_not_in_source"}


def test_more_precise_number_than_the_source_is_blocked():
    assert "number_not_in_source" in rules("Monthly uptime target is 99.99% [S3].")


def test_claim_the_source_does_not_make_is_blocked():
    assert "unsupported_claim" in rules(
        "Refunds include a complimentary laptop and headphones [S1]."
    )


def test_a_true_claim_cited_to_the_wrong_source_is_blocked():
    # Supported by S2, but cited as S1: a citation must point at the text that supports it.
    assert rules("The Pro plan costs 49 USD per month [S1].") >= {"unsupported_claim"}


def test_empty_answer_is_blocked():
    assert rules("   ") == {"empty_answer"}


def test_every_sentence_is_checked_not_just_the_first():
    answer = (
        "Refunds are issued within 14 days of a written request [S1]. "
        "Enterprise customers also get a dedicated account manager."
    )
    report = verify(answer, HITS)
    assert not report.passed
    assert report.sentences_checked == 2
    assert report.findings[0].sentence.startswith("Enterprise")


# --- helpers -----------------------------------------------------------------------------------


def test_sentences_split_after_citations_and_keep_decimals_intact():
    answer = "Uptime is 99.9% [S3]. Refunds take 14 days [S1]."
    assert split_sentences(answer) == ["Uptime is 99.9% [S3].", "Refunds take 14 days [S1]."]


def test_cited_ids_are_returned_in_order_of_first_use():
    assert cited_ids("a [S2] b [S1, S2] c [S3][S1]") == ["S2", "S1", "S3"]


# --- properties ----------------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(source=st.sampled_from(HITS))
def test_any_source_sentence_quoted_with_its_own_label_passes(source):
    sentence = source.chunk.text.rstrip(".")
    assert verify(f"{sentence} [{source.source_id}].", HITS).passed


@settings(max_examples=100, deadline=None)
@given(source=st.sampled_from(HITS), bogus=st.integers(min_value=4, max_value=99))
def test_any_label_outside_the_retrieved_set_fails(source, bogus):
    sentence = source.chunk.text.rstrip(".")
    assert not verify(f"{sentence} [S{bogus}].", HITS).passed


@pytest.mark.parametrize("min_support", [0.0, 0.5, 1.0])
def test_min_support_is_respected(min_support):
    answer = "Refunds are issued within 14 days, usually faster [S1]."
    report = verify(answer, HITS, min_support=min_support)
    # 5 of 7 content words are in the source: all but "usually" and "faster".
    assert report.passed is (min_support <= 5 / 7)
