import itertools

from hypothesis import given, settings
from hypothesis import strategies as st

from grounded_rag.chunker import chunk_document
from grounded_rag.envelopes import Document

WORDS = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=12)
SENTENCES = st.lists(WORDS, min_size=1, max_size=12).map(lambda ws: " ".join(ws) + ".")
# Sentences joined by spaces or single line breaks (addresses, lists): reformatting a
# paragraph that did not need splitting must be visible to the tests.
PARAGRAPHS = st.tuples(
    st.lists(SENTENCES, min_size=1, max_size=6), st.sampled_from([" ", "\n"])
).map(lambda parts: parts[1].join(parts[0]))
HEADINGS = WORDS.map(lambda w: f"## {w}")
BLOCKS = st.lists(st.one_of(PARAGRAPHS, HEADINGS), min_size=1, max_size=15)
LIMITS = st.integers(min_value=40, max_value=400)


def doc(text: str) -> Document:
    return Document(doc_id="d", title="D", text=text, metadata={"category": "test"})


def own_text(chunk) -> str:
    """A chunk's text without the overlap it repeats from the previous chunk."""
    return chunk.text[chunk.overlap :]


@settings(max_examples=200, deadline=None)
@given(blocks=BLOCKS, max_chars=LIMITS)
def test_chunks_respect_the_size_limit_overlap_included(blocks, max_chars):
    chunks = chunk_document(doc("\n\n".join(blocks)), max_chars=max_chars)
    assert all(len(chunk.text) <= max_chars for chunk in chunks)


@settings(max_examples=200, deadline=None)
@given(blocks=BLOCKS, max_chars=LIMITS)
def test_no_word_is_lost_or_reordered_once_overlap_is_removed(blocks, max_chars):
    text = "\n\n".join(blocks)
    chunks = chunk_document(doc(text), max_chars=max_chars)
    assert " ".join(own_text(chunk) for chunk in chunks).split() == text.split()


@settings(max_examples=200, deadline=None)
@given(blocks=BLOCKS, max_chars=LIMITS)
def test_overlap_repeats_the_end_of_the_previous_chunk(blocks, max_chars):
    chunks = chunk_document(doc("\n\n".join(blocks)), max_chars=max_chars)
    assert chunks[0].overlap == 0 if chunks else True
    for previous, chunk in itertools.pairwise(chunks):
        if chunk.overlap:
            repeated = chunk.text[: chunk.overlap].removesuffix("\n\n")
            assert previous.text.endswith(repeated)
            assert not own_text(chunk).startswith("#")
            assert not repeated.startswith("#")


@settings(max_examples=200, deadline=None)
@given(blocks=BLOCKS, max_chars=LIMITS)
def test_short_paragraphs_are_never_split(blocks, max_chars):
    chunks = chunk_document(doc("\n\n".join(blocks)), max_chars=max_chars)
    for block in blocks:
        if len(block) <= max_chars:
            assert any(block in chunk.text for chunk in chunks)


@settings(max_examples=200, deadline=None)
@given(blocks=BLOCKS, max_chars=LIMITS)
def test_a_heading_only_ends_a_chunk_when_it_cannot_travel_with_its_content(blocks, max_chars):
    """A heading may end a chunk only if it, the rest of its heading run and the first
    content block after it cannot fit in one chunk (stacked headings can make that so)."""
    chunks = chunk_document(doc("\n\n".join(blocks)), max_chars=max_chars)
    for position, chunk in enumerate(chunks[:-1]):
        last_block = chunk.text.split("\n\n")[-1]
        if not last_block.startswith("#"):
            continue
        run = [last_block]
        for later in chunks[position + 1 :]:
            for block in own_text(later).split("\n\n"):
                run.append(block)
                if not block.startswith("#"):
                    break
            else:
                continue
            break
        if not run[-1].startswith("#"):  # a content block follows the heading run
            assert len("\n\n".join(run)) > max_chars, run


def test_overlap_carries_the_last_sentence_into_the_next_chunk_of_a_section():
    text = (
        "## Refunds\n\nRefunds take 14 days. Annual plans are pro rata.\n\n"
        "Overage is not refundable."
    )
    chunks = chunk_document(doc(text), max_chars=80)
    assert len(chunks) == 2
    assert chunks[1].text == "Annual plans are pro rata.\n\nOverage is not refundable."
    assert chunks[1].overlap == len("Annual plans are pro rata.\n\n")


def test_no_overlap_into_a_new_section():
    text = (
        "## Refunds\n\nRefunds take 14 days of a request.\n\n"
        "## Support\n\nEmail support answers in 24 hours."
    )
    chunks = chunk_document(doc(text), max_chars=60)
    assert [chunk.overlap for chunk in chunks] == [0, 0]
    assert chunks[1].text.startswith("## Support")


def test_a_heading_that_ends_a_chunk_is_not_repeated_as_overlap():
    # The heading holds a sentence break ("Terms. Net 30 days"); its tail must not leak
    # into the next chunk as if it were body text.
    text = "Alpha beta gamma.\n\n## Terms. Net 30 days\n\nDelta epsilon zeta eta."
    chunks = chunk_document(doc(text), max_chars=40)
    assert [chunk.text for chunk in chunks] == [
        "Alpha beta gamma.\n\n## Terms. Net 30 days",
        "Delta epsilon zeta eta.",
    ]
    assert chunks[1].overlap == 0


def test_overlap_can_be_turned_off():
    text = (
        "## Refunds\n\nRefunds take 14 days. Annual plans are pro rata.\n\n"
        "Overage is not refundable."
    )
    chunks = chunk_document(doc(text), max_chars=80, overlap_sentences=0)
    assert [chunk.overlap for chunk in chunks] == [0, 0]


def test_chunks_carry_ids_metadata_and_section():
    text = "# Pricing\n\nPro costs 49 USD.\n\n## Enterprise\n\nContact sales."
    chunks = chunk_document(doc(text), max_chars=40)
    assert [chunk.chunk_id for chunk in chunks] == [f"d#{i}" for i in range(len(chunks))]
    assert all(chunk.metadata["category"] == "test" for chunk in chunks)
    assert chunks[-1].metadata["section"] == "Enterprise"


def test_empty_document_yields_no_chunks():
    assert chunk_document(doc("  \n\n  "), max_chars=100) == []
