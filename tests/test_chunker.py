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


def doc(text: str) -> Document:
    return Document(doc_id="d", title="D", text=text, metadata={"category": "test"})


@settings(max_examples=200, deadline=None)
@given(blocks=BLOCKS, max_chars=st.integers(min_value=40, max_value=400))
def test_chunks_respect_the_size_limit(blocks, max_chars):
    chunks = chunk_document(doc("\n\n".join(blocks)), max_chars=max_chars)
    assert all(len(chunk.text) <= max_chars for chunk in chunks)


@settings(max_examples=200, deadline=None)
@given(blocks=BLOCKS, max_chars=st.integers(min_value=40, max_value=400))
def test_no_word_is_lost_or_reordered(blocks, max_chars):
    text = "\n\n".join(blocks)
    chunks = chunk_document(doc(text), max_chars=max_chars)
    assert " ".join(chunk.text for chunk in chunks).split() == text.split()


@settings(max_examples=200, deadline=None)
@given(blocks=BLOCKS, max_chars=st.integers(min_value=40, max_value=400))
def test_short_paragraphs_are_never_split(blocks, max_chars):
    chunks = chunk_document(doc("\n\n".join(blocks)), max_chars=max_chars)
    for block in blocks:
        if len(block) <= max_chars:
            assert any(block in chunk.text for chunk in chunks)


@settings(max_examples=200, deadline=None)
@given(blocks=BLOCKS, max_chars=st.integers(min_value=40, max_value=400))
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
            for block in later.text.split("\n\n"):
                run.append(block)
                if not block.startswith("#"):
                    break
            else:
                continue
            break
        if not run[-1].startswith("#"):  # a content block follows the heading run
            assert len("\n\n".join(run)) > max_chars, run


def test_chunks_carry_ids_metadata_and_section():
    text = "# Pricing\n\nPro costs 49 USD.\n\n## Enterprise\n\nContact sales."
    chunks = chunk_document(doc(text), max_chars=40)
    assert [chunk.chunk_id for chunk in chunks] == [f"d#{i}" for i in range(len(chunks))]
    assert all(chunk.metadata["category"] == "test" for chunk in chunks)
    assert chunks[-1].metadata["section"] == "Enterprise"


def test_empty_document_yields_no_chunks():
    assert chunk_document(doc("  \n\n  "), max_chars=100) == []
