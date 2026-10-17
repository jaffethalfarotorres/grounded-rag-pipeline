import math

import pytest

from grounded_rag.embedder import HashingEmbedder, SentenceTransformerEmbedder
from grounded_rag.envelopes import Chunk
from grounded_rag.retrieval import BM25Index, HybridRetriever, rrf
from grounded_rag.text import content_tokens, coverage, tokens


def chunk(doc_id: str, text: str, index: int = 0, **metadata: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}#{index}",
        doc_id=doc_id,
        title=doc_id.title(),
        index=index,
        text=text,
        metadata=metadata,
    )


CORPUS = [
    chunk("refunds", "Refunds are issued within 14 days of a written request.", category="billing"),
    chunk("pricing", "The Pro plan costs 49 USD per month, billed monthly.", category="billing"),
    chunk("sla", "Monthly uptime target is 99.9% for every paid plan.", category="ops"),
    chunk("security", "Customer data is encrypted at rest with AES-256.", category="security"),
]


# --- text helpers ----------------------------------------------------------------------------


def test_tokens_keep_decimal_numbers_and_drop_case_and_symbols():
    assert tokens("Uptime: 99.9% (Pro)") == ["uptime", "99.9", "pro"]


def test_content_tokens_drop_stopwords():
    assert content_tokens("What is the refund window for the Pro plan?") == [
        "refund",
        "window",
        "pro",
        "plan",
    ]


def test_content_tokens_fold_plurals_so_questions_match_documents():
    assert content_tokens("refunds plans address") == ["refund", "plan", "address"]


def test_coverage_is_the_share_of_question_words_found_in_the_text():
    assert coverage("refund window", "Refunds have a 14 day window.") == 1.0
    assert coverage("refund deadline", "Refunds have a 14 day window.") == 0.5
    assert coverage("the of and", "anything") == 0.0


# --- embedders -------------------------------------------------------------------------------


def test_hashing_embedder_is_deterministic_and_unit_length():
    embedder = HashingEmbedder(dim=256)
    first, again = embedder.embed(["refund policy"]), embedder.embed(["refund policy"])
    assert first == again
    assert math.isclose(math.fsum(x * x for x in first[0]), 1.0, rel_tol=1e-9)


def test_hashing_embedder_puts_related_text_closer_than_unrelated_text():
    embedder = HashingEmbedder()
    query, related, unrelated = embedder.embed(
        ["refund request window", "refunds are issued after a request", "uptime target"]
    )

    def dot(a, b):
        return math.fsum(x * y for x, y in zip(a, b, strict=True))

    assert dot(query, related) > dot(query, unrelated)


def test_empty_text_embeds_to_the_zero_vector():
    assert set(HashingEmbedder(dim=8).embed([""])[0]) == {0.0}


def test_sentence_transformer_adapter_normalizes_and_converts_to_lists():
    class FakeModel:
        def get_sentence_embedding_dimension(self):
            return 2

        def encode(self, sentences, normalize_embeddings):
            assert normalize_embeddings is True
            return [[0.6, 0.8] for _ in sentences]

    embedder = SentenceTransformerEmbedder(FakeModel(), name="fake")
    assert embedder.dim == 2
    assert embedder.embed(["a", "b"]) == [[0.6, 0.8], [0.6, 0.8]]


# --- BM25 and fusion ---------------------------------------------------------------------------


def test_bm25_ranks_matching_chunks_and_ignores_the_rest():
    index = BM25Index()
    for item in CORPUS:
        index.add(item.chunk_id, content_tokens(item.text))
    ranked = index.search(content_tokens("refund request"))
    assert [chunk_id for chunk_id, _ in ranked] == ["refunds#0"]


def test_bm25_forgets_removed_chunks():
    index = BM25Index()
    index.add("a#0", ["refund"])
    index.remove("a#0")
    assert index.search(["refund"]) == []


def test_rrf_rewards_agreement_between_rankings():
    assert rrf([["a", "b"], ["b", "c"]])[0][0] == "b"


# --- hybrid retriever ---------------------------------------------------------------------------


@pytest.fixture
def retriever() -> HybridRetriever:
    retriever = HybridRetriever(HashingEmbedder())
    retriever.add(CORPUS)
    return retriever


def test_hybrid_search_returns_labelled_hits_best_first(retriever):
    hits = retriever.search("How long do refunds take?", top_k=2)
    assert hits[0].chunk.chunk_id == "refunds#0"
    assert [hit.source_id for hit in hits] == ["S1", "S2"]
    assert hits[0].bm25_rank == 1


def test_metadata_filters_restrict_the_candidates(retriever):
    hits = retriever.search("plan", top_k=4, filters={"category": "ops"})
    assert [hit.chunk.doc_id for hit in hits] == ["sla"]


def test_reingesting_a_document_replaces_all_of_its_old_chunks(retriever):
    old_version = [CORPUS[0], chunk("refunds", "Old appendix about refund exceptions.", index=1)]
    retriever.add(old_version)
    assert len(retriever) == len(CORPUS) + 1
    # The new version is a single chunk: the old appendix (refunds#1) must disappear too.
    retriever.add([chunk("refunds", "Refunds now take 30 days.", category="billing")])
    assert len(retriever) == len(CORPUS)
    hits = retriever.search("refund exceptions appendix", top_k=4)
    assert all("appendix" not in hit.chunk.text for hit in hits)
    assert "30 days" in retriever.search("refunds", top_k=1)[0].chunk.text


def test_removing_a_document_removes_it_from_both_indexes(retriever):
    retriever.remove_document("refunds")
    assert all(hit.chunk.doc_id != "refunds" for hit in retriever.search("refund request"))
