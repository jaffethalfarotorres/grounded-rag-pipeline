import math
import sys
from types import SimpleNamespace

import pytest

from grounded_rag.embedder import (
    EmbedderUnavailable,
    FastEmbedEmbedder,
    HashingEmbedder,
    make_embedder,
)
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


class FakeVector(list):
    """Stands in for a numpy row: fastembed yields arrays that have ``tolist``."""

    def tolist(self):
        return list(self)


class FakeFastEmbedModel:
    """fastembed.TextEmbedding: passage_embed / query_embed yield one array per text."""

    embedding_size = 2

    def passage_embed(self, texts):
        return (FakeVector([3.0, 4.0]) for _ in texts)  # not unit length on purpose

    def query_embed(self, query):
        return (FakeVector([0.0, 5.0]) for _ in ([query] if isinstance(query, str) else query))


def test_fastembed_adapter_converts_arrays_and_normalizes_passages():
    embedder = FastEmbedEmbedder(FakeFastEmbedModel(), name="fake")
    assert embedder.dim == 2
    assert embedder.embed(["a", "b"]) == [pytest.approx([0.6, 0.8])] * 2


def test_fastembed_queries_use_the_query_side_of_the_model():
    embedder = FastEmbedEmbedder(FakeFastEmbedModel(), name="fake")
    assert embedder.embed_query("a question") == pytest.approx([0.0, 1.0])


def test_hashing_embedder_queries_match_its_text_embeddings():
    embedder = HashingEmbedder()
    assert embedder.embed_query("refund policy") == embedder.embed(["refund policy"])[0]


def test_fastembed_missing_sdk_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed", None)
    with pytest.raises(EmbedderUnavailable, match=r"grounded-rag-pipeline\[embeddings\]"):
        FastEmbedEmbedder.from_name()


def test_fastembed_from_name_builds_the_model_with_the_default_name(monkeypatch):
    built = []

    def text_embedding(model_name):
        built.append(model_name)
        return FakeFastEmbedModel()

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=text_embedding))
    embedder = FastEmbedEmbedder.from_name()
    assert built == ["BAAI/bge-small-en-v1.5"]
    assert embedder.name == "BAAI/bge-small-en-v1.5"


def test_retriever_embeds_queries_through_the_query_side():
    class Spy(HashingEmbedder):
        queries: list[str]

        def __init__(self):
            super().__init__()
            self.queries = []

        def embed_query(self, text):
            self.queries.append(text)
            return super().embed_query(text)

    spy = Spy()
    retriever = HybridRetriever(spy)
    retriever.add(CORPUS)
    retriever.search("How long do refunds take?")
    assert spy.queries == ["How long do refunds take?"]


def test_make_embedder_selects_by_name():
    assert make_embedder("hashing").name == "hashing"
    with pytest.raises(ValueError, match="unknown embedder"):
        make_embedder("word2vec")


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
