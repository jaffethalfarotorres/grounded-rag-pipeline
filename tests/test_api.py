import pytest
from fastapi.testclient import TestClient

from grounded_rag.api import app_from_env, create_app
from grounded_rag.pipeline import RAGPipeline
from tests.corpus import DOCS, POISONED


@pytest.fixture
def client() -> TestClient:
    rag = RAGPipeline()
    rag.ingest(DOCS)
    return TestClient(create_app(rag))


def test_query_returns_a_typed_grounded_answer(client):
    response = client.post("/query", json={"question": "How long do refunds take?"})
    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "1.0"
    assert body["status"] == "answered"
    assert body["citations"][0]["doc_id"] == "refunds"


@pytest.mark.parametrize(
    "payload",
    [{"question": ""}, {"question": "refunds", "top_k": 0}, {"question": "x" * 2001}, {}],
)
def test_invalid_queries_are_rejected_by_the_schema(client, payload):
    assert client.post("/query", json=payload).status_code == 422


def test_ingest_reports_quarantined_passages(client):
    response = client.post("/ingest", json={"documents": [POISONED.model_dump()]})
    assert response.status_code == 200
    quarantined = response.json()["quarantined"]
    assert quarantined[0]["rules"] == ["instruction_override"]


def test_ingest_rejects_invalid_document_ids(client):
    bad = {**DOCS[0].model_dump(), "doc_id": "Not A Slug"}
    assert client.post("/ingest", json={"documents": [bad]}).status_code == 422


def test_health_reports_the_index(client):
    body = client.get("/health").json()
    assert body == {
        "status": "ok",
        "documents": 3,
        "chunks": 3,
        "index_version": 1,
        "generator": "extractive",
        "embedder": "hashing",
    }


def test_metrics_endpoint_serves_prometheus_text(client):
    client.post("/query", json={"question": "How long do refunds take?"})
    response = client.get("/metrics")
    assert response.headers["content-type"].startswith("text/plain")
    assert 'rag_queries_total{status="answered"} 1' in response.text


def test_app_factory_loads_a_corpus_directory_from_the_environment(tmp_path):
    (tmp_path / "sla.md").write_text("# SLA\n\nMonthly uptime target is 99.9%.", encoding="utf-8")
    app = app_from_env({"RAG_CORPUS_DIR": str(tmp_path), "RAG_BACKEND": "none"})
    body = TestClient(app).get("/health").json()
    assert body["documents"] == 1


def test_embedder_comes_from_the_environment(tmp_path):
    env = {"RAG_CORPUS_DIR": str(tmp_path), "RAG_EMBEDDER": "hashing"}
    assert TestClient(app_from_env(env)).get("/health").json()["embedder"] == "hashing"
    with pytest.raises(ValueError, match="unknown embedder"):
        app_from_env({**env, "RAG_EMBEDDER": "word2vec"})
