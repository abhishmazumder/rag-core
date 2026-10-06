import json
import uuid
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from rag_core.api import dependencies, routes
from rag_core.api.app import create_app
from rag_core.config import Settings
from rag_core.domain.models.document_chunk_metadata import DocumentChunkMetadata
from rag_core.domain.models.embedding import EmbeddingRequest, EmbeddingResponse
from rag_core.domain.models.rag import RAGResponse
from rag_core.domain.models.vector_search import VectorSearchResult


@pytest.fixture
def overrides():
    app = create_app()
    mocks = {
        "store": Mock(),
        "embedding": Mock(),
        "response": Mock(),
        "credential": Mock(),
        "settings": Settings(
            azure_search_endpoint="https://search.example.net",
            azure_search_index_name="idx",
            azure_ai_foundry_embedding_dimensions=8,
        ),
    }
    app.dependency_overrides[dependencies.get_vector_store] = lambda: mocks["store"]
    app.dependency_overrides[dependencies.get_embedding_model] = lambda: mocks["embedding"]
    app.dependency_overrides[dependencies.get_response_model] = lambda: mocks["response"]
    app.dependency_overrides[dependencies.get_credential] = lambda: mocks["credential"]
    app.dependency_overrides[dependencies.get_settings] = lambda: mocks["settings"]
    return TestClient(app), mocks


def _ingest(client, text: str = "hello world", candidate_id: str = "c1"):
    return client.post("/documents", json={"candidate_id": candidate_id, "text": text})


def _chunks(monkeypatch, chunks: list[str]):
    chunker = Mock(return_value=chunks)
    monkeypatch.setattr(routes, "recursive_chunks", chunker)
    return chunker


def test_post_documents_ingests_chunks(overrides, monkeypatch) -> None:
    client, mocks = overrides
    chunker = _chunks(monkeypatch, ["x", "y", "z"])
    embeddings = [[0.1], [0.2], [0.3]]
    mocks["embedding"].embed.return_value = EmbeddingResponse(embeddings=embeddings)

    response = _ingest(client, "raw text")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"candidate_id", "document_id", "chunk_ids"}
    assert body["candidate_id"] == "c1"
    uuid.UUID(body["document_id"])
    chunker.assert_called_once_with("raw text", chunk_size_chars=1000, overlap_chars=100)
    mocks["embedding"].embed.assert_called_once_with(EmbeddingRequest(texts=["x", "y", "z"]))
    mocks["store"].upsert_many.assert_called_once()
    documents = mocks["store"].upsert_many.call_args.args[0]
    assert [d.id for d in documents] == body["chunk_ids"]
    assert [d.text for d in documents] == ["x", "y", "z"]
    assert [d.embedding for d in documents] == embeddings
    assert [d.chunk_index for d in documents] == [0, 1, 2]
    for document in documents:
        assert document.metadata == DocumentChunkMetadata(
            source_document_id=body["document_id"],
            source_name=body["document_id"],
            document_type="text",
            candidate_id="c1",
            document_id=body["document_id"],
        )


def test_post_documents_with_no_chunks_skips_embedding_and_store(overrides, monkeypatch) -> None:
    client, mocks = overrides
    _chunks(monkeypatch, [])

    response = _ingest(client, "   ")

    assert response.status_code == 200
    assert response.json()["chunk_ids"] == []
    assert response.json()["candidate_id"] == "c1"
    mocks["embedding"].embed.assert_not_called()
    mocks["store"].upsert_many.assert_not_called()


def test_post_documents_embedding_count_mismatch_fails_before_upsert(
    overrides, monkeypatch
) -> None:
    client, mocks = overrides
    _chunks(monkeypatch, ["x", "y", "z"])
    mocks["embedding"].embed.return_value = EmbeddingResponse(embeddings=[[0.1], [0.2]])

    with pytest.raises(ValueError, match="returned 2 embeddings for 3 chunks"):
        _ingest(client)

    mocks["store"].upsert_many.assert_not_called()


def test_post_documents_chunk_ids_are_deterministic_per_document_id(overrides, monkeypatch) -> None:
    client, mocks = overrides
    _chunks(monkeypatch, ["x", "y"])
    mocks["embedding"].embed.return_value = EmbeddingResponse(embeddings=[[0.1], [0.2]])
    monkeypatch.setattr(routes.uuid, "uuid4", lambda: uuid.UUID(int=1))

    first = _ingest(client).json()
    repeated = _ingest(client).json()

    assert first == repeated
    assert len(set(first["chunk_ids"])) == 2
    assert all(chunk_id.startswith("chunk-") for chunk_id in first["chunk_ids"])
    name = json.dumps(
        ["c1", str(uuid.UUID(int=1)), None, "recursive", 1000, 100, 0], separators=(",", ":")
    )
    assert first["chunk_ids"][0] == f"chunk-{uuid.uuid5(uuid.NAMESPACE_DNS, name)}"


def test_post_documents_generates_new_document_id_per_request(overrides, monkeypatch) -> None:
    client, mocks = overrides
    _chunks(monkeypatch, ["x"])
    mocks["embedding"].embed.return_value = EmbeddingResponse(embeddings=[[0.1]])

    first = _ingest(client).json()
    second = _ingest(client).json()

    assert first["document_id"] != second["document_id"]
    assert set(first["chunk_ids"]).isdisjoint(second["chunk_ids"])


@pytest.mark.parametrize(
    "payload",
    [{"text": "x"}, {"candidate_id": "c1"}, {"candidate_id": "", "text": "x"}],
)
def test_post_documents_validates_request(overrides, payload) -> None:
    client, _ = overrides
    assert client.post("/documents", json=payload).status_code == 422


def test_post_query_vector_returns_answer_and_chunks(overrides, mocker) -> None:
    client, mocks = overrides
    metadata = DocumentChunkMetadata(
        source_document_id="s",
        source_name="n",
        document_type="text",
        candidate_id="c1",
        document_id="d1",
    )
    rag = mocker.patch.object(
        routes,
        "generate_rag_response",
        return_value=RAGResponse[DocumentChunkMetadata](
            answer="the answer",
            evidence=[
                VectorSearchResult[DocumentChunkMetadata](
                    id="chunk-1", text="t", score=0.5, metadata=metadata
                )
            ],
        ),
    )

    response = client.post(
        "/query",
        json={"query": "q", "candidate_id": "c1", "method": "vector", "top_k": 3},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "q"
    assert body["candidate_id"] == "c1"
    assert body["method"] == "vector"
    assert body["answer"] == "the answer"
    assert body["chunks"][0]["id"] == "chunk-1"
    assert body["chunks"][0]["score"] == 0.5
    assert body["chunks"][0]["metadata"]["document_id"] == "d1"
    kwargs = rag.call_args.kwargs
    assert kwargs["filters"] == {"candidate_id": "c1"}
    assert kwargs["top_k"] == 3
    assert kwargs["vector_store"] is mocks["store"]


@pytest.mark.parametrize("method", ["vector", "keyword", "hybrid", "semantic"])
def test_post_query_passes_method_to_rag_generation(overrides, mocker, method) -> None:
    client, _ = overrides
    rag = mocker.patch.object(
        routes,
        "generate_rag_response",
        return_value=RAGResponse[DocumentChunkMetadata](answer="a", evidence=[]),
    )

    response = client.post("/query", json={"query": "q", "candidate_id": "c", "method": method})

    assert response.status_code == 200
    assert response.json()["method"] == method
    assert rag.call_args.kwargs["method"] == method


def test_post_query_method_defaults_to_vector(overrides, mocker) -> None:
    client, _ = overrides
    rag = mocker.patch.object(
        routes,
        "generate_rag_response",
        return_value=RAGResponse[DocumentChunkMetadata](answer="a", evidence=[]),
    )

    response = client.post("/query", json={"query": "q", "candidate_id": "c"})

    assert response.status_code == 200
    assert response.json()["method"] == "vector"
    assert rag.call_args.kwargs["method"] == "vector"


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "q", "method": "vector"},
        {"query": "q", "candidate_id": "c", "method": "bogus"},
        {"query": "q", "candidate_id": "c", "method": "vector", "top_k": 0},
        {"candidate_id": "c", "method": "vector"},
    ],
)
def test_post_query_validates_request(overrides, payload) -> None:
    client, _ = overrides
    assert client.post("/query", json=payload).status_code == 422


def test_admin_index_setup_uses_canonical_provisioner(overrides, mocker) -> None:
    client, mocks = overrides
    build = mocker.patch.object(routes, "build_azure_search_index")
    create = mocker.patch.object(routes, "create_or_update_azure_search_index")

    response = client.post("/admin/index/setup")

    assert response.status_code == 200
    assert response.json() == {"index_name": "idx"}
    build.assert_called_once_with("idx", 8)
    create.assert_called_once_with(
        "https://search.example.net", build.return_value, mocks["credential"]
    )


def test_admin_index_setup_requires_configuration(overrides) -> None:
    client, _ = overrides
    client.app.dependency_overrides[dependencies.get_settings] = lambda: Settings(_env_file=None)
    assert client.post("/admin/index/setup").status_code == 503


def test_admin_index_clear_clears_store(overrides) -> None:
    client, mocks = overrides
    response = client.post("/admin/index/clear")
    assert response.status_code == 200
    assert response.json() == {"status": "cleared"}
    mocks["store"].clear.assert_called_once_with()


def test_delete_document_deletes_by_document_id(overrides) -> None:
    client, mocks = overrides
    mocks["store"].delete_by_document_id.return_value = 3
    response = client.delete("/documents/doc-1")
    assert response.status_code == 200
    assert response.json() == {"document_id": "doc-1", "deleted_chunks": 3}
    mocks["store"].delete_by_document_id.assert_called_once_with("doc-1")


def test_delete_chunk_deletes_by_id(overrides) -> None:
    client, mocks = overrides
    response = client.delete("/chunks/chunk-1")
    assert response.status_code == 200
    assert response.json() == {"chunk_id": "chunk-1", "deleted": True}
    mocks["store"].delete.assert_called_once_with("chunk-1")


def test_unconfigured_dependencies_return_503(mocker) -> None:
    mocker.patch.object(dependencies, "get_settings", return_value=Settings(_env_file=None))
    client = TestClient(create_app())
    response = client.post("/admin/index/clear")
    assert response.status_code == 503
