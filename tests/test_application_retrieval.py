from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from rag_core.application.retrieval import retrieve
from rag_core.domain.interfaces.embedding_model import EmbeddingModel
from rag_core.domain.models.embedding import EmbeddingRequest, EmbeddingResponse
from rag_core.domain.models.vector_search import (
    VectorSearchRequest,
    VectorSearchResponse,
)
from rag_core.infrastructure.vectorstores.azure_search import AzureAISearchVectorStore


class EvidenceMetadata(BaseModel):
    source: str


class FakeEmbeddingModel:
    def __init__(self, response: EmbeddingResponse) -> None:
        self.response = response
        self.received_request: EmbeddingRequest | None = None

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.received_request = request
        return self.response


def test_retrieve_embeds_query_and_delegates_vector_search() -> None:
    embedding_model = FakeEmbeddingModel(EmbeddingResponse(embeddings=[[0.1, 0.2, 0.3]]))
    vector_store = Mock(spec=AzureAISearchVectorStore)
    response = VectorSearchResponse[EvidenceMetadata](results=[])
    vector_store.vector_search.return_value = response
    filters = {"source": "guide.pdf"}

    result = retrieve(
        embedding_model=embedding_model,
        vector_store=vector_store,
        query="What does the guide say?",
        top_k=5,
        filters=filters,
    )

    assert isinstance(embedding_model, EmbeddingModel)
    assert embedding_model.received_request == EmbeddingRequest(texts=["What does the guide say?"])
    vector_store.vector_search.assert_called_once_with(
        VectorSearchRequest(
            query_vector=[0.1, 0.2, 0.3],
            top_k=5,
            filters=filters,
        )
    )
    assert result is response


@pytest.mark.parametrize("embeddings", [[], [[0.1], [0.2]]])
def test_retrieve_requires_exactly_one_embedding(embeddings: list[list[float]]) -> None:
    embedding_response = Mock(spec=EmbeddingResponse)
    embedding_response.embeddings = embeddings
    embedding_model = FakeEmbeddingModel(embedding_response)
    vector_store = Mock(spec=AzureAISearchVectorStore)

    with pytest.raises(ValueError, match="requires exactly one embedding"):
        retrieve(
            embedding_model=embedding_model,
            vector_store=vector_store,
            query="A query",
            top_k=3,
        )

    vector_store.vector_search.assert_not_called()


def _store() -> Mock:
    return Mock(spec=AzureAISearchVectorStore)


def _embedder() -> Mock:
    model = Mock()
    model.embed.return_value = EmbeddingResponse(embeddings=[[0.1, 0.2]])
    return model


def test_retrieve_defaults_to_vector_search() -> None:
    store, embedder = _store(), _embedder()

    result = retrieve(embedding_model=embedder, vector_store=store, query="q", top_k=2)

    store.vector_search.assert_called_once_with(
        VectorSearchRequest(query_vector=[0.1, 0.2], top_k=2, filters=None)
    )
    assert result is store.vector_search.return_value
    embedder.embed.assert_called_once_with(EmbeddingRequest(texts=["q"]))


def test_retrieve_explicit_vector_search() -> None:
    store, embedder = _store(), _embedder()
    filters = {"candidate_id": "c1"}

    retrieve(
        embedding_model=embedder,
        vector_store=store,
        query="q",
        top_k=2,
        filters=filters,
        method="vector",
    )

    store.vector_search.assert_called_once_with(
        VectorSearchRequest(query_vector=[0.1, 0.2], top_k=2, filters=filters)
    )
    store.keyword_search.assert_not_called()
    store.hybrid_search.assert_not_called()
    store.semantic_search.assert_not_called()


def test_retrieve_keyword_search_does_not_embed() -> None:
    store, embedder = _store(), _embedder()
    filters = {"candidate_id": "c1"}

    result = retrieve(
        embedding_model=embedder,
        vector_store=store,
        query="q",
        top_k=3,
        filters=filters,
        method="keyword",
    )

    store.keyword_search.assert_called_once_with("q", top_k=3, filters=filters)
    assert result is store.keyword_search.return_value
    embedder.embed.assert_not_called()
    store.vector_search.assert_not_called()


def test_retrieve_semantic_search_does_not_embed() -> None:
    store, embedder = _store(), _embedder()
    filters = {"candidate_id": "c1"}

    result = retrieve(
        embedding_model=embedder,
        vector_store=store,
        query="q",
        top_k=3,
        filters=filters,
        method="semantic",
    )

    store.semantic_search.assert_called_once_with("q", top_k=3, filters=filters)
    assert result is store.semantic_search.return_value
    embedder.embed.assert_not_called()
    store.vector_search.assert_not_called()


def test_retrieve_hybrid_search_embeds_query() -> None:
    store, embedder = _store(), _embedder()
    filters = {"candidate_id": "c1"}

    result = retrieve(
        embedding_model=embedder,
        vector_store=store,
        query="q",
        top_k=3,
        filters=filters,
        method="hybrid",
    )

    embedder.embed.assert_called_once_with(EmbeddingRequest(texts=["q"]))
    store.hybrid_search.assert_called_once_with("q", [0.1, 0.2], top_k=3, filters=filters)
    assert result is store.hybrid_search.return_value
    store.vector_search.assert_not_called()


def test_retrieve_hybrid_requires_exactly_one_embedding() -> None:
    store = _store()
    embedder = Mock()
    embedder.embed.return_value = EmbeddingResponse(embeddings=[[0.1], [0.2]])

    with pytest.raises(ValueError, match="requires exactly one embedding"):
        retrieve(embedding_model=embedder, vector_store=store, query="q", top_k=1, method="hybrid")

    store.hybrid_search.assert_not_called()


def test_retrieve_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unsupported search method"):
        retrieve(
            embedding_model=_embedder(),
            vector_store=_store(),
            query="q",
            top_k=1,
            method="bogus",  # type: ignore[arg-type]
        )
