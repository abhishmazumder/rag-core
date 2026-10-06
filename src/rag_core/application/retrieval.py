from typing import Literal

from pydantic import BaseModel

from rag_core.domain.interfaces.embedding_model import EmbeddingModel
from rag_core.domain.models.embedding import EmbeddingRequest
from rag_core.domain.models.vector_search import (
    VectorSearchRequest,
    VectorSearchResponse,
)
from rag_core.infrastructure.vectorstores.azure_search import AzureAISearchVectorStore

SearchMethod = Literal["vector", "keyword", "hybrid", "semantic"]


def retrieve[TMetadata: BaseModel](
    embedding_model: EmbeddingModel,
    vector_store: AzureAISearchVectorStore[TMetadata],
    query: str,
    top_k: int,
    filters: dict[str, str] | None = None,
    method: SearchMethod = "vector",
) -> VectorSearchResponse[TMetadata]:
    if method == "keyword":
        return vector_store.keyword_search(query, top_k=top_k, filters=filters)
    if method == "semantic":
        return vector_store.semantic_search(query, top_k=top_k, filters=filters)

    query_vector = _embed_query(embedding_model, query)
    if method == "vector":
        return vector_store.vector_search(
            VectorSearchRequest(query_vector=query_vector, top_k=top_k, filters=filters)
        )
    if method == "hybrid":
        return vector_store.hybrid_search(query, query_vector, top_k=top_k, filters=filters)
    raise ValueError(f"Unsupported search method: {method!r}.")


def _embed_query(embedding_model: EmbeddingModel, query: str) -> list[float]:
    embedding_response = embedding_model.embed(EmbeddingRequest(texts=[query]))
    if len(embedding_response.embeddings) != 1:
        raise ValueError(
            "single-query retrieval requires exactly one embedding; "
            f"received {len(embedding_response.embeddings)}"
        )
    return embedding_response.embeddings[0]
