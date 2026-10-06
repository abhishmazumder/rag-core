from typing import Protocol, runtime_checkable

from rag_core.domain.models.embedding import EmbeddingRequest, EmbeddingResponse


@runtime_checkable
class EmbeddingModel(Protocol):
    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse: ...
