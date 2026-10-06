from pydantic import BaseModel

from rag_core.domain.models.vector_search import VectorSearchResult


class RAGResponse[TMetadata: BaseModel](BaseModel):
    answer: str
    evidence: list[VectorSearchResult[TMetadata]]
