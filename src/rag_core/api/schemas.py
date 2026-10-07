from typing import Literal

from pydantic import BaseModel, Field

from rag_core.domain.models.document_chunk_metadata import DocumentChunkMetadata

QueryMethod = Literal["vector", "keyword", "hybrid", "semantic"]


class IngestDocumentRequest(BaseModel):
    candidate_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class IngestedChunk(BaseModel):
    id: str
    chunk_index: int
    text: str


class IngestDocumentResponse(BaseModel):
    candidate_id: str
    document_id: str
    chunks: list[IngestedChunk]


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    method: QueryMethod = "vector"
    top_k: int = Field(default=5, gt=0, le=50)


class QueryChunk(BaseModel):
    id: str
    text: str
    score: float
    metadata: DocumentChunkMetadata


class QueryResponse(BaseModel):
    query: str
    candidate_id: str
    method: QueryMethod
    answer: str
    chunks: list[QueryChunk]


class IndexSetupResponse(BaseModel):
    index_name: str


class IndexClearResponse(BaseModel):
    status: Literal["cleared"] = "cleared"


class DeleteDocumentResponse(BaseModel):
    document_id: str
    deleted_chunks: int


class DeleteChunkResponse(BaseModel):
    chunk_id: str
    deleted: bool = True
