from pydantic import BaseModel, Field


class VectorSearchRequest(BaseModel):
    query_vector: list[float] = Field(min_length=1)
    top_k: int = Field(gt=0)
    filters: dict[str, str] | None = None


class VectorSearchResult[TMetadata: BaseModel](BaseModel):
    id: str = Field(min_length=1)
    text: str
    metadata: TMetadata
    score: float


class VectorSearchResponse[TMetadata: BaseModel](BaseModel):
    results: list[VectorSearchResult[TMetadata]]
