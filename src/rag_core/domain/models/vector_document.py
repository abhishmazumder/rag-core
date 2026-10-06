from pydantic import BaseModel, Field


class VectorDocument[TMetadata: BaseModel](BaseModel):
    id: str = Field(min_length=1)
    text: str
    embedding: list[float]
    metadata: TMetadata
    chunk_index: int = Field(ge=0)
