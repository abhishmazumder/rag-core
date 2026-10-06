from typing import Annotated

from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    texts: list[str] = Field(min_length=1)


class EmbeddingResponse(BaseModel):
    embeddings: list[Annotated[list[float], Field(min_length=1)]] = Field(min_length=1)
