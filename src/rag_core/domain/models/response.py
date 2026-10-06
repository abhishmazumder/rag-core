from typing import Literal

from pydantic import BaseModel, Field


class ResponseMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ResponseRequest(BaseModel):
    messages: list[ResponseMessage] = Field(min_length=1)


class ResponseResponse(BaseModel):
    content: str
