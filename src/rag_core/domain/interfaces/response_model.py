from typing import Protocol, runtime_checkable

from rag_core.domain.models.response import ResponseRequest, ResponseResponse


@runtime_checkable
class ResponseModel(Protocol):
    def generate(self, request: ResponseRequest) -> ResponseResponse: ...
