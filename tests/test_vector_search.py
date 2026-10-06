import pytest
from pydantic import BaseModel, ValidationError

from rag_core.domain.models.vector_search import (
    VectorSearchRequest,
    VectorSearchResponse,
    VectorSearchResult,
)


class SearchMetadata(BaseModel):
    category: str


def test_vector_search_request_accepts_optional_filters() -> None:
    request = VectorSearchRequest(
        query_vector=[0.1, 0.2, 0.3],
        top_k=5,
        filters={"category": "reference"},
    )

    assert request.query_vector == [0.1, 0.2, 0.3]
    assert request.top_k == 5
    assert request.filters == {"category": "reference"}
    assert VectorSearchRequest(query_vector=[0.1], top_k=1).filters is None


@pytest.mark.parametrize("top_k", [0, -1])
def test_vector_search_request_rejects_non_positive_top_k(top_k: int) -> None:
    with pytest.raises(ValidationError):
        VectorSearchRequest(query_vector=[0.1], top_k=top_k)


def test_vector_search_response_contains_typed_results() -> None:
    result = VectorSearchResult[SearchMetadata](
        id="document-1",
        text="Found content",
        metadata=SearchMetadata(category="guide"),
        score=0.91,
    )
    response = VectorSearchResponse[SearchMetadata](results=[result])

    assert response.results == [result]
    assert response.results[0].metadata.category == "guide"
    assert response.results[0].score == 0.91
