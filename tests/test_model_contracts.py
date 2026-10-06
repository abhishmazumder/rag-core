import pytest
from pydantic import ValidationError

from rag_core.config import Settings
from rag_core.domain.interfaces.embedding_model import EmbeddingModel
from rag_core.domain.interfaces.response_model import ResponseModel
from rag_core.domain.models.embedding import EmbeddingRequest, EmbeddingResponse
from rag_core.domain.models.response import ResponseRequest, ResponseResponse


class FakeEmbeddingModel:
    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        return EmbeddingResponse(embeddings=[[0.1, 0.2] for _ in request.texts])


class FakeResponseModel:
    def generate(self, request: ResponseRequest) -> ResponseResponse:
        return ResponseResponse(content="Generated response")


def test_embedding_model_is_a_small_capability_contract() -> None:
    model = FakeEmbeddingModel()
    result = model.embed(EmbeddingRequest(texts=["sample"]))

    assert isinstance(model, EmbeddingModel)
    assert result.embeddings == [[0.1, 0.2]]


def test_embedding_request_requires_non_empty_text_batch() -> None:
    with pytest.raises(ValidationError):
        EmbeddingRequest(texts=[])


@pytest.mark.parametrize("embeddings", [[], [[]], [[0.1, "invalid"]]])
def test_embedding_response_rejects_empty_or_invalid_vectors(
    embeddings: list[list[object]],
) -> None:
    with pytest.raises(ValidationError):
        EmbeddingResponse(embeddings=embeddings)


def test_embedding_response_accepts_numeric_vectors() -> None:
    response = EmbeddingResponse(embeddings=[[0.1, 2]])

    assert response.embeddings == [[0.1, 2.0]]


def test_response_model_is_a_small_capability_contract() -> None:
    model = FakeResponseModel()
    result = model.generate(
        ResponseRequest(
            messages=[
                {"role": "user", "content": "Question"},
            ]
        )
    )

    assert isinstance(model, ResponseModel)
    assert result.content == "Generated response"


def test_response_request_requires_non_empty_supported_messages() -> None:
    with pytest.raises(ValidationError):
        ResponseRequest(messages=[])

    with pytest.raises(ValidationError):
        ResponseRequest(messages=[{"role": "tool", "content": "result"}])


def test_foundry_configuration_is_separate_for_each_capability() -> None:
    setting_names = set(Settings.model_fields)

    assert {
        "azure_ai_foundry_response_endpoint",
        "azure_ai_foundry_response_model",
        "azure_ai_foundry_embedding_endpoint",
        "azure_ai_foundry_embedding_model",
    } <= setting_names
    assert not any("api_key" in name for name in setting_names)
