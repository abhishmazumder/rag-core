from types import SimpleNamespace
from unittest.mock import Mock

from azure.core.credentials import TokenCredential

from rag_core.domain.interfaces.embedding_model import EmbeddingModel
from rag_core.domain.models.embedding import EmbeddingRequest, EmbeddingResponse
from rag_core.infrastructure.models.embedding.openai import OpenAIEmbeddingModel


def test_client_uses_endpoint_and_azure_identity_token_provider(mocker) -> None:
    credential = Mock(spec=TokenCredential)
    token_provider = mocker.Mock()
    create_token_provider = mocker.patch(
        "rag_core.infrastructure.models.embedding.openai.create_azure_ai_foundry_token_provider",
        return_value=token_provider,
    )
    client = mocker.patch("rag_core.infrastructure.models.embedding.openai.OpenAI")

    OpenAIEmbeddingModel(
        endpoint="https://example.services.ai.azure.com/openai/v1/",
        deployment="configured-embedding-deployment",
        credential=credential,
    )

    create_token_provider.assert_called_once_with(credential)
    client.assert_called_once_with(
        base_url="https://example.services.ai.azure.com/openai/v1/",
        api_key=token_provider,
    )


def test_embed_maps_vectors_and_preserves_input_order(mocker) -> None:
    credential = Mock(spec=TokenCredential)
    mocker.patch(
        "rag_core.infrastructure.models.embedding.openai.create_azure_ai_foundry_token_provider",
    )
    client = mocker.patch("rag_core.infrastructure.models.embedding.openai.OpenAI")
    client.return_value.embeddings.create.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(index=1, embedding=[0.2, 0.3]),
            SimpleNamespace(index=0, embedding=[0.1, 0.2]),
        ],
    )
    model = OpenAIEmbeddingModel(
        endpoint="https://example.openai.azure.com/openai/v1/",
        deployment="configured-embedding-deployment",
        credential=credential,
    )
    request = EmbeddingRequest(texts=["first text", "second text"])

    result = model.embed(request)

    assert isinstance(model, EmbeddingModel)
    assert result == EmbeddingResponse(embeddings=[[0.1, 0.2], [0.2, 0.3]])
    client.return_value.embeddings.create.assert_called_once_with(
        model="configured-embedding-deployment",
        input=["first text", "second text"],
    )
