from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from azure.core.credentials import TokenCredential

from rag_core.domain.interfaces.response_model import ResponseModel
from rag_core.domain.models.response import (
    ResponseMessage,
    ResponseRequest,
    ResponseResponse,
)
from rag_core.infrastructure.models.response.openai import OpenAIResponseModel


def test_client_uses_endpoint_and_azure_identity_token_provider(mocker) -> None:
    credential = Mock(spec=TokenCredential)
    token_provider = mocker.Mock()
    create_token_provider = mocker.patch(
        "rag_core.infrastructure.models.response.openai.create_azure_ai_foundry_token_provider",
        return_value=token_provider,
    )
    client = mocker.patch(
        "rag_core.infrastructure.models.response.openai.OpenAI",
    )

    OpenAIResponseModel(
        endpoint="https://example.services.ai.azure.com/openai/v1/",
        deployment="configured-deployment",
        credential=credential,
    )

    create_token_provider.assert_called_once_with(credential)
    client.assert_called_once_with(
        base_url="https://example.services.ai.azure.com/openai/v1/",
        api_key=token_provider,
    )


def test_generate_maps_messages_and_response(mocker) -> None:
    credential = Mock(spec=TokenCredential)
    token_provider = mocker.Mock()
    mocker.patch(
        "rag_core.infrastructure.models.response.openai.create_azure_ai_foundry_token_provider",
        return_value=token_provider,
    )
    client = mocker.patch(
        "rag_core.infrastructure.models.response.openai.OpenAI",
    )
    client.return_value.responses.create.return_value = SimpleNamespace(
        output_text="Generated answer",
    )
    model = OpenAIResponseModel(
        endpoint="https://example.openai.azure.com/openai/v1/",
        deployment="configured-deployment",
        credential=credential,
    )
    request = ResponseRequest(
        messages=[
            ResponseMessage(role="system", content="Be concise."),
            ResponseMessage(role="user", content="What is RAG?"),
        ],
    )

    result = model.generate(request)

    assert isinstance(model, ResponseModel)
    assert result == ResponseResponse(content="Generated answer")
    client.return_value.responses.create.assert_called_once_with(
        model="configured-deployment",
        input=[
            {"type": "message", "role": "system", "content": "Be concise."},
            {"type": "message", "role": "user", "content": "What is RAG?"},
        ],
    )


def test_generate_propagates_sdk_errors(mocker) -> None:
    credential = Mock(spec=TokenCredential)
    mocker.patch(
        "rag_core.infrastructure.models.response.openai.create_azure_ai_foundry_token_provider",
    )
    client = mocker.patch(
        "rag_core.infrastructure.models.response.openai.OpenAI",
    )
    error = RuntimeError("Provider request failed")
    client.return_value.responses.create.side_effect = error
    model = OpenAIResponseModel(
        endpoint="https://example.openai.azure.com/openai/v1/",
        deployment="configured-deployment",
        credential=credential,
    )

    with pytest.raises(RuntimeError) as raised:
        model.generate(
            ResponseRequest(
                messages=[ResponseMessage(role="user", content="Hello")],
            ),
        )
    assert raised.value is error
