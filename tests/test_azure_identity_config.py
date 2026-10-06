from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from rag_core.config import Settings
from rag_core.infrastructure import create_azure_credential
from rag_core.infrastructure.azure_identity import create_azure_ai_foundry_token_provider


def test_settings_do_not_require_api_key() -> None:
    settings = Settings(_env_file=None)

    assert settings.azure_ai_foundry_response_endpoint is None
    assert settings.azure_ai_foundry_response_model is None
    assert settings.azure_ai_foundry_embedding_endpoint is None
    assert settings.azure_ai_foundry_embedding_model is None
    assert settings.azure_ai_foundry_embedding_dimensions is None
    assert settings.azure_search_endpoint is None


def test_settings_load_embedding_dimensions_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("AZURE_AI_FOUNDRY_EMBEDDING_DIMENSIONS", "1024")

    settings = Settings(_env_file=None)

    assert settings.azure_ai_foundry_embedding_dimensions == 1024


def test_settings_accept_positive_embedding_dimensions() -> None:
    settings = Settings(_env_file=None, azure_ai_foundry_embedding_dimensions=1024)

    assert settings.azure_ai_foundry_embedding_dimensions == 1024


@pytest.mark.parametrize("dimensions", [0, -1])
def test_settings_reject_non_positive_embedding_dimensions(dimensions: int) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, azure_ai_foundry_embedding_dimensions=dimensions)


def test_settings_accept_model_specific_foundry_endpoints_and_identifiers() -> None:
    settings = Settings(
        _env_file=None,
        azure_ai_foundry_response_endpoint="https://foundry.example/anthropic",
        azure_ai_foundry_response_model="response-deployment",
        azure_ai_foundry_embedding_endpoint="https://foundry.example/openai/v1",
        azure_ai_foundry_embedding_model="embedding-deployment",
    )

    assert settings.azure_ai_foundry_response_endpoint.endswith("/anthropic")
    assert settings.azure_ai_foundry_response_model == "response-deployment"
    assert settings.azure_ai_foundry_embedding_endpoint.endswith("/openai/v1")
    assert settings.azure_ai_foundry_embedding_model == "embedding-deployment"


@patch("rag_core.infrastructure.azure_identity.DefaultAzureCredential")
def test_create_azure_credential_uses_default_azure_credential(
    credential_class: Mock,
) -> None:
    credential = Mock()
    credential_class.return_value = credential

    assert create_azure_credential() is credential

    credential_class.assert_called_once_with()


def test_foundry_token_provider_uses_shared_scope(mocker) -> None:
    credential = Mock()
    token_provider = Mock()
    get_provider = mocker.patch(
        "rag_core.infrastructure.azure_identity.get_bearer_token_provider",
        return_value=token_provider,
    )

    result = create_azure_ai_foundry_token_provider(credential)

    assert result is token_provider
    get_provider.assert_called_once_with(credential, "https://ai.azure.com/.default")
