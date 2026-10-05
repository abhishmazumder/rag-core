from unittest.mock import Mock, patch

from rag_core.config import Settings
from rag_core.infrastructure import create_azure_credential


def test_settings_do_not_require_api_key() -> None:
    settings = Settings(_env_file=None)

    assert settings.azure_openai_endpoint is None
    assert settings.azure_search_endpoint is None


@patch("rag_core.infrastructure.DefaultAzureCredential")
def test_create_azure_credential_uses_default_azure_credential(
    credential_class: Mock,
) -> None:
    credential = Mock()
    credential_class.return_value = credential

    assert create_azure_credential() is credential

    credential_class.assert_called_once_with()
