from collections.abc import Callable

from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

_AZURE_AI_FOUNDRY_SCOPE = "https://ai.azure.com/.default"


def create_azure_credential() -> DefaultAzureCredential:
    return DefaultAzureCredential()


def create_azure_ai_foundry_token_provider(
    credential: TokenCredential,
) -> Callable[[], str]:
    return get_bearer_token_provider(credential, _AZURE_AI_FOUNDRY_SCOPE)
