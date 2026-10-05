from azure.identity import DefaultAzureCredential


def create_azure_credential() -> DefaultAzureCredential:
    return DefaultAzureCredential()
