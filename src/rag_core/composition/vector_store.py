from pydantic import BaseModel

from rag_core.config import Settings
from rag_core.infrastructure import create_azure_credential
from rag_core.infrastructure.vectorstores.azure_search import AzureAISearchVectorStore


def create_vector_store[TMetadata: BaseModel](
    settings: Settings,
    metadata_model: type[TMetadata],
) -> AzureAISearchVectorStore[TMetadata]:
    endpoint = settings.azure_search_endpoint
    index_name = settings.azure_search_index_name
    if not endpoint or not index_name:
        raise ValueError("Azure AI Search endpoint and index name must be configured.")

    return AzureAISearchVectorStore(
        endpoint=endpoint,
        index_name=index_name,
        credential=create_azure_credential(),
        metadata_model=metadata_model,
    )
