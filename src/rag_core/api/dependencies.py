from collections.abc import Iterator
from functools import lru_cache

from azure.core.credentials import TokenCredential
from fastapi import HTTPException

from rag_core.composition.embedding_model import create_embedding_model
from rag_core.composition.response_model import create_response_model
from rag_core.composition.vector_store import create_vector_store
from rag_core.config import Settings
from rag_core.domain.interfaces.embedding_model import EmbeddingModel
from rag_core.domain.interfaces.response_model import ResponseModel
from rag_core.domain.models.document_chunk_metadata import DocumentChunkMetadata
from rag_core.infrastructure import create_azure_credential
from rag_core.infrastructure.vectorstores.azure_search import AzureAISearchVectorStore


def _not_configured(error: ValueError) -> HTTPException:
    return HTTPException(status_code=503, detail=str(error))


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_credential() -> TokenCredential:
    return create_azure_credential()


def get_embedding_model() -> EmbeddingModel:
    try:
        return create_embedding_model(get_settings())
    except ValueError as error:
        raise _not_configured(error) from error


def get_response_model() -> ResponseModel:
    try:
        return create_response_model(get_settings())
    except ValueError as error:
        raise _not_configured(error) from error


def get_vector_store() -> Iterator[AzureAISearchVectorStore[DocumentChunkMetadata]]:
    try:
        vector_store = create_vector_store(get_settings(), DocumentChunkMetadata)
    except ValueError as error:
        raise _not_configured(error) from error
    try:
        yield vector_store
    finally:
        vector_store.close()
