import pytest
from pydantic import BaseModel

from rag_core.composition.embedding_model import create_embedding_model
from rag_core.composition.response_model import create_response_model
from rag_core.composition.vector_store import create_vector_store
from rag_core.config import Settings


class SearchMetadata(BaseModel):
    source: str = "default"


def test_composition_creates_configured_openai_embedding_model(mocker) -> None:
    settings = Settings(
        _env_file=None,
        azure_ai_foundry_embedding_endpoint="https://example.openai.azure.com/openai/v1/",
        azure_ai_foundry_embedding_model="configured-embedding-deployment",
    )
    credential = mocker.Mock()
    create_credential = mocker.patch(
        "rag_core.composition.embedding_model.create_azure_credential",
        return_value=credential,
    )
    embedding_model_class = mocker.patch(
        "rag_core.composition.embedding_model.OpenAIEmbeddingModel",
    )
    embedding_model = embedding_model_class.return_value

    result = create_embedding_model(settings)

    assert result is embedding_model
    create_credential.assert_called_once_with()
    embedding_model_class.assert_called_once_with(
        endpoint="https://example.openai.azure.com/openai/v1/",
        deployment="configured-embedding-deployment",
        credential=credential,
    )


@pytest.mark.parametrize(
    ("endpoint", "deployment"),
    [
        (None, "configured-embedding-deployment"),
        ("https://example.openai.azure.com/openai/v1/", None),
    ],
)
def test_embedding_composition_requires_configuration(endpoint, deployment) -> None:
    settings = Settings(
        _env_file=None,
        azure_ai_foundry_embedding_endpoint=endpoint,
        azure_ai_foundry_embedding_model=deployment,
    )

    with pytest.raises(ValueError, match="embedding endpoint and model/deployment"):
        create_embedding_model(settings)


def test_composition_creates_configured_openai_response_model(mocker) -> None:
    settings = Settings(
        _env_file=None,
        azure_ai_foundry_response_endpoint="https://example.openai.azure.com/openai/v1/",
        azure_ai_foundry_response_model="configured-deployment",
    )
    credential = mocker.Mock()
    create_credential = mocker.patch(
        "rag_core.composition.response_model.create_azure_credential",
        return_value=credential,
    )
    response_model_class = mocker.patch(
        "rag_core.composition.response_model.OpenAIResponseModel",
    )
    response_model = response_model_class.return_value

    result = create_response_model(settings)

    assert result is response_model
    create_credential.assert_called_once_with()
    response_model_class.assert_called_once_with(
        endpoint="https://example.openai.azure.com/openai/v1/",
        deployment="configured-deployment",
        credential=credential,
    )


@pytest.mark.parametrize(
    ("endpoint", "deployment"),
    [
        (None, "configured-deployment"),
        ("https://example.openai.azure.com/openai/v1/", None),
    ],
)
def test_composition_requires_response_configuration(endpoint, deployment) -> None:
    settings = Settings(
        _env_file=None,
        azure_ai_foundry_response_endpoint=endpoint,
        azure_ai_foundry_response_model=deployment,
    )

    with pytest.raises(ValueError, match="response endpoint and model/deployment"):
        create_response_model(settings)


def test_composition_creates_configured_azure_search_vector_store(mocker) -> None:
    settings = Settings(
        _env_file=None,
        azure_search_endpoint="https://search.example.net",
        azure_search_index_name="configured-index",
    )
    credential = mocker.Mock()
    metadata_model = SearchMetadata
    create_credential = mocker.patch(
        "rag_core.composition.vector_store.create_azure_credential",
        return_value=credential,
    )
    vector_store_class = mocker.patch(
        "rag_core.composition.vector_store.AzureAISearchVectorStore",
    )
    vector_store = vector_store_class.return_value

    result = create_vector_store(settings, metadata_model)

    assert result is vector_store
    create_credential.assert_called_once_with()
    vector_store_class.assert_called_once_with(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=credential,
        metadata_model=metadata_model,
    )


@pytest.mark.parametrize(
    ("endpoint", "index_name"),
    [
        (None, "configured-index"),
        ("https://search.example.net", None),
    ],
)
def test_vector_store_composition_requires_configuration(endpoint, index_name, mocker) -> None:
    settings = Settings(
        _env_file=None,
        azure_search_endpoint=endpoint,
        azure_search_index_name=index_name,
    )
    create_credential = mocker.patch(
        "rag_core.composition.vector_store.create_azure_credential",
    )
    vector_store_class = mocker.patch(
        "rag_core.composition.vector_store.AzureAISearchVectorStore",
    )

    with pytest.raises(ValueError, match="Azure AI Search endpoint and index name"):
        create_vector_store(settings, SearchMetadata)

    create_credential.assert_not_called()
    vector_store_class.assert_not_called()
