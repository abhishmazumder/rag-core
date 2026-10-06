from unittest.mock import Mock

import pytest
from azure.core.credentials import TokenCredential
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from pydantic import BaseModel

from rag_core.domain.models.vector_document import VectorDocument
from rag_core.domain.models.vector_search import (
    VectorSearchRequest,
    VectorSearchResponse,
    VectorSearchResult,
)
from rag_core.infrastructure.vectorstores import azure_search
from rag_core.infrastructure.vectorstores.azure_search import AzureAISearchVectorStore


class EvidenceMetadata(BaseModel):
    source: str
    page: int


class ConflictingMetadata(BaseModel):
    text: str


class OptionalEvidenceMetadata(BaseModel):
    source: str
    page: int | None = None


def test_constructor_creates_search_client_for_configured_index(mocker) -> None:
    credential = Mock(spec=TokenCredential)
    client_class = mocker.patch.object(azure_search, "SearchClient")

    AzureAISearchVectorStore(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=credential,
        metadata_model=EvidenceMetadata,
    )

    client_class.assert_called_once_with(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=credential,
    )


def test_upsert_maps_document_and_flattened_metadata(mocker) -> None:
    client_class = mocker.patch.object(azure_search, "SearchClient")
    client_class.return_value.merge_or_upload_documents.return_value = [Mock(succeeded=True)]
    store = AzureAISearchVectorStore(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=Mock(spec=TokenCredential),
        metadata_model=EvidenceMetadata,
    )
    document = VectorDocument[EvidenceMetadata](
        id="document-1",
        text="Relevant text",
        embedding=[0.1, 0.2],
        metadata=EvidenceMetadata(source="guide.pdf", page=3),
        chunk_index=2,
    )

    result = store.upsert(document)

    assert result is None
    client_class.return_value.merge_or_upload_documents.assert_called_once_with(
        documents=[
            {
                "id": "document-1",
                "text": "Relevant text",
                "embedding": [0.1, 0.2],
                "chunk_index": 2,
                "source": "guide.pdf",
                "page": 3,
            }
        ],
    )


def test_upsert_rejects_metadata_that_would_overwrite_document_fields(mocker) -> None:
    client_class = mocker.patch.object(azure_search, "SearchClient")
    store = AzureAISearchVectorStore(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=Mock(spec=TokenCredential),
        metadata_model=ConflictingMetadata,
    )
    document = VectorDocument[ConflictingMetadata](
        id="document-1",
        text="Relevant text",
        embedding=[0.1, 0.2],
        metadata=ConflictingMetadata(text="conflicting metadata"),
        chunk_index=0,
    )

    with pytest.raises(ValueError, match="metadata fields conflict"):
        store.upsert(document)

    client_class.return_value.merge_or_upload_documents.assert_not_called()


def test_upsert_surfaces_document_level_indexing_failures(mocker) -> None:
    client_class = mocker.patch.object(azure_search, "SearchClient")
    client_class.return_value.merge_or_upload_documents.return_value = [
        Mock(succeeded=False, key="document-1", error_message="invalid field")
    ]
    store = AzureAISearchVectorStore(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=Mock(spec=TokenCredential),
        metadata_model=EvidenceMetadata,
    )
    document = VectorDocument[EvidenceMetadata](
        id="document-1",
        text="Relevant text",
        embedding=[0.1, 0.2],
        metadata=EvidenceMetadata(source="guide.pdf", page=3),
        chunk_index=0,
    )

    with pytest.raises(RuntimeError, match="document-1: invalid field"):
        store.upsert(document)


def test_close_closes_the_owned_search_client(mocker) -> None:
    client_class = mocker.patch.object(azure_search, "SearchClient")
    store = AzureAISearchVectorStore(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=Mock(spec=TokenCredential),
        metadata_model=EvidenceMetadata,
    )

    store.close()

    client_class.return_value.close.assert_called_once_with()


def test_search_constructs_vector_query_and_maps_result_fields(mocker) -> None:
    client_class = mocker.patch.object(azure_search, "SearchClient")
    client = client_class.return_value
    client.search.return_value = [
        {
            "id": "document-1",
            "text": "Matched text",
            "embedding": [0.1, 0.2],
            "source": "guide.pdf",
            "page": 3,
            "@search.score": 0.876,
        }
    ]
    store = AzureAISearchVectorStore(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=Mock(spec=TokenCredential),
        metadata_model=EvidenceMetadata,
    )
    request = VectorSearchRequest(query_vector=[0.4, 0.5], top_k=4)

    response = store.vector_search(request)

    assert isinstance(response, VectorSearchResponse)
    assert response.results == [
        VectorSearchResult[EvidenceMetadata](
            id="document-1",
            text="Matched text",
            metadata=EvidenceMetadata(source="guide.pdf", page=3),
            score=0.876,
        )
    ]
    client.search.assert_called_once()
    kwargs = client.search.call_args.kwargs
    assert kwargs["top"] == 4
    assert kwargs["select"] == ["id", "text", "source", "page"]
    assert len(kwargs["vector_queries"]) == 1
    query = kwargs["vector_queries"][0]
    assert query.vector == [0.4, 0.5]
    assert query.k_nearest_neighbors == 4
    assert query.fields == "embedding"
    assert "search_text" not in client.search.call_args.kwargs


def test_search_maps_exact_string_filters_with_odata_escaping(mocker) -> None:
    client_class = mocker.patch.object(azure_search, "SearchClient")
    client_class.return_value.search.return_value = []
    store = AzureAISearchVectorStore(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=Mock(spec=TokenCredential),
        metadata_model=EvidenceMetadata,
    )

    response = store.vector_search(
        VectorSearchRequest(
            query_vector=[0.4, 0.5],
            top_k=3,
            filters={"source": "O'Brien.pdf", "category": "manual"},
        )
    )

    assert response == VectorSearchResponse[EvidenceMetadata](results=[])
    kwargs = client_class.return_value.search.call_args.kwargs
    assert kwargs["filter"] == "source eq 'O''Brien.pdf' and category eq 'manual'"


def test_search_omits_filter_when_none_and_returns_empty_results(mocker) -> None:
    client_class = mocker.patch.object(azure_search, "SearchClient")
    client_class.return_value.search.return_value = iter([])
    store = AzureAISearchVectorStore(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=Mock(spec=TokenCredential),
        metadata_model=EvidenceMetadata,
    )

    response = store.vector_search(VectorSearchRequest(query_vector=[0.4, 0.5], top_k=2))

    assert response == VectorSearchResponse[EvidenceMetadata](results=[])
    assert "filter" not in client_class.return_value.search.call_args.kwargs


def test_keyword_search_passes_text_query_and_top_k_and_maps_results(mocker) -> None:
    client_class = mocker.patch.object(azure_search, "SearchClient")
    client = client_class.return_value
    client.search.return_value = [
        {
            "id": "document-1",
            "text": "Annual leave policy",
            "source": "handbook.pdf",
            "page": 3,
            "@search.score": 2.75,
        }
    ]
    store = AzureAISearchVectorStore(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=Mock(spec=TokenCredential),
        metadata_model=EvidenceMetadata,
    )

    response = store.keyword_search("annual leave", top_k=3)

    assert response == VectorSearchResponse[EvidenceMetadata](
        results=[
            VectorSearchResult[EvidenceMetadata](
                id="document-1",
                text="Annual leave policy",
                metadata=EvidenceMetadata(source="handbook.pdf", page=3),
                score=2.75,
            )
        ]
    )
    client.search.assert_called_once_with(
        search_text="annual leave",
        top=3,
        select=["id", "text", "source", "page"],
    )


def test_keyword_search_applies_existing_filters(mocker) -> None:
    client_class = mocker.patch.object(azure_search, "SearchClient")
    client_class.return_value.search.return_value = []
    store = AzureAISearchVectorStore(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=Mock(spec=TokenCredential),
        metadata_model=EvidenceMetadata,
    )

    response = store.keyword_search(
        "annual leave",
        top_k=2,
        filters={"source": "O'Brien.pdf"},
    )

    assert response == VectorSearchResponse[EvidenceMetadata](results=[])
    assert client_class.return_value.search.call_args.kwargs["filter"] == (
        "source eq 'O''Brien.pdf'"
    )


def test_keyword_search_propagates_azure_sdk_exceptions(mocker) -> None:
    client_class = mocker.patch.object(azure_search, "SearchClient")
    error = HttpResponseError(message="Search service unavailable")
    client_class.return_value.search.side_effect = error
    store = AzureAISearchVectorStore(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=Mock(spec=TokenCredential),
        metadata_model=EvidenceMetadata,
    )

    with pytest.raises(HttpResponseError) as raised:
        store.keyword_search("annual leave", top_k=3)

    assert raised.value is error


def test_hybrid_search_passes_text_and_vector_queries_and_maps_results(mocker) -> None:
    client_class = mocker.patch.object(azure_search, "SearchClient")
    client = client_class.return_value
    client.search.return_value = [
        {
            "id": "document-1",
            "text": "Annual leave policy",
            "source": "handbook.pdf",
            "page": 3,
            "@search.score": 0.875,
        }
    ]
    store = AzureAISearchVectorStore(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=Mock(spec=TokenCredential),
        metadata_model=EvidenceMetadata,
    )

    response = store.hybrid_search(
        "annual leave",
        [0.1, 0.2],
        top_k=3,
        filters={"source": "handbook.pdf"},
    )

    assert response == VectorSearchResponse[EvidenceMetadata](
        results=[
            VectorSearchResult[EvidenceMetadata](
                id="document-1",
                text="Annual leave policy",
                metadata=EvidenceMetadata(source="handbook.pdf", page=3),
                score=0.875,
            )
        ]
    )
    client.search.assert_called_once()
    kwargs = client.search.call_args.kwargs
    assert kwargs["search_text"] == "annual leave"
    assert kwargs["top"] == 3
    assert kwargs["select"] == ["id", "text", "source", "page"]
    assert kwargs["filter"] == "source eq 'handbook.pdf'"
    assert len(kwargs["vector_queries"]) == 1
    vector_query = kwargs["vector_queries"][0]
    assert vector_query.vector == [0.1, 0.2]
    assert vector_query.k_nearest_neighbors == 3
    assert vector_query.fields == "embedding"


def test_hybrid_search_propagates_azure_sdk_exceptions(mocker) -> None:
    client_class = mocker.patch.object(azure_search, "SearchClient")
    error = HttpResponseError(message="Search service unavailable")
    client_class.return_value.search.side_effect = error
    store = AzureAISearchVectorStore(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=Mock(spec=TokenCredential),
        metadata_model=EvidenceMetadata,
    )

    with pytest.raises(HttpResponseError) as raised:
        store.hybrid_search("annual leave", [0.1, 0.2], top_k=3)

    assert raised.value is error


def test_semantic_search_passes_semantic_query_and_maps_reranker_results(mocker) -> None:
    client_class = mocker.patch.object(azure_search, "SearchClient")
    client = client_class.return_value
    client.search.return_value = [
        {
            "id": "document-1",
            "text": "Annual leave policy",
            "source": "handbook.pdf",
            "page": 3,
            "@search.score": 1.25,
            "@search.reranker_score": 3.75,
        }
    ]
    store = AzureAISearchVectorStore(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=Mock(spec=TokenCredential),
        metadata_model=EvidenceMetadata,
    )

    response = store.semantic_search(
        "annual leave",
        top_k=3,
        filters={"source": "handbook.pdf"},
    )

    assert response == VectorSearchResponse[EvidenceMetadata](
        results=[
            VectorSearchResult[EvidenceMetadata](
                id="document-1",
                text="Annual leave policy",
                metadata=EvidenceMetadata(source="handbook.pdf", page=3),
                score=3.75,
            )
        ]
    )
    client.search.assert_called_once_with(
        search_text="annual leave",
        query_type="semantic",
        semantic_configuration_name=azure_search.SEMANTIC_CONFIGURATION_NAME,
        top=3,
        select=["id", "text", "source", "page"],
        filter="source eq 'handbook.pdf'",
    )


def test_semantic_search_propagates_azure_sdk_exceptions(mocker) -> None:
    client_class = mocker.patch.object(azure_search, "SearchClient")
    error = HttpResponseError(message="Search service unavailable")
    client_class.return_value.search.side_effect = error
    store = AzureAISearchVectorStore(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=Mock(spec=TokenCredential),
        metadata_model=EvidenceMetadata,
    )

    with pytest.raises(HttpResponseError) as raised:
        store.semantic_search("annual leave", top_k=3)

    assert raised.value is error


@pytest.mark.parametrize("operation", ["search", "get_by_id"])
def test_read_operations_apply_defaults_for_omitted_metadata_fields(mocker, operation: str) -> None:
    client_class = mocker.patch.object(azure_search, "SearchClient")
    document = {
        "id": "document-1",
        "text": "Stored text",
        "embedding": [0.1, 0.2],
        "chunk_index": 0,
        "source": "guide.pdf",
        "@search.score": 0.9,
    }
    client_class.return_value.search.return_value = [document]
    client_class.return_value.get_document.return_value = document
    store = AzureAISearchVectorStore(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=Mock(spec=TokenCredential),
        metadata_model=OptionalEvidenceMetadata,
    )

    if operation == "search":
        result = store.vector_search(VectorSearchRequest(query_vector=[0.4, 0.5], top_k=2)).results[
            0
        ]
        metadata = result.metadata
    else:
        result = store.get_by_id("document-1")
        assert result is not None
        metadata = result.metadata

    assert metadata == OptionalEvidenceMetadata(source="guide.pdf", page=None)


def test_search_rejects_invalid_filter_field_names(mocker) -> None:
    client_class = mocker.patch.object(azure_search, "SearchClient")
    store = AzureAISearchVectorStore(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=Mock(spec=TokenCredential),
        metadata_model=EvidenceMetadata,
    )

    with pytest.raises(ValueError, match="invalid Azure AI Search filter field"):
        store.vector_search(
            VectorSearchRequest(
                query_vector=[0.4, 0.5],
                top_k=2,
                filters={"source eq 'x'": "value"},
            )
        )

    client_class.return_value.search.assert_not_called()


def test_get_by_id_maps_document_and_typed_metadata(mocker) -> None:
    client_class = mocker.patch.object(azure_search, "SearchClient")
    client_class.return_value.get_document.return_value = {
        "id": "document-1",
        "text": "Stored text",
        "embedding": [0.1, 0.2, 0.3],
        "chunk_index": 4,
        "source": "guide.pdf",
        "page": 7,
    }
    store = AzureAISearchVectorStore(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=Mock(spec=TokenCredential),
        metadata_model=EvidenceMetadata,
    )

    document = store.get_by_id("document-1")

    assert document == VectorDocument[EvidenceMetadata](
        id="document-1",
        text="Stored text",
        embedding=[0.1, 0.2, 0.3],
        metadata=EvidenceMetadata(source="guide.pdf", page=7),
        chunk_index=4,
    )
    client_class.return_value.get_document.assert_called_once_with(
        key="document-1",
        selected_fields=["id", "text", "embedding", "chunk_index", "source", "page"],
    )


def test_get_by_id_returns_none_for_missing_document(mocker) -> None:
    client_class = mocker.patch.object(azure_search, "SearchClient")
    client_class.return_value.get_document.side_effect = ResourceNotFoundError(
        message="Document not found",
    )
    store = AzureAISearchVectorStore(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=Mock(spec=TokenCredential),
        metadata_model=EvidenceMetadata,
    )

    assert store.get_by_id("missing-id") is None


def test_get_by_id_preserves_non_not_found_azure_errors(mocker) -> None:
    client_class = mocker.patch.object(azure_search, "SearchClient")
    error = HttpResponseError(message="Search service unavailable")
    client_class.return_value.get_document.side_effect = error
    store = AzureAISearchVectorStore(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=Mock(spec=TokenCredential),
        metadata_model=EvidenceMetadata,
    )

    with pytest.raises(HttpResponseError) as raised:
        store.get_by_id("document-1")

    assert raised.value is error


def test_delete_calls_azure_search_with_document_key_and_returns_none(mocker) -> None:
    client_class = mocker.patch.object(azure_search, "SearchClient")
    client_class.return_value.delete_documents.return_value = [
        Mock(succeeded=True, key="document-1", error_message=None)
    ]
    store = AzureAISearchVectorStore(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=Mock(spec=TokenCredential),
        metadata_model=EvidenceMetadata,
    )

    result = store.delete("document-1")

    assert result is None
    client_class.return_value.delete_documents.assert_called_once_with(
        documents=[{"id": "document-1"}]
    )


def test_delete_surfaces_unsuccessful_document_result(mocker) -> None:
    client_class = mocker.patch.object(azure_search, "SearchClient")
    client_class.return_value.delete_documents.return_value = [
        Mock(succeeded=False, key="document-1", error_message="delete rejected")
    ]
    store = AzureAISearchVectorStore(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=Mock(spec=TokenCredential),
        metadata_model=EvidenceMetadata,
    )

    with pytest.raises(RuntimeError, match="document-1: delete rejected"):
        store.delete("document-1")


def test_delete_of_missing_document_is_idempotently_successful(mocker) -> None:
    client_class = mocker.patch.object(azure_search, "SearchClient")
    client_class.return_value.delete_documents.return_value = [
        Mock(succeeded=True, key="missing-id", error_message=None)
    ]
    store = AzureAISearchVectorStore(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=Mock(spec=TokenCredential),
        metadata_model=EvidenceMetadata,
    )

    assert store.delete("missing-id") is None


def test_delete_preserves_azure_sdk_exceptions(mocker) -> None:
    client_class = mocker.patch.object(azure_search, "SearchClient")
    error = HttpResponseError(message="Search service unavailable")
    client_class.return_value.delete_documents.side_effect = error
    store = AzureAISearchVectorStore(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=Mock(spec=TokenCredential),
        metadata_model=EvidenceMetadata,
    )

    with pytest.raises(HttpResponseError) as raised:
        store.delete("document-1")

    assert raised.value is error


def test_clear_deletes_all_documents_in_index(mocker) -> None:
    client_class = mocker.patch.object(azure_search, "SearchClient")
    client = client_class.return_value
    client.search.return_value = [{"id": "document-1"}, {"id": "document-2"}]
    client.delete_documents.return_value = [
        Mock(succeeded=True, key="document-1", error_message=None),
        Mock(succeeded=True, key="document-2", error_message=None),
    ]
    store = AzureAISearchVectorStore(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=Mock(spec=TokenCredential),
        metadata_model=EvidenceMetadata,
    )

    result = store.clear()

    assert result is None
    client.search.assert_called_once_with(search_text="*", select=["id"])
    client.delete_documents.assert_called_once_with(
        documents=[{"id": "document-1"}, {"id": "document-2"}]
    )


def test_clear_on_empty_index_does_not_issue_deletes(mocker) -> None:
    client_class = mocker.patch.object(azure_search, "SearchClient")
    client = client_class.return_value
    client.search.return_value = []
    store = AzureAISearchVectorStore(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=Mock(spec=TokenCredential),
        metadata_model=EvidenceMetadata,
    )

    assert store.clear() is None

    client.delete_documents.assert_not_called()


def test_clear_surfaces_unsuccessful_document_deletion(mocker) -> None:
    client_class = mocker.patch.object(azure_search, "SearchClient")
    client = client_class.return_value
    client.search.return_value = [{"id": "document-1"}]
    client.delete_documents.return_value = [
        Mock(succeeded=False, key="document-1", error_message="delete rejected")
    ]
    store = AzureAISearchVectorStore(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=Mock(spec=TokenCredential),
        metadata_model=EvidenceMetadata,
    )

    with pytest.raises(RuntimeError, match="document-1: delete rejected"):
        store.clear()


def test_clear_preserves_azure_sdk_exceptions(mocker) -> None:
    client_class = mocker.patch.object(azure_search, "SearchClient")
    client = client_class.return_value
    client.search.return_value = [{"id": "document-1"}]
    error = HttpResponseError(message="Search service unavailable")
    client.delete_documents.side_effect = error
    store = AzureAISearchVectorStore(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=Mock(spec=TokenCredential),
        metadata_model=EvidenceMetadata,
    )

    with pytest.raises(HttpResponseError) as raised:
        store.clear()

    assert raised.value is error


def test_delete_by_document_id_deletes_matching_chunks(mocker) -> None:
    client_class = mocker.patch.object(azure_search, "SearchClient")
    client = client_class.return_value
    client.search.return_value = [{"id": "chunk-1"}, {"id": "chunk-2"}]
    client.delete_documents.return_value = [
        Mock(succeeded=True, key="chunk-1", error_message=None),
        Mock(succeeded=True, key="chunk-2", error_message=None),
    ]
    store = AzureAISearchVectorStore(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=Mock(spec=TokenCredential),
        metadata_model=EvidenceMetadata,
    )

    assert store.delete_by_document_id("doc'1") == 2

    client.search.assert_called_once_with(
        search_text="*", select=["id"], filter="document_id eq 'doc''1'"
    )
    client.delete_documents.assert_called_once_with(
        documents=[{"id": "chunk-1"}, {"id": "chunk-2"}]
    )


def test_delete_by_document_id_without_matches_issues_no_deletes(mocker) -> None:
    client_class = mocker.patch.object(azure_search, "SearchClient")
    client = client_class.return_value
    client.search.return_value = []
    store = AzureAISearchVectorStore(
        endpoint="https://search.example.net",
        index_name="configured-index",
        credential=Mock(spec=TokenCredential),
        metadata_model=EvidenceMetadata,
    )

    assert store.delete_by_document_id("doc-1") == 0
    client.delete_documents.assert_not_called()
