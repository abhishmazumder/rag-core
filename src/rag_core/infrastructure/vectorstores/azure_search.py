import re
from collections.abc import Mapping, Sequence

from azure.core.credentials import TokenCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from pydantic import BaseModel

from rag_core.domain.models.vector_document import VectorDocument
from rag_core.domain.models.vector_search import (
    VectorSearchRequest,
    VectorSearchResponse,
    VectorSearchResult,
)

_ID_FIELD = "id"
_TEXT_FIELD = "text"
_EMBEDDING_FIELD = "embedding"
_CHUNK_INDEX_FIELD = "chunk_index"
_DOCUMENT_FIELDS = frozenset({_ID_FIELD, _TEXT_FIELD, _EMBEDDING_FIELD, _CHUNK_INDEX_FIELD})
_FILTER_FIELD_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_DELETE_BATCH_SIZE = 1000
SEMANTIC_CONFIGURATION_NAME = "rag-core-semantic-config"


class AzureAISearchVectorStore[TMetadata: BaseModel]:
    def __init__(
        self,
        endpoint: str,
        index_name: str,
        credential: TokenCredential,
        metadata_model: type[TMetadata],
    ) -> None:
        self._client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=credential,
        )
        self._metadata_model = metadata_model

    def upsert(self, document: VectorDocument[TMetadata]) -> None:
        self.upsert_many([document])

    def upsert_many(self, documents: Sequence[VectorDocument[TMetadata]]) -> None:
        azure_documents = [self._to_azure_document(document) for document in documents]
        if not azure_documents:
            return
        results = self._client.merge_or_upload_documents(documents=azure_documents)
        failures = [result for result in results if not result.succeeded]
        if failures:
            failure_details = "; ".join(
                f"{result.key}: {result.error_message}" for result in failures
            )
            raise RuntimeError(f"Azure AI Search document upsert failed: {failure_details}")

    @staticmethod
    def _to_azure_document(document: VectorDocument[TMetadata]) -> dict[str, object]:
        metadata = document.metadata.model_dump(mode="json")
        conflicting_fields = _DOCUMENT_FIELDS.intersection(metadata)
        if conflicting_fields:
            fields = ", ".join(sorted(conflicting_fields))
            raise ValueError(f"metadata fields conflict with VectorDocument fields: {fields}")

        return {
            _ID_FIELD: document.id,
            _TEXT_FIELD: document.text,
            _EMBEDDING_FIELD: document.embedding,
            _CHUNK_INDEX_FIELD: document.chunk_index,
            **metadata,
        }

    def vector_search(
        self,
        request: VectorSearchRequest,
    ) -> VectorSearchResponse[TMetadata]:
        vector_query = VectorizedQuery(
            vector=request.query_vector,
            k_nearest_neighbors=request.top_k,
            fields=_EMBEDDING_FIELD,
        )
        search_results = self._client.search(
            vector_queries=[vector_query],
            top=request.top_k,
            select=[_ID_FIELD, _TEXT_FIELD, *self._metadata_model.model_fields],
            **self._search_options(request),
        )

        results = [
            VectorSearchResult[TMetadata](
                id=result[_ID_FIELD],
                text=result[_TEXT_FIELD],
                metadata=self._metadata_from_document(result),
                score=result["@search.score"],
            )
            for result in search_results
        ]
        return VectorSearchResponse[TMetadata](results=results)

    def keyword_search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> VectorSearchResponse[TMetadata]:
        search_results = self._client.search(
            search_text=query,
            top=top_k,
            select=[_ID_FIELD, _TEXT_FIELD, *self._metadata_model.model_fields],
            **self._filter_options(filters),
        )
        results = [
            VectorSearchResult[TMetadata](
                id=result[_ID_FIELD],
                text=result[_TEXT_FIELD],
                metadata=self._metadata_from_document(result),
                score=result["@search.score"],
            )
            for result in search_results
        ]
        return VectorSearchResponse[TMetadata](results=results)

    def hybrid_search(
        self,
        query: str,
        query_vector: list[float],
        *,
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> VectorSearchResponse[TMetadata]:
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields=_EMBEDDING_FIELD,
        )
        search_results = self._client.search(
            search_text=query,
            vector_queries=[vector_query],
            top=top_k,
            select=[_ID_FIELD, _TEXT_FIELD, *self._metadata_model.model_fields],
            **self._filter_options(filters),
        )
        results = [
            VectorSearchResult[TMetadata](
                id=result[_ID_FIELD],
                text=result[_TEXT_FIELD],
                metadata=self._metadata_from_document(result),
                score=result["@search.score"],
            )
            for result in search_results
        ]
        return VectorSearchResponse[TMetadata](results=results)

    def semantic_search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> VectorSearchResponse[TMetadata]:
        search_results = self._client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name=SEMANTIC_CONFIGURATION_NAME,
            top=top_k,
            select=[_ID_FIELD, _TEXT_FIELD, *self._metadata_model.model_fields],
            **self._filter_options(filters),
        )
        results = [
            VectorSearchResult[TMetadata](
                id=result[_ID_FIELD],
                text=result[_TEXT_FIELD],
                metadata=self._metadata_from_document(result),
                score=result["@search.reranker_score"],
            )
            for result in search_results
        ]
        return VectorSearchResponse[TMetadata](results=results)

    def get_by_id(self, document_id: str) -> VectorDocument[TMetadata] | None:
        selected_fields = [
            _ID_FIELD,
            _TEXT_FIELD,
            _EMBEDDING_FIELD,
            _CHUNK_INDEX_FIELD,
            *self._metadata_model.model_fields,
        ]
        try:
            result = self._client.get_document(
                key=document_id,
                selected_fields=selected_fields,
            )
        except ResourceNotFoundError:
            return None

        return VectorDocument[TMetadata](
            id=result[_ID_FIELD],
            text=result[_TEXT_FIELD],
            embedding=result[_EMBEDDING_FIELD],
            metadata=self._metadata_from_document(result),
            chunk_index=result[_CHUNK_INDEX_FIELD],
        )

    def delete(self, document_id: str) -> None:
        results = self._client.delete_documents(documents=[{_ID_FIELD: document_id}])
        failures = [result for result in results if not result.succeeded]
        if failures:
            failure_details = "; ".join(
                f"{result.key}: {result.error_message}" for result in failures
            )
            raise RuntimeError(f"Azure AI Search document delete failed: {failure_details}")

    def delete_by_document_id(self, document_id: str) -> int:
        chunks = self._client.search(
            search_text="*",
            select=[_ID_FIELD],
            **self._filter_options({"document_id": document_id}),
        )
        chunk_ids = [chunk[_ID_FIELD] for chunk in chunks]
        for offset in range(0, len(chunk_ids), _DELETE_BATCH_SIZE):
            batch = chunk_ids[offset : offset + _DELETE_BATCH_SIZE]
            results = self._client.delete_documents(
                documents=[{_ID_FIELD: chunk_id} for chunk_id in batch]
            )
            failures = [result for result in results if not result.succeeded]
            if failures:
                failure_details = "; ".join(
                    f"{result.key}: {result.error_message}" for result in failures
                )
                raise RuntimeError(f"Azure AI Search document delete failed: {failure_details}")
        return len(chunk_ids)

    def clear(self) -> None:
        documents = self._client.search(search_text="*", select=[_ID_FIELD])
        document_ids = [document[_ID_FIELD] for document in documents]
        for offset in range(0, len(document_ids), _DELETE_BATCH_SIZE):
            batch = document_ids[offset : offset + _DELETE_BATCH_SIZE]
            results = self._client.delete_documents(
                documents=[{_ID_FIELD: document_id} for document_id in batch]
            )
            failures = [result for result in results if not result.succeeded]
            if failures:
                failure_details = "; ".join(
                    f"{result.key}: {result.error_message}" for result in failures
                )
                raise RuntimeError(f"Azure AI Search document clear failed: {failure_details}")

    @staticmethod
    def _search_options(request: VectorSearchRequest) -> dict[str, str]:
        return AzureAISearchVectorStore._filter_options(request.filters)

    @staticmethod
    def _filter_options(filters: dict[str, str] | None) -> dict[str, str]:
        if not filters:
            return {}

        clauses = []
        for field_name, value in filters.items():
            if not _FILTER_FIELD_NAME.fullmatch(field_name):
                raise ValueError(f"invalid Azure AI Search filter field name: {field_name!r}")
            escaped_value = value.replace("'", "''")
            clauses.append(f"{field_name} eq '{escaped_value}'")
        return {"filter": " and ".join(clauses)}

    def close(self) -> None:
        self._client.close()

    def _metadata_from_document(self, document: Mapping[str, object]) -> TMetadata:
        metadata = {
            field_name: document[field_name]
            for field_name in self._metadata_model.model_fields
            if field_name in document
        }
        return self._metadata_model.model_validate(metadata)
