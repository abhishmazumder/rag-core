from azure.core.credentials import TokenCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

from rag_core.infrastructure.vectorstores.azure_search import SEMANTIC_CONFIGURATION_NAME

HNSW_ALGORITHM_NAME = "hnsw-config"
VECTOR_SEARCH_PROFILE_NAME = "default-profile"


def build_azure_search_index(index_name: str, embedding_dimensions: int) -> SearchIndex:
    """Build the canonical index supporting vector, keyword, hybrid and semantic search."""
    if not index_name:
        raise ValueError("index_name must not be empty.")
    if embedding_dimensions <= 0:
        raise ValueError("embedding_dimensions must be greater than 0.")

    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        SearchableField(name="text", type=SearchFieldDataType.String),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=embedding_dimensions,
            vector_search_profile_name=VECTOR_SEARCH_PROFILE_NAME,
        ),
        SimpleField(
            name="chunk_index",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
        ),
        SimpleField(name="candidate_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="document_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="source_document_id", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="source_name", type=SearchFieldDataType.String),
        SearchableField(
            name="document_type",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SimpleField(name="page_number", type=SearchFieldDataType.Int32, filterable=True),
        SimpleField(name="source_url", type=SearchFieldDataType.String),
    ]
    return SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=VectorSearch(
            # The established configuration sets no HNSW parameters; Azure defaults apply.
            algorithms=[HnswAlgorithmConfiguration(name=HNSW_ALGORITHM_NAME)],
            profiles=[
                VectorSearchProfile(
                    name=VECTOR_SEARCH_PROFILE_NAME,
                    algorithm_configuration_name=HNSW_ALGORITHM_NAME,
                )
            ],
        ),
        semantic_search=SemanticSearch(
            configurations=[
                SemanticConfiguration(
                    name=SEMANTIC_CONFIGURATION_NAME,
                    # The established configuration prioritizes only the text content field.
                    prioritized_fields=SemanticPrioritizedFields(
                        content_fields=[SemanticField(field_name="text")],
                    ),
                )
            ]
        ),
    )


def create_or_update_azure_search_index(
    endpoint: str,
    index: SearchIndex,
    credential: TokenCredential,
) -> None:
    client = SearchIndexClient(endpoint=endpoint, credential=credential)
    try:
        client.create_or_update_index(index)
    finally:
        client.close()
