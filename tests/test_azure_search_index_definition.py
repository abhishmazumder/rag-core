import pytest
from azure.search.documents.indexes.models import SearchFieldDataType

from rag_core.infrastructure.vectorstores.azure_search import SEMANTIC_CONFIGURATION_NAME
from rag_core.infrastructure.vectorstores.index_provisioner import build_azure_search_index

_EXPECTED_FIELDS = [
    "id",
    "text",
    "embedding",
    "chunk_index",
    "candidate_id",
    "document_id",
    "source_document_id",
    "source_name",
    "document_type",
    "page_number",
    "source_url",
]


def _index():
    return build_azure_search_index("configured-index", 1536)


def _fields():
    return {field.name: field for field in _index().fields}


def test_index_contains_exactly_the_canonical_fields() -> None:
    index = _index()

    assert index.name == "configured-index"
    assert [field.name for field in index.fields] == _EXPECTED_FIELDS
    assert all(field.hidden is not True for field in index.fields)


def test_id_is_the_filterable_string_key() -> None:
    field = _fields()["id"]

    assert field.type == SearchFieldDataType.String
    assert field.key is True
    assert field.filterable is True


def test_text_is_searchable() -> None:
    field = _fields()["text"]

    assert field.type == SearchFieldDataType.String
    assert field.searchable is True


def test_chunk_index_is_filterable_sortable_int32() -> None:
    field = _fields()["chunk_index"]

    assert field.type == SearchFieldDataType.Int32
    assert field.filterable is True
    assert field.sortable is True


@pytest.mark.parametrize(
    "name", ["candidate_id", "document_id", "source_document_id", "document_type"]
)
def test_identity_and_type_fields_are_filterable_strings(name: str) -> None:
    field = _fields()[name]

    assert field.type == SearchFieldDataType.String
    assert field.filterable is True


def test_descriptive_fields() -> None:
    fields = _fields()

    assert fields["source_name"].searchable is True
    assert fields["document_type"].searchable is True
    assert fields["page_number"].type == SearchFieldDataType.Int32
    assert fields["page_number"].filterable is True
    assert fields["source_url"].type == SearchFieldDataType.String
    assert fields["source_url"].searchable is not True
    assert fields["source_url"].filterable is not True


def test_vector_search_profile_references_hnsw_and_embedding_uses_profile() -> None:
    index = _index()
    embedding = _fields()["embedding"]
    algorithm = index.vector_search.algorithms[0]
    profile = index.vector_search.profiles[0]

    assert embedding.type == SearchFieldDataType.Collection(SearchFieldDataType.Single)
    assert embedding.vector_search_dimensions == 1536
    assert embedding.vector_search_profile_name == profile.name
    assert type(algorithm).__name__ == "HnswAlgorithmConfiguration"
    assert profile.algorithm_configuration_name == algorithm.name


def test_semantic_configuration_prioritizes_text_content() -> None:
    index = _index()
    configurations = index.semantic_search.configurations

    assert [configuration.name for configuration in configurations] == [SEMANTIC_CONFIGURATION_NAME]
    prioritized = configurations[0].prioritized_fields
    assert [field.field_name for field in prioritized.content_fields] == ["text"]
    assert prioritized.title_field is None
    assert not prioritized.keywords_fields


def test_vector_and_semantic_search_coexist() -> None:
    index = _index()

    assert index.vector_search is not None
    assert index.semantic_search is not None


def test_dimensions_are_taken_from_argument_and_validated() -> None:
    assert {f.name: f for f in build_azure_search_index("i", 3072).fields}[
        "embedding"
    ].vector_search_dimensions == 3072
    with pytest.raises(ValueError, match="embedding_dimensions"):
        build_azure_search_index("i", 0)
    with pytest.raises(ValueError, match="index_name"):
        build_azure_search_index("", 3)
