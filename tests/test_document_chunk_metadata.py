import pytest
from pydantic import ValidationError

from rag_core.domain.models.document_chunk_metadata import DocumentChunkMetadata


def _values() -> dict[str, object]:
    return {
        "source_document_id": "employee-handbook",
        "source_name": "Employee Handbook",
        "document_type": "policy",
        "page_number": 2,
        "source_url": "https://example.com/handbook",
        "candidate_id": "candidate-1",
        "document_id": "document-1",
    }


def test_document_chunk_metadata_accepts_approved_fields() -> None:
    metadata = DocumentChunkMetadata(**_values())

    assert metadata.model_dump() == _values()
    assert set(DocumentChunkMetadata.model_fields) == set(_values())
    assert "chunk_index" not in DocumentChunkMetadata.model_fields


def test_document_chunk_metadata_optional_fields_default_to_none() -> None:
    values = _values()
    del values["page_number"], values["source_url"]

    metadata = DocumentChunkMetadata(**values)

    assert metadata.page_number is None
    assert metadata.source_url is None


@pytest.mark.parametrize(
    "missing_field",
    ["source_document_id", "source_name", "document_type", "candidate_id", "document_id"],
)
def test_document_chunk_metadata_requires_non_optional_fields(missing_field: str) -> None:
    values = _values()
    del values[missing_field]

    with pytest.raises(ValidationError):
        DocumentChunkMetadata(**values)
