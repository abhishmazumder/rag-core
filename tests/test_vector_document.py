import pytest
from pydantic import BaseModel, ValidationError

from rag_core.domain.models.vector_document import VectorDocument


class EvidenceMetadata(BaseModel):
    source: str
    page: int


class KnowledgeMetadata(BaseModel):
    category: str


class EvidenceDocument(VectorDocument[EvidenceMetadata]):
    pass


class KnowledgeDocument(VectorDocument[KnowledgeMetadata]):
    pass


def test_vector_document_validates_fields_and_preserves_metadata() -> None:
    metadata = EvidenceMetadata(source="guide.pdf", page=3)

    document = VectorDocument[EvidenceMetadata](
        id="chunk-1",
        text="Relevant evidence",
        embedding=[0.1, 1, -0.3],
        metadata=metadata,
        chunk_index=0,
    )

    assert document.id == "chunk-1"
    assert document.text == "Relevant evidence"
    assert document.embedding == [0.1, 1.0, -0.3]
    assert document.metadata is metadata


def test_concrete_document_types_support_typed_metadata() -> None:
    evidence = EvidenceDocument(
        id="evidence-1",
        text="Evidence content",
        embedding=[0.1],
        metadata=EvidenceMetadata(source="guide.pdf", page=3),
        chunk_index=0,
    )
    knowledge = KnowledgeDocument(
        id="knowledge-1",
        text="Knowledge content",
        embedding=[0.2],
        metadata=KnowledgeMetadata(category="reference"),
        chunk_index=1,
    )

    assert evidence.metadata.page == 3
    assert knowledge.metadata.category == "reference"


def test_vector_document_supports_different_metadata_models() -> None:
    document = VectorDocument[KnowledgeMetadata](
        id="knowledge-1",
        text="Reusable knowledge",
        embedding=[0.2, 0.4],
        metadata=KnowledgeMetadata(category="reference"),
        chunk_index=0,
    )

    assert document.metadata.category == "reference"


@pytest.mark.parametrize(
    "missing_field",
    ["id", "text", "embedding", "metadata", "chunk_index"],
)
def test_vector_document_requires_all_fields(missing_field: str) -> None:
    values: dict[str, object] = {
        "id": "chunk-1",
        "text": "Relevant evidence",
        "embedding": [0.1, 0.2],
        "metadata": EvidenceMetadata(source="guide.pdf", page=3),
        "chunk_index": 0,
    }
    del values[missing_field]

    with pytest.raises(ValidationError):
        VectorDocument[EvidenceMetadata](**values)


def test_vector_document_rejects_non_numeric_embedding_values() -> None:
    with pytest.raises(ValidationError):
        VectorDocument[EvidenceMetadata](
            id="chunk-1",
            text="Relevant evidence",
            embedding=[0.1, "not-a-number"],
            metadata=EvidenceMetadata(source="guide.pdf", page=3),
        )


def test_vector_document_validates_generic_metadata_type() -> None:
    with pytest.raises(ValidationError):
        VectorDocument[EvidenceMetadata](
            id="chunk-1",
            text="Relevant evidence",
            embedding=[0.1, 0.2],
            metadata={"source": "guide.pdf", "page": "not-an-integer"},
        )


def test_vector_document_stores_chunk_index() -> None:
    document = VectorDocument[EvidenceMetadata](
        id="chunk-1",
        text="Relevant evidence",
        embedding=[0.1],
        metadata=EvidenceMetadata(source="guide.pdf", page=3),
        chunk_index=4,
    )

    assert document.chunk_index == 4


def test_vector_document_rejects_negative_chunk_index() -> None:
    values = {
        "id": "chunk-1",
        "text": "Relevant evidence",
        "embedding": [0.1],
        "metadata": EvidenceMetadata(source="guide.pdf", page=3),
    }

    with pytest.raises(ValidationError):
        VectorDocument[EvidenceMetadata](**values, chunk_index=-1)
