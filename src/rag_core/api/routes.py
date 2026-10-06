import json
import uuid
from typing import Annotated

from azure.core.credentials import TokenCredential
from fastapi import APIRouter, Depends, HTTPException

from rag_core.api.dependencies import (
    get_credential,
    get_embedding_model,
    get_response_model,
    get_settings,
    get_vector_store,
)
from rag_core.api.schemas import (
    DeleteChunkResponse,
    DeleteDocumentResponse,
    IndexClearResponse,
    IndexSetupResponse,
    IngestDocumentRequest,
    IngestDocumentResponse,
    QueryChunk,
    QueryRequest,
    QueryResponse,
)
from rag_core.application.chunking import recursive_chunks
from rag_core.application.rag_generation import generate_rag_response
from rag_core.config import Settings
from rag_core.domain.interfaces.embedding_model import EmbeddingModel
from rag_core.domain.interfaces.response_model import ResponseModel
from rag_core.domain.models.document_chunk_metadata import DocumentChunkMetadata
from rag_core.domain.models.embedding import EmbeddingRequest
from rag_core.domain.models.vector_document import VectorDocument
from rag_core.infrastructure.vectorstores.azure_search import AzureAISearchVectorStore
from rag_core.infrastructure.vectorstores.index_provisioner import (
    build_azure_search_index,
    create_or_update_azure_search_index,
)

router = APIRouter()

VectorStoreDep = Annotated[
    AzureAISearchVectorStore[DocumentChunkMetadata], Depends(get_vector_store)
]
EmbeddingModelDep = Annotated[EmbeddingModel, Depends(get_embedding_model)]
ResponseModelDep = Annotated[ResponseModel, Depends(get_response_model)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
CredentialDep = Annotated[TokenCredential, Depends(get_credential)]

CHUNKING_STRATEGY = "recursive"
CHUNK_SIZE_CHARS = 1000
OVERLAP_CHARS = 100
DOCUMENT_TYPE = "text"


@router.post("/documents", response_model=IngestDocumentResponse)
def ingest_document_endpoint(
    request: IngestDocumentRequest,
    embedding_model: EmbeddingModelDep,
    vector_store: VectorStoreDep,
) -> IngestDocumentResponse:
    document_id = str(uuid.uuid4())
    metadata = DocumentChunkMetadata(
        source_document_id=document_id,
        source_name=document_id,
        document_type=DOCUMENT_TYPE,
        candidate_id=request.candidate_id,
        document_id=document_id,
    )

    chunk_texts = recursive_chunks(
        request.text,
        chunk_size_chars=CHUNK_SIZE_CHARS,
        overlap_chars=OVERLAP_CHARS,
    )
    if not chunk_texts:
        return IngestDocumentResponse(
            candidate_id=request.candidate_id, document_id=document_id, chunk_ids=[]
        )

    embeddings = embedding_model.embed(EmbeddingRequest(texts=chunk_texts)).embeddings
    if len(embeddings) != len(chunk_texts):
        raise ValueError(
            f"Embedding model returned {len(embeddings)} embeddings for "
            f"{len(chunk_texts)} chunks; expected exactly one per chunk."
        )

    documents = []
    for chunk_index, (chunk_text, embedding) in enumerate(
        zip(chunk_texts, embeddings, strict=True)
    ):
        chunk_name = json.dumps(
            [
                metadata.candidate_id,
                metadata.document_id,
                metadata.page_number,
                CHUNKING_STRATEGY,
                CHUNK_SIZE_CHARS,
                OVERLAP_CHARS,
                chunk_index,
            ],
            separators=(",", ":"),
        )
        documents.append(
            VectorDocument[DocumentChunkMetadata](
                id=f"chunk-{uuid.uuid5(uuid.NAMESPACE_DNS, chunk_name)}",
                text=chunk_text,
                embedding=embedding,
                metadata=metadata,
                chunk_index=chunk_index,
            )
        )
    vector_store.upsert_many(documents)

    return IngestDocumentResponse(
        candidate_id=request.candidate_id,
        document_id=document_id,
        chunk_ids=[document.id for document in documents],
    )


@router.post("/query", response_model=QueryResponse)
def query_endpoint(
    request: QueryRequest,
    embedding_model: EmbeddingModelDep,
    response_model: ResponseModelDep,
    vector_store: VectorStoreDep,
) -> QueryResponse:
    rag_response = generate_rag_response(
        embedding_model=embedding_model,
        vector_store=vector_store,
        response_model=response_model,
        query=request.query,
        top_k=request.top_k,
        filters={"candidate_id": request.candidate_id},
        method=request.method,
    )
    return QueryResponse(
        query=request.query,
        candidate_id=request.candidate_id,
        method=request.method,
        answer=rag_response.answer,
        chunks=[
            QueryChunk(
                id=result.id,
                text=result.text,
                score=result.score,
                metadata=result.metadata,
            )
            for result in rag_response.evidence
        ],
    )


@router.post("/admin/index/setup", response_model=IndexSetupResponse)
def setup_index_endpoint(settings: SettingsDep, credential: CredentialDep) -> IndexSetupResponse:
    endpoint = settings.azure_search_endpoint
    index_name = settings.azure_search_index_name
    dimensions = settings.azure_ai_foundry_embedding_dimensions
    if not endpoint or not index_name or dimensions is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Azure AI Search endpoint, index name and embedding dimensions must be configured."
            ),
        )

    index = build_azure_search_index(index_name, dimensions)
    create_or_update_azure_search_index(endpoint, index, credential)
    return IndexSetupResponse(index_name=index_name)


@router.post("/admin/index/clear", response_model=IndexClearResponse)
def clear_index_endpoint(vector_store: VectorStoreDep) -> IndexClearResponse:
    vector_store.clear()
    return IndexClearResponse()


@router.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
def delete_document_endpoint(
    document_id: str, vector_store: VectorStoreDep
) -> DeleteDocumentResponse:
    deleted = vector_store.delete_by_document_id(document_id)
    return DeleteDocumentResponse(document_id=document_id, deleted_chunks=deleted)


@router.delete("/chunks/{chunk_id}", response_model=DeleteChunkResponse)
def delete_chunk_endpoint(chunk_id: str, vector_store: VectorStoreDep) -> DeleteChunkResponse:
    vector_store.delete(chunk_id)
    return DeleteChunkResponse(chunk_id=chunk_id)
