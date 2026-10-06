# Contracts

The stable data models and capability contracts of rag-core. For how they fit
together see [ARCHITECTURE.md](ARCHITECTURE.md). Contracts must not expose Azure
SDK or model-provider SDK types.

## Part 1: Generic rag-core contracts

### Embedding

```text
EmbeddingRequest   texts: list[str]                  (non-empty batch)
EmbeddingResponse  embeddings: list[list[float]]     (one vector per input)
EmbeddingModel     embed(EmbeddingRequest) -> EmbeddingResponse
```

Embedding dimensions come from the configured deployment and must match the vector
field dimensions of the index.

### Response generation

```text
ResponseMessage    role (system | user | assistant), content
ResponseRequest    messages: list[ResponseMessage]
ResponseResponse   content: str
ResponseModel      generate(ResponseRequest) -> ResponseResponse
```

The application uses these without assuming OpenAI, GPT, or any particular model
API. Concrete integrations own the API-specific mapping.

### Vector documents

`VectorDocument[TMetadata]` is the provider-neutral unit of storage:

- `id`: stable string ID
- `text`: the original text
- `embedding`: `list[float]`
- `chunk_index`: position of the chunk within its source
- `metadata`: a Pydantic model chosen by the consuming application

It contains no Azure, Foundry, client, or credential types. rag-core does not
prescribe which metadata fields exist.

### Search

```text
VectorSearchRequest   query_vector, top_k (positive), filters (optional)
VectorSearchResult    id, text, typed metadata, score
VectorSearchResponse  results: list[VectorSearchResult]
```

`VectorSearchRequest` is used by vector search only. Keyword, hybrid, and semantic
search take a text query (and, for hybrid, a query vector) plus `top_k` and
`filters` directly on `AzureAISearchVectorStore`.

Filters are `dict[str, str]`. In the Azure AI Search implementation each entry
becomes an exact string-equality OData clause (`field eq 'value'`) with
apostrophes escaped, so a filter key must name a filterable index field.

### Retrieval and RAG

```text
retrieve(..., method="vector") -> VectorSearchResponse
RAGResponse[TMetadata]         answer: str, evidence: list[VectorSearchResult]
```

`method` is one of `vector`, `keyword`, `hybrid`, `semantic` (default `vector`).
Vector and hybrid embed the query; keyword and semantic do not. When retrieval
finds nothing, the RAG use case returns an explicit insufficient-evidence answer
without calling the `ResponseModel`.

### Vector store

There is no generic `VectorStore` contract. The concrete
`AzureAISearchVectorStore` works on an already-provisioned index and does not
create or update it. Its metadata model must be compatible with the index fields.

## Part 2: Example application contracts

The following belong to the example candidate-document `/api` application and the
Azure index it uses, not to the generic core.

### Example metadata: `DocumentChunkMetadata`

```text
source_document_id: str
source_name: str
document_type: str
page_number: int | None = None
source_url: str | None = None
candidate_id: str
document_id: str
```

The model currently lives under `domain/models`, but its candidate fields are
example-specific; a different consumer would define its own metadata model. The
Azure store flattens metadata into top-level index fields, so the index must define
matching fields. The current canonical index (`build_azure_search_index`) is the
concrete representation of this example and adds `id`, `text`, `embedding`, and
`chunk_index`. See ARCHITECTURE.md for how model and index evolve together.

### Example API models (`api/schemas.py`)

```text
IngestDocumentRequest   candidate_id, text                       (both non-empty)
IngestDocumentResponse  candidate_id, document_id, chunk_ids
QueryRequest            query, candidate_id, method="vector", top_k=5 (1..50)
QueryResponse           query, candidate_id, method, answer, chunks
QueryChunk              id, text, score, metadata
```

`IndexSetupResponse`, `IndexClearResponse`, `DeleteDocumentResponse`, and
`DeleteChunkResponse` are the administrative responses. `QueryMethod` mirrors the
application's `SearchMethod` values; they are kept separate so the application does
not import the API layer.