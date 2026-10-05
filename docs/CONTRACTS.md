# Domain Contracts

These contracts are the most important LLD boundary in the repository.

They must remain provider-neutral.

## 1. Chat

### ChatMessage

Represents one logical conversational message.

Fields:

- `role`
- `content`

Initial roles:

- `system`
- `user`
- `assistant`

Do not import provider message classes.

### ChatOptions

Contains only options that are meaningful across providers.

Initial candidates:

- `temperature`
- `max_output_tokens`

Provider-specific controls do not belong here.

### ChatRequest

Conceptual shape:

```text
ChatRequest
├── messages: list[ChatMessage]
└── options: ChatOptions | None
```

The model selection should normally be configuration-owned.

If per-request model selection is eventually required, it must remain provider-neutral.

### ChatResponse

Conceptual shape:

```text
ChatResponse
├── content: str
├── model: str
└── finish_reason: str | None
```

Do not expose OpenAI `Response`, Anthropic response objects, or another SDK type.

### ChatModel

Conceptual contract:

```python
generate(request: ChatRequest) -> ChatResponse
```

## 2. Embeddings

### EmbeddingRequest

```text
EmbeddingRequest
└── texts: list[str]
```

Batch support is part of the contract because ingestion commonly embeds many chunks.

### EmbeddingResponse

```text
EmbeddingResponse
├── embeddings: list[list[float]]
└── model: str
```

The application should not need to know the provider SDK response structure.

### EmbeddingModel

Conceptual contract:

```python
embed(request: EmbeddingRequest) -> EmbeddingResponse
```

## 3. Vector storage and search

### VectorDocument

`VectorDocument` represents the data stored in a vector store. It is intended
to be a generic base that concrete document types can extend. Its common data
includes:

- `id`
- `text`
- `embedding`
- `metadata`

It is provider-neutral data, not a place for Azure AI Search SDK behavior.
Concrete document types must be representable by the schema of their intended
store. The exact mapping and validation mechanism will be designed when the
document and schema components are implemented.

### VectorStoreSchema

`VectorStoreSchema` describes the storage structure expected by a vector
store. It may describe document fields, field types, vector field and
dimensions, searchable/filterable properties, metadata fields, and other
index-level configuration.

Concrete schemas may extend the schema concept. Schema represents storage
structure; it is not the runtime store or the stored document.

`VectorDocument`, `VectorStoreSchema`, and `VectorStore` are separate concepts.
They do not inherit from each other, and no generic type relationship between
a document and a schema is required at this stage.

### VectorSearchRequest

Initial conceptual fields:

```text
VectorSearchRequest
├── vector: list[float]
├── top_k: int
└── filters: dict[str, str] | None
```

Do not expose Azure `VectorizedQuery`.

If hybrid/semantic search becomes part of the generic contract, add explicit capability-oriented fields rather than provider SDK objects.

### VectorSearchResult

Must contain enough information for RAG evidence assembly.

Initial fields should include:

- document/chunk identifier
- text
- metadata required for citation
- retrieval score where available

Scores should be treated as retrieval metadata, not as probabilities.

### VectorSearchResponse

```text
VectorSearchResponse
└── results: list[VectorSearchResult]
```

### VectorStore

`VectorStore` is the provider-neutral runtime storage contract. Its
responsibilities may include adding/upserting documents, searching/querying
vectors, retrieving results, and deleting documents. Exact operations and
signatures will be designed when the storage contract is implemented.

Conceptual search contract:

```python
search(request: VectorSearchRequest) -> VectorSearchResponse
```

Provider-specific SDK types and request construction remain inside the
infrastructure adapter. Runtime `VectorStore` implementations use a
provisioned index; they do not create it.

## 4. Index provisioning

Index structure is described by a concrete `VectorStoreSchema`. A separate
setup/operations script, conceptually `scripts/setup_vector_index.py`, applies
that schema to the selected provider's index. For Azure AI Search, that script
owns index provisioning/configuration; the runtime `AzureAISearchVectorStore`
uses the resulting index.

This separates:

- document data (`VectorDocument`)
- storage structure (`VectorStoreSchema`)
- runtime storage operations (`VectorStore`)
- provisioning operations (`setup_vector_index.py`)

## 5. Evidence

The RAG application should normalize retrieved records into a provider-neutral evidence representation.

A generic evidence object should be able to preserve:

- stable evidence/chunk ID
- source document ID
- source document name
- source URL/reference
- text
- metadata
- retrieval score

Do not introduce BGV-specific fields into this generic model.

## 6. RAG response

The RAG pipeline should return a provider-neutral response containing at minimum:

```text
RAGResponse
├── answer
└── evidence
```

Citations should be traceable back to the evidence objects.

## 7. Contract design rule

If a new provider requires a field that existing providers do not need, do not immediately add that field to the canonical contract.

Ask:

1. Is this capability common?
2. Is it required by the application?
3. Can it be represented without provider-specific semantics?

If not, keep it inside the adapter.
