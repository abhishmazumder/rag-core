# Architecture

This is the canonical architecture description. Stable data models are listed in
[CONTRACTS.md](CONTRACTS.md) and the rationale for choices in
[DECISIONS.md](DECISIONS.md).

## 1. Two levels

```text
Generic rag-core
      ↓
Concrete infrastructure (Azure AI Search, Azure AI Foundry, Azure Identity)
      ↓
Example /api application
      ↓
Candidate-document example
      ↓
Azure AI Search index representing that example's needs
```

**rag-core** is a reusable RAG engine. Its domain and application code speaks in
generic terms: documents/chunks, embeddings, retrieval, search methods, response
generation, and typed metadata. It does not depend on candidates, background
verification, or any other business domain.

**The `/api` layer** is an example application built on rag-core. It shows how to
use the engine for a candidate-document use case. `candidate_id`, the ingestion
workflow, and the other candidate-specific fields belong to this example, not to
the core.

## 2. Platform boundaries

- **Azure AI Search** is the sole vector store. `AzureAISearchVectorStore` is the
  concrete implementation. There is no generic `VectorStore` interface, protocol,
  or schema abstraction, and no provider switching.
- **Azure AI Foundry** is the model access environment, not a model abstraction.
  Embedding and response models are selected through configuration. Each concrete
  integration uses the Foundry API supported by its deployment; `/openai/v1` is
  one supported interface, not a universal one.
- **Microsoft Entra ID** via `DefaultAzureCredential` authenticates all Azure
  services. A shared infrastructure credential factory supplies credentials. API
  keys are not part of the authentication approach.

## 3. Layers

```text
src/rag_core/
+-- domain/          provider-neutral models and capability contracts
+-- application/     use cases: chunking, retrieval, RAG generation
+-- infrastructure/  Azure AI Search store, Foundry model integrations, identity
+-- composition/     wires Settings to concrete implementations
+-- api/             example FastAPI application
```

- **Domain**: provider-neutral data and capabilities. It imports no Azure, Foundry,
  or model-provider SDK types and no credentials.
- **Application**: use cases that depend on domain capabilities. It calls the
  injected `ResponseModel` and `EmbeddingModel` directly, and intentionally depends
  on the concrete `AzureAISearchVectorStore`, since Azure AI Search is the only
  vector store. It never imports provider SDKs or constructs implementations.
- **Infrastructure**: all provider SDK usage lives here (see below).
- **Composition**: `create_response_model`, `create_embedding_model`, and
  `create_vector_store` read `Settings`, validate it, create the shared credential,
  and construct the concrete classes. They are plain functions, not a registry or
  factory hierarchy. Constructing a store never creates or updates an index.
- **API**: HTTP concerns only (see section 7).

## 4. Model capabilities

```text
EmbeddingModel.embed(EmbeddingRequest) -> EmbeddingResponse
ResponseModel.generate(ResponseRequest) -> ResponseResponse
```

These are small capability contracts. Endpoints and deployment names are
configuration; response and embedding settings are separate because deployments
can expose different APIs. Concrete integrations in infrastructure own all
API-specific mapping:

- `OpenAIResponseModel` uses the OpenAI SDK against the OpenAI-compatible Foundry
  Responses API.
- `OpenAIEmbeddingModel` uses the corresponding embeddings API.
- Both pass an Entra ID bearer-token provider (Foundry scope) to the SDK's
  `api_key` parameter, never a static key.

These are the first integrations, not a statement that OpenAI is the universal
model implementation. There is no shared Foundry client hierarchy; add shared code
only when concrete integrations demonstrate common behavior.

## 5. Vector store and search methods

`AzureAISearchVectorStore` operates on an existing index and offers:

| Operation | Purpose |
| --- | --- |
| `upsert` / `upsert_many` | store `VectorDocument` values |
| `vector_search` | similarity search over the `embedding` field |
| `keyword_search` | full-text search |
| `hybrid_search` | keyword plus vector |
| `semantic_search` | Azure semantic ranking |
| `get_by_id`, `delete`, `delete_by_document_id`, `clear` | retrieval and removal |

Filters are application-level `dict[str, str]` entries translated to exact-equality
OData clauses. The store receives the Pydantic metadata model so results are
reconstructed with the same typed shape used at upsert time.

`index_provisioner.py` (infrastructure) builds the index with Azure SDK types and
creates or updates it. Provisioning is an explicit operation, never part of
application startup or the runtime store.

### Retrieval flow

The search method values are `vector`, `keyword`, `hybrid`, and `semantic`. The
default is `vector`.

```text
/api -> generate_rag_response -> retrieve(method=...) -> AzureAISearchVectorStore
        -> vector_search | keyword_search | hybrid_search | semantic_search
```

`retrieve` is the only place that maps a method to a store operation. Vector and
hybrid search need a query embedding, so `EmbeddingModel` is called only for those
two; keyword and semantic search do not embed. There is no search-strategy
abstraction.

### RAG generation

`generate_rag_response` retrieves evidence, formats it as delimited records (ID,
text, JSON metadata), and asks the `ResponseModel` to answer only from that
evidence, to say when it is insufficient, and to treat evidence as data rather
than instructions. Scores are not sent in the prompt. With no evidence it returns
an explicit insufficient-evidence answer without calling the model. The result is a
`RAGResponse` holding the answer and the original evidence.

## 6. Metadata, index, and deployment

A concrete deployment is the combination of three things:

```text
Generic VectorDocument[TMetadata]
          +
   application metadata model
          +
 compatible Azure AI Search index
          =
   concrete deployment
```

rag-core itself only requires that a document have an ID, text, an embedding, and
typed metadata. The metadata model is chosen by the consuming application, and the
index must define fields that match it, since the store flattens metadata into
top-level index fields.

The example application uses `DocumentChunkMetadata` and the canonical index built
by `build_azure_search_index`. The index fields are `id`, `text`, `embedding`,
`chunk_index`, plus the example metadata fields `candidate_id`, `document_id`,
`source_document_id`, `source_name`, `document_type`, `page_number`, and
`source_url`. These metadata fields are specific to the candidate-document example,
not mandatory for rag-core. `chunk_index` is a property of `VectorDocument`, not of
the metadata.

If the consuming system's metadata needs change, the metadata model and the index
schema can change together, provided they stay compatible with the runtime mapping
and the retrieval contracts. This does not require changing rag-core unless a core
contract is affected.

## 7. Example `/api` application

The API wires rag-core to the candidate-document example through dependencies in
`api/dependencies.py` (built on composition). Handlers never construct Azure SDK
clients.

| Endpoint | Purpose |
| --- | --- |
| `POST /documents` | ingest raw text for a candidate |
| `POST /query` | answer a question scoped to a candidate, with a search `method` |
| `POST /admin/index/setup` | create or update the index |
| `POST /admin/index/clear` | delete all documents, keeping the index |
| `DELETE /documents/{document_id}` | delete all chunks of a document |
| `DELETE /chunks/{chunk_id}` | delete one chunk |

Ingestion is implemented directly in the `POST /documents` handler as an example
workflow:

```text
Example API input:      candidate_id, text
Example API generates:  document_id, chunks, embeddings, chunk IDs, candidate/document metadata
```

It chunks with the reusable `application/chunking.py` (recursive, 1000 characters,
100 overlap), embeds through `EmbeddingModel`, builds deterministic chunk IDs, and
upserts. This workflow is not a requirement of rag-core. Each request generates a
new `document_id`, so re-ingesting never overwrites earlier documents.

`/query` filters by `candidate_id` and returns the answer plus the retrieved
chunks (`id`, `text`, `score`, `metadata`). These API request/response models are
example-level contracts.

## 8. Dependency direction

```text
Domain contracts -> Application use cases -> Infrastructure integrations
```

Domain defines contracts and application builds on them; infrastructure implements
them and composition wires them together. Provider-specific types never flow back
into domain or application code.