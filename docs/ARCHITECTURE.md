# Architecture

## 1. Architectural principles

The system follows dependency inversion and capability-oriented interfaces.

The application must not know whether a model is supplied by OpenAI, Azure OpenAI, Anthropic, Google, or another provider.

Likewise, retrieval must not know whether storage is Azure AI Search, Qdrant, Pinecone, PostgreSQL/pgvector, or another implementation.

## 2. Layers

```text
src/rag-core/
|
+-- domain/
|   +-- models/
|   +-- interfaces/
|
+-- application/
|   +-- retrieval/
|   +-- rag/
|
+-- infrastructure/
|   +-- llm/
|   +-- embeddings/
|   +-- vectorstores/
|
+-- composition/
+-- scripts/
    +-- setup_vector_index.py
```

### Domain

Contains provider-neutral models and interfaces.

Examples:

- `VectorDocument`
- `VectorStoreSchema`
- `ChatRequest`
- `ChatResponse`
- `EmbeddingRequest`
- `EmbeddingResponse`
- `VectorSearchRequest`
- `VectorSearchResponse`
- `ChatModel`
- `EmbeddingModel`
- `VectorStore`

No provider SDK imports.

### Application

Contains RAG use cases.

Examples:

- `EvidenceRetriever`
- `ContextBuilder`
- `PromptBuilder`
- `RAGPipeline`

No provider SDK imports.

### Infrastructure

Contains adapters to external systems.

Examples:

- `OpenAIResponsesChatModel`
- `OpenAIEmbeddingModel`
- `AzureAISearchVectorStore`

Provider SDK imports are allowed here.

Azure service authentication uses Microsoft Entra ID through
`DefaultAzureCredential` from `azure-identity`. The infrastructure/composition
boundary creates and supplies credentials to Azure SDK clients. API keys are
not part of this project's Azure authentication approach. The domain and
application layers must not import Azure Identity.

`AzureAISearchVectorStore` is the provider-specific runtime adapter. It maps
provider-neutral documents and search requests to Azure AI Search SDK types and
maps search results back to application/domain representations. Azure SDK
types stay inside infrastructure.

Index provisioning is separate from runtime storage. A setup script such as
`scripts/setup_vector_index.py` applies a concrete `VectorStoreSchema` to Azure
AI Search. Runtime `VectorStore` implementations use the provisioned index;
they do not create it.

### Composition

Builds the concrete object graph from configuration.

Example:

```text
ChatModel       -> OpenAIResponsesChatModel
EmbeddingModel  -> OpenAIEmbeddingModel
VectorStore     -> AzureAISearchVectorStore
```

## 3. Dependency graph

```text
                 RAGPipeline
                     |
          +----------+----------+
          |          |          |
          v          v          v
      Retriever  PromptBuilder ChatModel
          |
     +----+----+
     |         |
     v         v
Embedding   VectorStore
   Model
```

The interfaces are defined in the inner/domain layer.

Concrete implementations live in infrastructure.

## 3.1 Vector storage responsibilities

Vector storage uses three independent concepts:

| Concept | Responsibility |
| --- | --- |
| `VectorDocument` | The data to store: an identifier, text, embedding, and metadata. |
| `VectorStoreSchema` | The expected storage structure, including fields and index-level configuration. |
| `VectorStore` | The provider-neutral runtime contract for operations such as upserting, searching, retrieving, and deleting documents. |

These concepts have explicit contracts between them, not inheritance
relationships with each other. A concrete document must be representable by
the schema for its intended index, but the mapping/validation mechanism and
exact method signatures will be designed when these components are
implemented. Do not introduce generic document/schema type coupling before a
demonstrated need.

For Azure AI Search, the intended flow is:

```text
VectorDocument -> AzureAISearchVectorStore -> Azure AI Search
VectorStoreSchema -> setup_vector_index.py -> Azure AI Search index
```

The schema describes the desired structure, the setup script provisions it,
the runtime store operates on the existing index, and documents supply data.

## 4. Chat abstraction

```text
ChatRequest
    |
    v
ChatModel.generate(...)
    |
    v
ChatResponse
```

Initial implementation:

```text
OpenAIResponsesChatModel
```

The adapter translates our canonical request into the OpenAI Responses API request and translates the provider response into `ChatResponse`.

The rest of the codebase must never depend on OpenAI response objects.

## 5. Embedding abstraction

```text
EmbeddingRequest
    |
    v
EmbeddingModel.embed(...)
    |
    v
EmbeddingResponse
```

The concrete embedding adapter owns provider-specific API calls.

## 6. Vector-store abstraction

```text
VectorSearchRequest
    |
    v
VectorStore.search(...)
    |
    v
VectorSearchResponse
```

The first implementation is Azure AI Search.

The abstraction is intentionally limited to capabilities needed by this project.

## 7. Why Strategy + Adapter

Strategy gives the application interchangeable implementations.

Adapter converts a provider's SDK/API contract into our canonical contract.

Example:

```text
                    ChatModel
                       ^
                       |
            +----------+----------+
            |                     |
            | implements          | implements
            |                     |
OpenAIResponsesChatModel   OtherProviderChatModel
            |                     |
            v                     v
     OpenAI Responses        Other Provider
          API                     API
```

## 8. Why request/response objects

Do not expose raw SDK request/response types.

Instead:

```text
Application
    |
    | ChatRequest
    v
ChatModel
    |
    | provider mapping
    v
Provider SDK
    |
    | provider response
    v
Adapter
    |
    | ChatResponse
    v
Application
```

This gives us a stable internal contract while provider APIs change independently.

## 9. What should not be abstracted

Do not create abstractions simply because they are theoretically possible.

Examples:

- no generic `AIProvider` with every AI capability
- no universal provider parameter bag
- no generic repository hierarchy
- no provider-independent imitation of every vector database feature

Abstract stable application capabilities, not every external API detail.
