# Implementation Plan

Build the repository in small, verifiable increments.

Do not ask Copilot to implement the entire repository in one prompt.

## Phase 0 — Project foundation

Create:

- uv project
- Python package
- test package
- Ruff configuration
- pytest configuration
- `.gitignore`
- `.env.example`
- README

Exit criteria:

- package imports
- pytest runs
- Ruff runs

## Phase 1 — Configuration

Implement typed settings.

Responsibilities:

- environment loading
- provider selection
- model names
- vector-store configuration

Do not create provider clients yet.

Exit criteria:

- settings can be loaded in a unit test
- secrets are not committed

## Phase 2 — Domain contracts

Implement:

- `VectorDocument` as provider-neutral stored data
- `VectorStoreSchema` as provider-neutral storage structure
- `ChatMessage`
- `ChatOptions`
- `ChatRequest`
- `ChatResponse`
- `EmbeddingRequest`
- `EmbeddingResponse`
- `VectorSearchRequest`
- `VectorSearchResult`
- `VectorSearchResponse`
- evidence models
- `RAGResponse`

Implement:

- `ChatModel`
- `EmbeddingModel`
- `VectorStore`

Keep `VectorDocument`, `VectorStoreSchema`, and `VectorStore` separate. Do not
make them inherit from each other or introduce generic document/schema type
coupling without a demonstrated need.

Exit criteria:

- domain tests pass
- no provider SDK imports in domain

## Phase 3 — OpenAI Responses adapter

Implement:

`OpenAIResponsesChatModel`

Responsibilities:

- receive `ChatRequest`
- map to Responses API input
- invoke OpenAI SDK
- map provider response to `ChatResponse`

Do not let provider objects escape.

Exit criteria:

- adapter unit tests with mocked SDK
- one explicit integration test can call the real provider when credentials are available

## Phase 4 — Embedding adapter

Implement:

`OpenAIEmbeddingModel`

Responsibilities:

- receive `EmbeddingRequest`
- call embedding endpoint
- map output to `EmbeddingResponse`

Support batch input.

Exit criteria:

- mocked unit tests
- optional real embedding smoke test

## Phase 5 — Azure AI Search adapter

Implement:

`AzureAISearchVectorStore`

Responsibilities:

- translate canonical search request
- map `VectorDocument` values to and from Azure AI Search representations
- perform vector/hybrid search as supported
- map Azure results to `VectorSearchResponse`

Do not expose Azure Search SDK objects.
The runtime adapter uses an already-provisioned index and does not create it.

Implement index provisioning separately in a setup/operations script such as
`scripts/setup_vector_index.py`. The script uses a concrete
`VectorStoreSchema` to create or configure the Azure AI Search index; it does
not make index creation a runtime `VectorStore` responsibility.

Exit criteria:

- mocked adapter tests
- setup-script tests that do not require Azure credentials
- optional real provisioning and adapter smoke tests against a clean development index

## Phase 6 — Retrieval

Implement:

`EvidenceRetriever`

Dependencies:

- `EmbeddingModel`
- `VectorStore`

Flow:

```text
query
  -> embedding
  -> vector search
  -> normalized evidence
```

Add candidate/case-like scope only if needed by generic retrieval. Do not introduce BGV-specific terminology into the generic repository.

Exit criteria:

- application tests use fake embedding/vector implementations
- no provider SDK imports

## Phase 7 — Context and prompt

Implement:

- `ContextBuilder`
- `PromptBuilder`

Responsibilities must remain separate.

ContextBuilder:

```text
Evidence -> formatted context
```

PromptBuilder:

```text
query + context -> ChatRequest
```

Exit criteria:

- deterministic unit tests
- insufficient-evidence instruction tested

## Phase 8 — RAG pipeline

Implement:

`RAGPipeline`

Dependencies:

- `EvidenceRetriever`
- `ContextBuilder`
- `PromptBuilder`
- `ChatModel`

Flow:

```text
query
 -> retrieve
 -> build context
 -> build ChatRequest
 -> generate
 -> RAGResponse
```

Exit criteria:

- pipeline test uses fakes only
- provider-independent application layer

## Phase 9 — Composition

Implement factories/composition root.

Configuration determines:

```text
ChatModel implementation
EmbeddingModel implementation
VectorStore implementation
```

The application receives constructed dependencies.

Exit criteria:

- switching implementation requires configuration/composition changes, not RAG logic changes

## Phase 10 — Evaluation

Add a small evaluation dataset.

Measure:

- retrieval relevance
- evidence coverage
- groundedness
- citation correctness
- insufficient-evidence behavior

Do not treat retrieval scores as probabilities.

## Phase 11 — Hardening

Add only after the core works:

- structured logging
- retry/error translation
- timeouts
- provider error mapping
- telemetry
- token/usage accounting
- optional caching
- streaming if required

Do not add these before the basic architecture is working.

## Phase 12 — Extensions

Possible future work:

- additional chat providers
- additional embedding providers
- additional vector stores
- reranker abstraction
- structured output
- tool calling
- conversation state
- evaluation framework
- agent orchestration

These are future extensions, not prerequisites for Vanilla RAG.
