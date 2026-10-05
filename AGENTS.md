# Vanilla RAG — Agent Instructions

## Purpose

This repository is a standalone learning and reference implementation of a provider-neutral Vanilla RAG system in Python.

The project is intentionally built without LangChain, LangGraph, LlamaIndex, or another RAG framework. The purpose is to understand and implement the RAG mechanics directly before introducing orchestration frameworks.

The implementation must remain useful as a foundation for later agentic-AI work.

## Primary engineering goals

1. Keep RAG application logic independent of model providers.
2. Keep embedding logic independent of embedding providers.
3. Keep vector-storage logic independent of any particular vector database/search engine.
4. Hide provider SDKs behind adapters.
5. Use explicit request/response domain contracts.
6. Keep interfaces small and capability-oriented.
7. Make dependencies explicit through dependency injection.
8. Prefer simple, readable OOP over premature framework-style abstraction.
9. Add tests as each component is implemented.
10. Do not silently introduce provider-specific concepts into domain/application code.

## Current technology choices

- Python 3.12+
- uv for project/environment/dependency management
- Pydantic for domain/request/response models
- pytest for tests
- Ruff for formatting/linting
- OpenAI Responses API for the initial chat-model adapter
- OpenAI embeddings for the initial embedding adapter
- Azure AI Search for the initial vector-store adapter
- Environment configuration through `.env` / typed settings
- No API keys committed to source control

These are implementation choices, not domain abstractions. The rest of the code must not become coupled to them.

## Architecture rule

The application must depend on our own contracts:

- `ChatModel`
- `EmbeddingModel`
- `VectorStore`

Provider implementations depend on those contracts.

Do not make application code import:

- OpenAI SDK classes
- Azure AI Search SDK classes
- Anthropic SDK classes
- Qdrant/Pinecone/etc. SDK classes

Provider SDK imports belong under `infrastructure/`.

## Required patterns

Use these patterns deliberately:

- Strategy: interchangeable model/storage implementations
- Adapter: provider SDK → our canonical contracts
- Dependency Injection: supply implementations to application services
- Factory/composition: select concrete implementations from configuration
- Dependency Inversion: inner/application code depends on abstractions
- Interface Segregation: keep ChatModel, EmbeddingModel, and VectorStore separate
- Single Responsibility: one clear responsibility per class

Do not add patterns merely for pattern's sake.

Avoid introducing:

- Service Locator
- Generic Repository abstractions with no need
- Abstract Factory hierarchies
- Event buses
- Mediators
- Microservices
- DI frameworks
- excessive base classes

## Canonical contracts

Provider-neutral request/response models must be designed before provider adapters.

At minimum:

- `ChatRequest`
- `ChatResponse`
- `ChatMessage`
- `ChatOptions`
- `EmbeddingRequest`
- `EmbeddingResponse`
- `VectorSearchRequest`
- `VectorSearchResponse`
- `VectorSearchResult`

The application passes these objects across abstraction boundaries.

Provider adapters map them to and from provider-specific SDK/API representations.

Never leak provider response objects outside an adapter.

## Chat model rule

The first implementation uses the OpenAI Responses API.

The rest of the application must call our interface, for example:

`ChatModel.generate(ChatRequest) -> ChatResponse`

Do not spread `client.responses.create(...)` through the codebase.

The OpenAI adapter owns all Responses API mapping.

Do not design the abstraction around Chat Completions. If another provider is added later, its adapter maps the same canonical contract to that provider's API.

## Embedding rule

The application must depend on `EmbeddingModel`, not an OpenAI/Azure SDK.

The embedding interface should support both single and batch embedding use cases without exposing provider-specific response types.

Embedding dimensionality is a property of a concrete embedding model/index configuration, not a universal assumption of the domain.

## Vector-store rule

The application must depend on `VectorStore`.

The first implementation is Azure AI Search.

The abstraction should expose only the capabilities required by this Vanilla RAG application. Do not create a fake universal vector-database API containing every feature of every provider.

Vector-store adapters own provider-specific filtering, hybrid/vector/semantic query translation, and SDK objects.

Keep these vector-storage concepts separate:

- `VectorDocument` represents provider-neutral document data.
- `VectorStoreSchema` represents expected storage/index structure.
- `VectorStore` represents runtime storage operations.

They do not inherit from one another. Do not introduce generic document/schema
type coupling without a demonstrated need. The separate
`scripts/setup_vector_index.py` setup/operations script applies a concrete
schema to provision an index; runtime `VectorStore` implementations use the
index and do not create it. Azure AI Search SDK usage and mapping belong in
`AzureAISearchVectorStore` under infrastructure.

## RAG flow

The core flow is:

1. Receive user query and retrieval scope.
2. Build retrieval request.
3. Generate query embedding through `EmbeddingModel`.
4. Search through `VectorStore`.
5. Return provider-neutral evidence.
6. Build grounded context.
7. Build model request.
8. Generate through `ChatModel`.
9. Return a provider-neutral RAG response containing answer and evidence/citations.

Retrieval, context assembly, prompt construction, and generation must remain separate responsibilities.

## Grounding rule

The RAG answer must be grounded in retrieved evidence.

If the retrieved evidence is insufficient, the generated answer must explicitly say that the available evidence is insufficient rather than inventing an answer.

The system should preserve evidence references so the answer can be audited.

## Testing rule

Every new abstraction must have tests.

Prefer:

- unit tests for domain models
- unit tests for context/prompt logic
- adapter tests with mocked provider SDKs
- application-service tests with fake interfaces
- a small number of integration tests for real provider connections

Do not make the entire unit-test suite dependent on Azure/OpenAI credentials.

A fake `ChatModel`, `EmbeddingModel`, or `VectorStore` should be easy to construct for tests.

## Configuration rule

Keep configuration separate from domain models.

Provider selection and deployment/model configuration belong in configuration/composition.

Do not put secrets in source code.

`.env` is local-only and must be gitignored.

`.env.example` must document required configuration keys without real secrets.

## Coding style

- Python type hints are required for public methods.
- Prefer Pydantic models for external/application contracts.
- Prefer immutable/value-like domain data where practical.
- Use descriptive names.
- Keep methods small.
- Avoid unnecessary inheritance.
- Avoid global mutable clients.
- Do not use `Any` unless there is a documented reason.
- Raise meaningful exceptions at clear boundaries.
- Do not catch broad `Exception` unless translating/logging at a deliberate boundary.

## Dependency direction

Preferred dependency direction:

`domain -> application -> infrastructure/composition`

More precisely, application/domain define contracts; infrastructure implements them. Infrastructure must not force provider-specific concepts into domain/application models.

## Implementation discipline

Work incrementally.

Before implementing a major component:

1. Read the relevant design document.
2. Check existing interfaces/models.
3. Implement the smallest useful change.
4. Add/update tests.
5. Run formatting/linting/tests.
6. Do not implement future phases just because they are mentioned in a design document.

If a requested change conflicts with this architecture, explain the conflict and propose the smallest architectural change needed.

## Repository scope

This repository is `rag-core`.

It is NOT the BGV capstone repository.

Do not introduce BGV-specific domain concepts such as candidates, cases, background checks, employment verification, or identity verification into the generic RAG core.

BGV integration will happen later as a separate concern.

## Documentation rule

When an architectural decision changes, update the relevant Markdown design document.

Important decisions should be recorded in `docs/decisions/`.

Do not silently change the architecture through implementation.

## Current milestone

The immediate milestone is to establish the generic contracts and clean project foundation, then implement:

1. project setup with uv
2. configuration
3. domain contracts/models
4. provider adapters
5. vector-store adapter
6. retrieval
7. context assembly
8. prompt construction
9. generation
10. end-to-end Vanilla RAG
11. evaluation
12. observability/error handling improvements
13. optional extensions

Do not jump directly to agents or LangGraph.
