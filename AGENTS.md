# Vanilla RAG — Agent Instructions

## Purpose

This repository is a standalone learning and reference implementation of a Vanilla RAG system in Python.

The project is intentionally built without LangChain, LangGraph, LlamaIndex, or another RAG framework. The purpose is to understand and implement the RAG mechanics directly before introducing orchestration frameworks.

The implementation must remain useful as a foundation for later agentic-AI work.

## Primary engineering goals

1. Keep RAG application logic independent of model-specific APIs.
2. Keep embedding logic behind a small capability contract.
3. Use Azure AI Search as the sole vector store.
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
- Azure AI Foundry as the model platform for configurable response and embedding models
- Azure AI Search as the sole vector store
- Environment configuration through `.env` / typed settings
- No API keys committed to source control

These are implementation choices, not domain abstractions. The rest of the code must not become coupled to them.

## Architecture rule

The application must depend on the model capabilities and the concrete Azure
AI Search integration selected by this architecture:

- `ResponseModel`
- `EmbeddingModel`
- `AzureAISearchVectorStore`

Model integrations depend on `ResponseModel` or `EmbeddingModel` and use the
Foundry API supported by the configured model. Azure AI Search is not selected
through a generic provider abstraction.

Do not make application code import:

- OpenAI SDK classes
- Azure AI Search SDK classes

Azure/model SDK imports belong under `infrastructure/`.

## Required patterns

Use these patterns deliberately:

- Adapter: a model-specific Foundry API → our capability contracts
- Dependency Injection: supply implementations to application services
- Composition: wire configured concrete model integrations and Azure AI Search
- Dependency Inversion: inner/application code depends on abstractions
- Interface Segregation: keep `ResponseModel` and `EmbeddingModel` separate
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

Provider-neutral request/response models must be designed before model
integrations.

At minimum:

- `ResponseRequest`
- `ResponseResponse`
- `ResponseMessage`
- `EmbeddingRequest`
- `EmbeddingResponse`
- `VectorSearchRequest`
- `VectorSearchResponse`
- `VectorSearchResult`

The application passes these objects across abstraction boundaries.

Concrete Foundry integrations map them to and from the model API used by the
configured deployment.

Never leak provider response objects outside an adapter.

## Response model rule

The application uses `ResponseModel.generate(ResponseRequest) ->
ResponseResponse`. Do not hard-code OpenAI, GPT, or a specific model API into
the application/domain layers. Each concrete Azure AI Foundry integration owns
the request/response mapping supported by its configured model endpoint.

## Embedding rule

The application must depend on `EmbeddingModel`, not a model SDK.

The embedding interface should support batch use without exposing
model-specific response types. Its implementation uses Azure AI Foundry and
the API supported by the configured model.

Embedding dimensionality is a property of a concrete embedding model/index configuration, not a universal assumption of the domain.

## Vector-store rule

Azure AI Search is the sole vector-store implementation. Use one concrete
`AzureAISearchVectorStore`. There is no generic `VectorStore` interface,
protocol, or schema, and none must be added; also avoid provider-selection
factories and search-strategy abstractions. Its runtime operations use an
already-provisioned index. Provisioning belongs to a separate setup component.

Keep vector data, runtime operations, and provisioning separate:

- `VectorDocument` represents provider-neutral document data.
- `AzureAISearchVectorStore` represents runtime Azure AI Search operations.

Azure AI Search schema definitions use Azure SDK types directly in
infrastructure/provisioning; there is no separate domain schema model. The
canonical `build_azure_search_index` in `index_provisioner.py` defines the index and
`create_or_update_azure_search_index` creates or updates it.
`AzureAISearchVectorStore` uses an existing index and does not create it.
Azure SDK imports remain out of domain and application code.

## RAG flow

The core flow is:

1. Receive user query and retrieval scope.
2. Build retrieval request.
3. Generate query embedding through `EmbeddingModel`.
4. Search through `AzureAISearchVectorStore`.
5. Return provider-neutral evidence.
6. Build grounded context.
7. Build model request.
8. Generate through `ResponseModel`.
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

Do not make the unit-test suite dependent on Azure credentials or live Foundry
model calls.

A fake `ResponseModel`, `EmbeddingModel`, or Azure Search client should be
easy to construct for tests.

## Configuration rule

Keep configuration separate from domain models.

Model endpoint and model/deployment configuration belong in settings and
composition. Azure AI Search is not provider-selected.

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

Do not introduce BGV-specific domain concepts such as candidates, cases, background checks, employment verification, or identity verification into the generic RAG core. Candidate/BGV concepts are never requirements of rag-core.

The `/api` package is an example application/use case built on rag-core: a candidate-document workflow with its own request/response schemas, ingestion logic in `POST /documents`, example metadata (`DocumentChunkMetadata`), and a compatible Azure AI Search index. That metadata model and index may evolve together without changing rag-core, unless a core contract changes.

BGV integration will happen later as a separate concern.

## Documentation rule

When an architectural decision changes, update the relevant Markdown design document.

Important decisions should be recorded in `docs/DECISIONS.md`. `docs/ARCHITECTURE.md` is the canonical architecture description.

Do not silently change the architecture through implementation.

## Current milestone

The contracts, Azure integrations, retrieval (vector/keyword/hybrid/semantic), RAG generation, chunking, and the example `/api` are implemented. See `docs/IMPLEMENTATION_PLAN.md` for status; the next steps are a real Azure smoke test, evaluation, and hardening.

Do not jump directly to agents or LangGraph.
