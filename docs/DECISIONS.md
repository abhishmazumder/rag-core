# Architecture Decisions

## ADR-001 — Standalone Vanilla RAG repository

### Decision

Vanilla RAG is implemented as its own repository, separate from the BGV capstone.

### Reason

The RAG implementation is a generic learning artifact and should not be polluted by BGV-specific domain assumptions.

BGV integration can happen later.

---

## ADR-002 — Provider-neutral interfaces

### Decision

> Superseded in part by ADR-011. Originally application code was to depend on
> `ChatModel`, `EmbeddingModel`, and `VectorStore`. In the current design these
> are `ResponseModel` and `EmbeddingModel` capabilities only; there is no
> generic `VectorStore`.

### Decision

Application code depends on small model capability contracts rather than concrete
model providers.

### Reason

Model integrations can change without rewriting application logic.

---

## ADR-003 — Canonical request/response contracts

### Decision

Use our own request/response classes for chat, embeddings, and vector search.

### Reason

Provider SDK request/response models must not leak into the application.

This creates an explicit mapping boundary.

---

## ADR-004 — Responses API for initial OpenAI chat adapter

### Decision

The first OpenAI chat implementation uses the Responses API.

### Reason

This is the current API direction for new OpenAI integrations. The provider-specific choice remains behind `ResponseModel`.

OpenAI's migration guidance states that the Assistants API was sunset on August 26, 2026 and directs new integrations to Responses. See the official migration guidance:
https://platform.openai.com/docs/guides/migrate-to-responses

### Consequence

The class is now named `OpenAIResponseModel` and implements `ResponseModel`.
The rest of the application does not know that the Responses API is being used.

---

## ADR-005 — No generic mega-provider interface

### Decision

Do not create one `AIProvider` interface containing chat, embeddings, vision, speech, tools, and other capabilities.

### Reason

Different capabilities evolve independently. Small interfaces follow interface segregation and make testing easier.

---

## ADR-006 — Azure AI Search as initial vector store

### Decision

Azure AI Search is the chosen concrete vector-store platform (see ADR-011: it is
the only one).

### Reason

It is the selected learning stack for the current RAG implementation.

### Consequence

Azure SDK types remain inside infrastructure. No generic `VectorStore` interface
exists.

---

## ADR-007 — No RAG framework initially

### Decision

Do not use LangChain, LangGraph, or LlamaIndex in the initial implementation.

### Reason

The purpose is to understand the mechanics of:

- embedding
- retrieval
- context assembly
- prompt construction
- generation
- evidence/citations
- evaluation

Frameworks can be introduced after these mechanics are understood.

---

## ADR-008 — No provider-specific parameters in generic contracts unless justified

### Decision

Only add a parameter to a canonical contract when it represents a real application-level capability.

### Reason

Provider-specific parameter leakage makes an abstraction misleading.

Provider-specific features stay in adapters until the application genuinely needs them.

---

## ADR-009 — Azure authentication through Microsoft Entra ID

### Decision

Azure services use Microsoft Entra ID authentication through Azure Identity's
`DefaultAzureCredential`. Azure API keys are not part of this project's
authentication approach.

### Reason

This keeps credentials out of configuration and source code while allowing
local development and deployed workloads to use the appropriate Azure identity.

### Consequence

Configuration contains service endpoints and deployment/index names, not
credentials. Infrastructure or composition supplies `DefaultAzureCredential`
to Azure SDK clients. The domain layer remains independent of Azure Identity.

---

## ADR-010 — Separate vector document, runtime store, and provisioning

### Decision

Keep vector document data, runtime storage operations, and index provisioning
as separate responsibilities:

- `VectorDocument` represents data to be stored.
- Azure AI Search is the sole platform for runtime storage operations.
- Azure Search index definitions belong to infrastructure/provisioning and
  use Azure AI Search SDK types directly.

Provisioning the index is a separate infrastructure responsibility
(`index_provisioner.py`). It creates or updates the index; the runtime
store uses the provisioned index and does not create it. Azure SDK types stay
out of domain/application contracts.

### Reason

Separating document data, runtime operations, and provisioning keeps index
lifecycle work out of runtime application behavior.

### Consequence

Implementation and tests treat document data, runtime operations, and index
provisioning as distinct boundaries.

---

## ADR-011 — Azure AI Search storage and Azure AI Foundry model platform

### Status

Accepted. This decision supersedes the earlier provider-neutral/multi-provider
vector-store direction (ADR-002, ADR-006), the universal-schema implication in
ADR-010, and any implication that OpenAI is the required model provider. Earlier
ADRs remain as historical records; ADR-010's separation of document data, schema,
and runtime operations remains in force.

### Decision

- Azure AI Search is the sole vector store. The runtime implementation is
  `AzureAISearchVectorStore`; no generic provider-switching `VectorStore`
  protocol, alternative database adapters, or vector-store registry is needed.
  - `VectorDocument` remains a domain data base model that supports concrete
    subclasses with strongly typed metadata.
  - Azure AI Search index definitions use SDK types directly in
    infrastructure/provisioning. There is no separate domain schema model.
  - A separate setup component creates or updates the physical index from an
    Azure AI Search SDK index definition. Runtime storage operations use an
    existing index and never provision it.
- Azure AI Foundry is the model platform. The application uses small
  `EmbeddingModel` and `ResponseModel` capabilities. Model endpoint and
  model/deployment selection are configuration-driven.
- Concrete model integrations use the Foundry API supported by the selected
  model. `/openai/v1` is one supported interface, not the universal API.
  Model-specific mapping remains in infrastructure; no model-provider
  registry or universal SDK hierarchy is introduced.
- `OpenAIResponseModel` is the first concrete `ResponseModel` integration. It
  uses the OpenAI Python SDK with the OpenAI-compatible Foundry Responses API;
  future integrations may use different APIs and SDKs.
- Azure integrations use Microsoft Entra ID through the shared
  `DefaultAzureCredential` infrastructure. API keys are not used.

### Reason

Azure AI Search is the fixed storage choice, while Azure AI Foundry exposes
models through different API surfaces. Selective capability abstractions keep
the RAG application independent of those model-specific details without
pretending that all Foundry models share one inference API.

### Consequence

Domain/application models remain free of Azure SDK and model-provider client
types. Concrete model integrations and the sole Azure Search implementation
live in infrastructure. Endpoint/model settings are separate for response
and embedding roles. The OpenAI SDK is included only because the first
concrete response integration uses the OpenAI-compatible Foundry API.

---

## ADR-012 — rag-core is generic; `/api` is an example application

### Decision

rag-core (domain, application, infrastructure) is a generic RAG engine with no
dependency on any business domain. The `/api` layer is an example application that
uses it for a candidate-document use case. The candidate-specific metadata fields
and the Azure AI Search index that stores them are that example's concrete
representation and may evolve with the consuming system's needs.

### Reason

Keeps the reusable engine free of candidate or BGV assumptions (see ADR-001) while
still demonstrating a realistic consumer.

### Consequence

Changing the example's metadata requires changing the metadata model and the index
together, not rag-core's contracts, unless a core contract is actually affected.
Document ingestion is implemented directly in the `POST /documents` route rather
than as a separate application service.

---

## ADR-013 — Selectable search method without a strategy abstraction

### Decision

`retrieve(method=...)` chooses among `vector_search`, `keyword_search`,
`hybrid_search`, and `semantic_search` on the concrete `AzureAISearchVectorStore`.
The default is `vector`. Only vector and hybrid embed the query.

### Reason

The methods are the real operations Azure AI Search offers; a strategy interface or
factory would add indirection without a second implementation.

### Consequence

The API and application keep separate but identical method Literals so the
application does not import the API layer.
