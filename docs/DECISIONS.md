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

Application code depends on:

- `ChatModel`
- `EmbeddingModel`
- `VectorStore`

rather than concrete providers.

### Reason

We want to support multiple providers without rewriting application logic.

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

This is the current API direction for new OpenAI integrations. The provider-specific choice remains behind `ChatModel`.

OpenAI's migration guidance states that the Assistants API was sunset on August 26, 2026 and directs new integrations to Responses. See the official migration guidance:
https://platform.openai.com/docs/guides/migrate-to-responses

### Consequence

The class may be named:

`OpenAIResponsesChatModel`

The rest of the application does not know that Responses API is being used.

---

## ADR-005 — No generic mega-provider interface

### Decision

Do not create one `AIProvider` interface containing chat, embeddings, vision, speech, tools, and other capabilities.

### Reason

Different capabilities evolve independently. Small interfaces follow interface segregation and make testing easier.

---

## ADR-006 — Azure AI Search as initial vector store

### Decision

Use Azure AI Search for the first concrete vector-store implementation.

### Reason

It is the selected learning stack for the current RAG implementation.

### Consequence

Azure SDK types remain inside the infrastructure adapter.

The generic `VectorStore` contract does not become an Azure Search API clone.

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

## ADR-010 — Separate vector document, store, and schema responsibilities

### Decision

Keep `VectorDocument`, `VectorStore`, and `VectorStoreSchema` as three separate
concepts:

- `VectorDocument` represents data to be stored.
- `VectorStore` defines the provider-neutral runtime storage contract.
- `VectorStoreSchema` describes the expected storage/index structure.

They do not inherit from one another. A document must be representable by its
intended schema, but the validation and mapping mechanism will be designed
when these components are implemented. No generic document/schema type
relationship will be introduced without a demonstrated need.

Provisioning the index is a separate responsibility of a setup script such as
`scripts/setup_vector_index.py`. The script applies a concrete schema to the
provider; the runtime store uses the provisioned index and does not create it.
For the initial provider, `AzureAISearchVectorStore` owns Azure AI Search
runtime integration and provider-specific mapping, while Azure SDK types stay
out of domain/application contracts.

### Reason

Separating data, runtime storage operations, and storage structure keeps each
responsibility independently understandable and provider-neutral, while
keeping index lifecycle work out of runtime application behavior.

### Consequence

Implementation and tests will treat document data, schema description,
provider-neutral store operations, and index provisioning as distinct
boundaries. This decision does not prescribe exact method signatures or
implementation details.
