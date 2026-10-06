# RAG Core

A Vanilla RAG learning implementation in Python, built without RAG orchestration
frameworks (no LangChain, LangGraph, or LlamaIndex).

## What this repository contains

- **rag-core**: a reusable RAG engine built from generic concepts: documents and
  chunks, embeddings, retrieval, search methods, response generation, and
  metadata. It has no dependency on any business domain.
- **Concrete infrastructure**: Azure AI Search as the sole vector store, Azure AI
  Foundry as the model platform, and Microsoft Entra ID (`DefaultAzureCredential`)
  for authentication. There are no API keys.
- **An example `/api` application** (FastAPI): a candidate-document use case
  built on top of rag-core. Its candidate fields, ingestion workflow, and the
  Azure AI Search index that stores them are examples of one consumer, not
  requirements of rag-core.

```text
generic rag-core -> concrete infrastructure -> example /api -> candidate-document example -> its Azure AI Search index
```

## Quick start

See [docs/SETUP.md](docs/SETUP.md) to configure Azure resources, sign in, set up
the index, and run the API. Run the tests with `uv run pytest`.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): the canonical architecture
- [docs/CONTRACTS.md](docs/CONTRACTS.md): stable contracts and data models
- [docs/SETUP.md](docs/SETUP.md): configuration and running
- [docs/TESTING.md](docs/TESTING.md): test strategy
- [docs/DECISIONS.md](docs/DECISIONS.md): decision records
- [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) and
  [docs/ROADMAP.md](docs/ROADMAP.md): status and future work
- [docs/COPILOT_WORKFLOW.md](docs/COPILOT_WORKFLOW.md) and `AGENTS.md`: working
  rules for contributors and Copilot

## What this project deliberately does not use

- RAG frameworks
- Alternative vector databases, a generic `VectorStore` interface, or provider
  switching
- A model-provider registry or a universal model SDK abstraction