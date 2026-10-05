# RAG Core

A provider-neutral Vanilla RAG implementation built in Python.

The project exists to learn RAG engineering from first principles before introducing frameworks such as LangChain and LangGraph.

## Initial stack

- Python 3.12+
- uv
- Pydantic
- pytest
- Ruff
- OpenAI Responses API
- OpenAI embeddings
- Azure AI Search

The architecture is provider-neutral even though the first concrete implementations use OpenAI and Azure AI Search.

## Architectural objective

Application code should depend on capabilities rather than vendors:

```text
ChatModel
EmbeddingModel
VectorStore
```

Concrete infrastructure adapters implement those capabilities:

```text
OpenAIResponsesChatModel
OpenAIEmbeddingModel
AzureAISearchVectorStore
```

This allows future providers to be added without rewriting RAG application logic.

### Vector storage responsibilities

Vector storage keeps three concepts separate:

- `VectorDocument` represents the data to store.
- `VectorStoreSchema` describes the expected storage/index structure.
- `VectorStore` defines provider-neutral runtime storage operations.

They are not inheritance relationships. A separate setup script,
`scripts/setup_vector_index.py`, will apply a concrete schema to provision an
index. The runtime store uses that index; it does not create it. The Azure AI
Search adapter owns provider SDK mappings, while application/domain contracts
remain provider-neutral.

## Documentation

Read these in order:

1. `AGENTS.md` — repository-wide engineering rules
2. `docs/ARCHITECTURE.md` — LLD and dependency boundaries
3. `docs/CONTRACTS.md` — request/response contracts
4. `docs/SETUP.md` — project setup with uv
5. `docs/IMPLEMENTATION_PLAN.md` — staged implementation plan
6. `docs/TESTING.md` — testing strategy
7. `docs/DECISIONS.md` — important architectural decisions

## Current RAG flow

```text
User Query
    |
    v
Retrieval Request
    |
    v
EmbeddingModel
    |
    v
VectorStore
    |
    v
Evidence
    |
    v
ContextBuilder
    |
    v
PromptBuilder
    |
    v
ChatModel
    |
    v
RAGResponse
```

## What this project deliberately does not use

- LangChain
- LangGraph
- LlamaIndex
- Qdrant
- Pinecone
- provider-specific objects outside infrastructure
- large framework abstractions

Those may be studied later. The purpose here is to understand what those frameworks are abstracting.
