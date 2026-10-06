# Roadmap

This repository learns Vanilla RAG mechanics before adding orchestration
frameworks. The current design is in [ARCHITECTURE.md](ARCHITECTURE.md) and current
status in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

## Progression

```text
Typed documents and metadata
   ↓
Azure AI Search index and runtime store
   ↓
Embeddings and retrieval (vector, keyword, hybrid, semantic)
   ↓
Grounded response generation
   ↓
Example FastAPI application        <- current
   ↓
Real Azure smoke test              <- next
   ↓
Evaluation
   ↓
Hardening and optional extensions
```

## Future work

- End-to-end smoke test against real Azure resources.
- Evaluation of retrieval relevance, groundedness, citations, and
  insufficient-evidence behavior, evaluating retrieval separately from generation.
- Hardening: observability, retries, timeouts, HTTP error mapping, and a
  re-ingestion/cleanup policy.
- Optional: additional Foundry model integrations, OCR and document storage for the
  example application, and a decision on the default search method.
- Later, as a separate concern, integrating rag-core into a larger system such as
  the BGV capstone.

## Phase boundary

Do not turn this repository into an agent framework. LangChain, LangGraph, agents,
and other orchestration are not prerequisites for the Vanilla RAG flow.