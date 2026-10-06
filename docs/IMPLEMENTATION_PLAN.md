# Implementation Plan

Current implementation sequence and status. Work in small, verifiable increments and
do not implement future phases early. Design lives in
[ARCHITECTURE.md](ARCHITECTURE.md); future work in [ROADMAP.md](ROADMAP.md).

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Project foundation: uv project, pytest, Ruff, `.env.example`, README | Done |
| 1 | Typed settings and shared `DefaultAzureCredential` factory (no API keys) | Done |
| 2 | Domain models and capabilities: `VectorDocument`, search models, `EmbeddingModel`, `ResponseModel`, `RAGResponse` | Done |
| 3 | Foundry response integration (`OpenAIResponseModel`) | Done |
| 4 | Foundry embedding integration (`OpenAIEmbeddingModel`) | Done |
| 5 | Azure AI Search index provisioning (`index_provisioner.py`) | Done |
| 6 | `AzureAISearchVectorStore`: upsert, vector/keyword/hybrid/semantic search, get/delete/clear | Done |
| 7 | Retrieval with selectable search method (`retrieve`) | Done |
| 8 | RAG generation: grounded evidence prompt and `RAGResponse` | Done |
| 9 | Chunking (`application/chunking.py`) | Done |
| 10 | Example FastAPI application: ingestion, query, index admin, deletion | Done |
| 11 | Real Azure smoke test of the end-to-end flow | Next |
| 12 | Evaluation | Planned |
| 13 | Hardening and optional extensions | Planned |

## Next: real Azure smoke test

Run the API flow against a provisioned index and configured Foundry deployments:
set up the index, ingest text, and query with each search method. This is an
explicit manual check; normal pytest stays offline.

## Planned: evaluation

Measure retrieval relevance, evidence coverage, groundedness, citation correctness,
and insufficient-evidence behavior, separately for retrieval and generation.

## Planned: hardening

Observability, retries and timeouts, mapping application errors (for example an
embedding-count mismatch) to HTTP errors, and a re-ingestion/cleanup policy.
Additional Foundry model integrations are optional. Do not add other vector
databases or agent frameworks.