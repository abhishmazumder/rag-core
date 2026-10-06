# Testing Strategy

Unit tests are deterministic and run without network access, Azure credentials,
Azure AI Search, or live model calls.

```powershell
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

## 1. Offline unit tests

- **Domain**: validation, defaults, and serialization of the provider-neutral
  models; `VectorDocument` with different metadata types; capability contracts.
  Domain tests import no Azure or model-provider SDK types.
- **Infrastructure**: mock SDK clients to verify mapping boundaries: index
  definition and provisioning, `AzureAISearchVectorStore` document and search
  mapping, each Foundry integration's request/response mapping, and Azure Identity
  setup. SDK objects must not escape infrastructure.
- **Application**: use fakes or mocks for `EmbeddingModel`, `ResponseModel`, and
  the store. Cover retrieval for every search method (including that embedding is
  called only for vector and hybrid), chunking, and RAG generation including
  insufficient-evidence behavior.
- **Composition and configuration**: settings load without credentials; factories
  validate configuration.
- **Example API**: FastAPI `TestClient` with dependency overrides. Cover request
  validation, response shapes, routing, ingestion behavior in `POST /documents`, and
  that `/query` forwards `method`. No Azure service is contacted.

Use `Settings(_env_file=None)` in tests that must not read a local `.env`.

## 2. Azure-backed validation

Real Azure checks are manual and opt-in, separate from the unit suite. They need
Azure Identity sign-in, network access, a development Search service, and
configured Foundry deployments. The next planned one is the end-to-end smoke test in
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). No automated integration suite
exists yet.

## 3. Evaluation

Evaluate retrieval relevance and scope isolation separately from generation. For
RAG, verify grounded answers, evidence references, and insufficient-evidence
behavior. This is planned work.

## 4. Regression rule

When a bug is fixed, add a regression test with the fix.