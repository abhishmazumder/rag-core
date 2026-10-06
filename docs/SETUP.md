# Project Setup

How to configure and run the project. For the conceptual design see
[ARCHITECTURE.md](ARCHITECTURE.md).

## 1. Python and dependencies

Use Python 3.12 or newer and `uv`.

```powershell
uv sync
```

Runtime dependencies: `pydantic`, `pydantic-settings`, `azure-identity`,
`azure-search-documents`, `openai` (used by the first response and embedding
integrations), and `fastapi`. Development tools: `pytest`, `pytest-mock`, `httpx`
(for API tests), and `ruff`. Prefer `uv add` / `uv add --dev` for changes.

## 2. Azure resources

You need:

- an Azure AI Search service
- an Azure AI Foundry resource with an embedding deployment and a response
  deployment

## 3. Environment configuration

Copy `.env.example` to `.env` (gitignored). It holds only endpoints and names, never
credentials:

| Setting | Meaning |
| --- | --- |
| `AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_INDEX_NAME` | Azure AI Search service and index |
| `AZURE_AI_FOUNDRY_RESPONSE_ENDPOINT`, `AZURE_AI_FOUNDRY_RESPONSE_MODEL` | response model endpoint and deployment name |
| `AZURE_AI_FOUNDRY_EMBEDDING_ENDPOINT`, `AZURE_AI_FOUNDRY_EMBEDDING_MODEL` | embedding model endpoint and deployment name |
| `AZURE_AI_FOUNDRY_EMBEDDING_DIMENSIONS` | vector length returned by the embedding deployment |

Response and embedding endpoints are separate because deployments can expose
different API surfaces. For the current OpenAI-compatible integrations, use the full
`/openai/v1/` URL and the deployment name. Set the dimensions from authoritative
information for your deployment (the example value `1536` matches
`text-embedding-3-small`); it must equal the real output length because it defines
the index vector field.

## 4. Authentication

All Azure services use Microsoft Entra ID through `DefaultAzureCredential`. Locally,
sign in with the Azure CLI (`az login`); deployed workloads can use a managed
identity. Grant the identity only the roles it needs on the Search and Foundry
resources (including permission to create indexes for index setup). No API keys are
used or supported.

## 5. Index setup

Index provisioning is explicit and never happens at startup. With the API running,
call `POST /admin/index/setup` to create or update the canonical index named by
`AZURE_SEARCH_INDEX_NAME`, using `AZURE_AI_FOUNDRY_EMBEDDING_DIMENSIONS`. This index
is the representation used by the example candidate-document API. If you change the
index fields, change the metadata model to match.

## 6. Running the API

The example API is the FastAPI app produced by `rag_core.api.app.create_app`. Serve
it with an ASGI server; `uvicorn` is not currently a project dependency, so add it
(`uv add uvicorn`) if you want to run the API locally:

```powershell
uv run uvicorn rag_core.api.app:create_app --factory --reload
```

Endpoints are listed in ARCHITECTURE.md. Missing configuration yields a 503 from the
affected endpoint.

## 7. Validation

```powershell
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

See [TESTING.md](TESTING.md). Unit tests need no credentials or network access.