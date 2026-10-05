# Project Setup

## 1. Repository

The repository is standalone:

```text
rag-core/
```

It is not part of the BGV project.

## 2. Python

Use Python 3.12 or newer.

Check:

```powershell
python --version
```

## 3. Create with uv

From the parent directory:

```powershell
uv init rag-core
cd rag-core
```

If starting from an existing empty repository, initialize it appropriately with uv rather than creating a second project layout.

Create/use the environment through uv.

## 4. Core dependencies

The initial project needs:

- pydantic
- pydantic-settings
- openai
- azure-search-documents
- azure-identity

Development dependencies:

- pytest
- pytest-mock or unittest.mock as appropriate
- ruff

Prefer `uv add` / `uv add --dev` rather than manually editing dependency versions.

## 4.1 Azure authentication

Azure services use Microsoft Entra ID authentication through
`DefaultAzureCredential` from `azure-identity`. API keys are not part of this
project's Azure authentication approach.

`DefaultAzureCredential` uses the available Azure Identity environment,
development, or managed identity credential. Sign in with an appropriate
development identity or configure an Azure-hosted identity with the required
service permissions. Azure SDK clients must receive this credential when they
are added; credentials do not belong in application or domain code.

## 5. Source layout

Use a `src` layout:

```text
src/
└── rag_core/
```

Tests:

```text
tests/
```

## 6. Environment

Create:

```text
.env
.env.example
```

`.env` must be gitignored.

`.env.example` contains Azure service endpoints and deployment/index names,
with no secrets or API-key settings. Azure authentication is provided by
Microsoft Entra ID through `DefaultAzureCredential`.

## 6.1 Azure AI Search index setup

Index provisioning is a setup/operations task, separate from runtime
`VectorStore` behavior. A future script such as
`scripts/setup_vector_index.py` will apply a concrete `VectorStoreSchema` to
create or configure the Azure AI Search index. The runtime
`AzureAISearchVectorStore` will use the provisioned index and will not create
it. Azure SDK clients used by the setup script or runtime adapter must receive
`DefaultAzureCredential`; neither path uses API keys.

## 7. Initial configuration categories

Configuration will eventually include:

```text
LLM provider
LLM model
Embedding provider
Embedding model
Vector-store provider
Vector-store/index configuration
```

Provider-specific settings must remain grouped and clearly named.

## 8. Validation

After setup, run:

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The repository should remain green after each implementation step.
