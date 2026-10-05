# Learning Roadmap

This repository is the standalone Vanilla RAG implementation used to learn the RAG foundations that precede framework-based agentic development.

## Learning sequence

```text
Embeddings
   ↓
VectorDocument
   ↓
VectorStoreSchema
   ↓
Index provisioning
   ↓
VectorStore runtime operations
   ↓
Metadata filtering
   ↓
Hybrid retrieval
   ↓
Semantic reranking
   ↓
Context assembly
   ↓
Prompt construction
   ↓
LLM generation
   ↓
Grounded answers
   ↓
Citations/evidence
   ↓
Evaluation
```

The project should demonstrate that these mechanisms can be built without LangChain/LangGraph.

## Vector-storage boundaries

Keep the following responsibilities distinct:

- `VectorDocument` is the data to store.
- `VectorStoreSchema` describes the expected index/storage structure.
- `VectorStore` is the provider-neutral runtime storage contract.
- `scripts/setup_vector_index.py` provisions/configures the index separately
  from runtime store operations.

The initial runtime adapter is `AzureAISearchVectorStore`. Azure SDK mapping
belongs in infrastructure; the domain/application contracts remain
provider-neutral.

## Phase boundary

Do not turn this repository into an agent framework.

Later learning phases can introduce:

- LangChain
- LangGraph
- agent state
- tools
- multi-step orchestration

The Vanilla RAG repository should remain understandable on its own.

## Evidence-first principle

Retrieval is not merely a prelude to generation.

We should be able to inspect:

- what was retrieved
- why it was eligible
- metadata/filter scope
- ranking information
- what evidence entered the prompt
- what answer was generated

This makes the system suitable for evaluation and later audit-oriented applications.

## Evaluation principle

A working answer is not sufficient evidence that retrieval is good.

Evaluate retrieval separately from generation.

At minimum, demonstrate:

- relevant evidence is retrieved
- irrelevant evidence is reduced
- scope filters work
- unsupported questions do not receive invented evidence
- citations point to actual retrieved evidence
