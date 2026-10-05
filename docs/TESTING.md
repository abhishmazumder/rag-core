# Testing Strategy

## Testing philosophy

The application layer should be testable without network access or provider credentials.

## 1. Domain tests

Test:

- validation
- serialization
- defaults
- invalid values

These tests must not call external services.

## 2. Adapter unit tests

Mock provider SDK clients.

For example, `OpenAIResponsesChatModel` tests should verify:

1. canonical `ChatRequest` is translated correctly
2. provider SDK is called correctly
3. provider response is mapped correctly
4. provider exceptions are translated appropriately

Do not make these tests depend on real API credentials.

For vector storage, test the provider-neutral document/schema behavior and
Azure adapter mappings separately. Mock Azure SDK clients in adapter tests.
Test index provisioning independently from runtime store operations; ordinary
unit tests must not create an index or require Azure credentials.

## 3. Application tests

Use fakes:

```text
FakeChatModel
FakeEmbeddingModel
FakeVectorStore
```

Example:

```text
RAGPipeline
    |
    +-- FakeRetriever
    +-- FakeChatModel
```

This allows deterministic tests.

## 4. Integration tests

Keep real-provider tests separate.

They may require:

- credentials
- network
- development Azure AI Search index
- OpenAI access

Mark or organize them so normal unit-test execution does not require cloud access.
For Azure AI Search, an explicit integration test may provision a clean
development index using the setup script before exercising the runtime store.

## 5. Retrieval evaluation

Evaluation is different from unit testing.

A retrieval evaluation dataset should contain:

- query
- expected relevant evidence
- expected source/chunk
- optional acceptable alternatives

Measure retrieval behavior rather than relying only on manual inspection.

## 6. RAG evaluation

Test:

### Grounded question

Evidence contains the answer.

Expected:

- answer uses evidence
- citations/evidence references are present

### Unsupported question

Evidence does not contain enough information.

Expected:

- model is instructed to state insufficiency
- no fabricated evidence

### Scope isolation

When a retrieval scope is supplied, results outside that scope must not appear.

## 7. Regression rule

When a bug is fixed, add a regression test before or with the fix.
