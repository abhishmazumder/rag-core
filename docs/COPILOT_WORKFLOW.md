# GitHub Copilot Workflow

## How to use Copilot in this repository

Do not ask Copilot to "build the whole RAG system".

Work one milestone at a time.

Before each task:

1. Read `AGENTS.md`.
2. Read the relevant design document.
3. Inspect existing code.
4. Make the smallest change satisfying the task.
5. Add tests.
6. Run tests and Ruff.
7. Review the diff.
8. Update documentation if an architectural decision changed.

## Preferred prompt style

Use prompts like:

> Read AGENTS.md and docs/CONTRACTS.md. Implement only the `ChatMessage`, `ChatOptions`, `ChatRequest`, and `ChatResponse` Pydantic models. Do not implement the provider adapter, factory, pipeline, or future phases. Add unit tests. Run Ruff and pytest.

Then separately:

> Read AGENTS.md and docs/CONTRACTS.md. Implement the `ChatModel` protocol only. Do not add provider-specific code.

Then:

> Read AGENTS.md, docs/ARCHITECTURE.md, and docs/CONTRACTS.md. Implement `OpenAIResponsesChatModel` as an adapter around the OpenAI Responses API. Keep all OpenAI SDK imports inside infrastructure. Add mocked unit tests.

## Anti-pattern prompts

Avoid:

> Build a complete enterprise RAG framework.

Avoid:

> Make everything generic and support all providers.

Avoid:

> Create an abstract base class for every class.

These prompts encourage premature abstraction.

## Review questions

After Copilot changes code, ask:

1. Did provider SDK imports appear outside infrastructure?
2. Did an SDK response type leak into the application?
3. Did we add a provider-specific field to a generic contract?
4. Did we add unnecessary inheritance?
5. Can the application be tested with fakes?
6. Can a second provider implement the interface without changing the application?
7. Did the change implement more than the current milestone?
8. Are tests present?
9. Is configuration separate from domain models?
10. Did any secret enter source control?

## Definition of done

A task is not complete merely because the code runs.

It is complete when:

- implementation matches the relevant contract
- tests exist
- tests pass
- Ruff passes
- provider boundaries are respected
- documentation remains accurate
