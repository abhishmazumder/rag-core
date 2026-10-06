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

Ask for one focused change at a time, for example one capability or one concrete
integration. See `AGENTS.md` and [ARCHITECTURE.md](ARCHITECTURE.md) for the
constraints.

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
6. Is Azure AI Search still the sole vector store, with no generic `VectorStore`?
7. Did the change implement more than the current milestone?
8. Are tests present?
9. Is configuration separate from domain models, and is candidate-specific
   logic kept out of the generic core?
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
