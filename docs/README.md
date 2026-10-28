# Documentation

Start with [Quick Start](QUICK_START.md) to install Intuit and make a conversation request.

## Clients

Use an endpoint to name a model with each request and keep history yourself. Use an inference-capable router to select an active model and let it keep history. Catalog-only routers supply metadata, not inference.

- [Endpoints](endpoints/README.md) compares OpenAI, Anthropic, Qwen, and TypeSafe.
- [Routers](routers/README.md) compares OpenRouter, Zen, LiteLLM, and Models.dev.
- [Model Catalogs](routers/CATALOG.md) explains discovery, capabilities, limits, and prices.

## Shared Guides

| Guide | What you'll find |
| --- | --- |
| [Models](MODELS.md) | Cached settings, sampling parameters, structured output, and tool choice. |
| [Context](CONTEXT.md) | Conversation messages, history mutation, and compaction. |
| [Responses](RESPONSES.md) | Completion choices, usage, reasoning, and single/batch embeddings. |
| [Tools](TOOLS.md) | Function registration, manual execution, policy decisions, and round limits. |
| [Decisions](DECISIONS.md) | Predicate, choice, and score requests and typed answers. |
| [Errors](ERRORS.md) | Request and parsing exceptions, timeouts, and retry considerations. |

## Development

- [Extending Intuit](EXTENDING.md) covers custom configurations, endpoints, and routers.
- [Testing](https://github.com/cetio/intuit/blob/main/TESTING.md) covers the library build, offline tests, and integration prerequisites.
- [Contributing](https://github.com/cetio/intuit/blob/main/CONTRIBUTING.md) covers focused changes and documentation conventions.
