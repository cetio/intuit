# Routers

Routers implement `IRouter`, which exposes conversation history, tools, model configurations, and a catalog. Inference-capable routers use one active model instead of accepting a model name with each request. Some implementations currently provide catalog access only.

- [Getting Started](GETTING_STARTED.md) makes a conversation through OpenRouter.
- [Model Catalogs](CATALOG.md) explains discovery, metadata, and pricing units.
- [Models](../MODELS.md) covers common request settings.

## Implementations

| Guide | Class | Inference support |
| --- | --- | --- |
| [OpenRouter](OPENROUTER.md) | `OpenRouter` | Completions, embeddings, and System One decisions. |
| [OpenCode Zen](ZEN.md) | `Zen` | Completions for supported API families. |
| [LiteLLM](LITELLM.md) | `LiteLLM` | Catalog only. |
| [Models.dev](MODELSDEV.md) | `ModelsDev` | Catalog only, without authentication. |

Provider and model support still determines whether a particular request works. Catalog metadata doesn't guarantee inference support, and streaming isn't implemented.

## Conversation History

Completion requests use and update the router's `Context`. Embedding and decision requests use the active model but don't change conversation history.

Changing models preserves history. If you want to keep history yourself and name a model per request, use an [endpoint](../endpoints/README.md) instead.
