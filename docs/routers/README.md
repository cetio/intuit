# Routers

An `IRouter` presents models from one or more backing providers through a single stateful interface. A router owns a `Context`, a tool registry, a model catalog, and one active model.

- [Getting Started](GETTING_STARTED.md) — discover models, select an active model, configure requests, and use the maintained context.

Built-in routers:

| Class | Model catalog | Requests |
| --- | --- | --- |
| `OpenRouter` | Dynamic catalog from `/api/v1/models`. | Completions and embeddings. |
| `LiteLLM` | Dynamic catalog from `/v1/model/info`. | Not yet implemented. |
| `ModelsDev` | Public catalog from `/api.json?type=all`, including decision models. | Catalog only; not an inference provider. |

`OpenRouter` can make requests through the public OpenRouter API or a compatible deployment. `LiteLLM` currently exposes catalog metadata only; active-model selection, model configuration, completions, and embeddings throw.

`ModelsDev` needs no API key. Construct it with `new ModelsDev()` and access `router.catalog` to fetch and cache the catalog; `router.refresh()` explicitly reloads it.

Catalog IDs use `provider/model` (for example, `openrouter/openai/gpt-5`) to preserve provider-specific limits and prices. Prices use USD per million tokens, consistent across router catalogs. Only fields represented by `ModelDetails` are mapped: names, descriptions, base input/output prices, context/output limits, modalities, and explicitly reported capabilities. Cache/audio/reasoning prices, pricing tiers, lifecycle flags, and provider wiring are not represented.

Active-model selection, model configuration, completions, decisions, and embeddings always throw; models.dev is not an inference provider.

Routers differ from endpoints in two important ways:

- Router request functions use the active model instead of accepting a model name.
- Completion requests always use and update the router's maintained context.

See [Getting Started](GETTING_STARTED.md) for the implemented surface and current limitations.
