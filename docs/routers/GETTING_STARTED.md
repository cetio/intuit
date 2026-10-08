# Getting Started with Routers

Routers provide model discovery and a stateful completion interface over one or more backing providers. They maintain conversation context and select one active model for requests.

## OpenRouter

Construct `OpenRouter` with an API key. The public OpenRouter host is used by default:

```d
import intuit;

auto router = new OpenRouter("sk-or-key");
```

The complete constructor accepts a base URL and display name:

```d
auto router = new OpenRouter(
    "sk-or-key",
    "https://openrouter.ai",
    "OpenRouter"
);
```

The URL is used as-is. Requests append `/api/v1/models`, `/api/v1/chat/completions`, or `/api/v1/embeddings`.

## Model Catalog

`catalog()` lazily fetches model metadata and caches it by model ID. `refresh()` replaces the cache from the remote catalog:

```d
ModelDetails[string] catalog = router.catalog();
ModelDetails details = catalog["openai/gpt-4o"];

router.refresh();
```

`ModelDetails` exposes:

| Field | Purpose |
| --- | --- |
| `id` | Model slug used for activation and requests. |
| `name` | Human-readable display name. |
| `description` | Provider description. |
| `contextLength` | Total context window in tokens. |
| `maxCompletionTokens` | Maximum completion size reported by the top provider. |
| `inputModalities` | Supported input modalities. |
| `outputModalities` | Supported output modalities. |
| `capabilities` | Normalized OpenAI-compatible parameters and features. |
| `promptCost` | Cost in USD per input token. |
| `completionCost` | Cost in USD per output token. |

Modalities use the `Modality` enum: `text`, `image`, `audio`, `pdf`, and `embedding`. Capabilities use the
`ModelCapability` enum, normalizing the known parameters shared across provider catalogs.

## Selecting a Model

Set the active model before making requests:

```d
router.active("openai/gpt-4o");
writeln(router.active);
```

On first activation, OpenRouter fetches the catalog if it is empty. Aliases absent from the catalog are still accepted. When metadata exists, activation updates `router.context.compactor.maxTokens` to the model's `contextLength` while preserving existing messages.

Every active model has a cached `ModelConfig`:

```d
ModelConfig cfg = router.config();
cfg.temperature = 0.4;
cfg.maxTokens = 1024;

ModelConfig other = router.config("anthropic/claude-sonnet-4");
```

`config()` returns the active model's configuration and throws when no model is active. `config(modelName)` creates or returns a configuration without changing the active model.

## Completions

Supplying data appends a user message to the maintained context, sends the complete context, and appends the assistant response:

```d
import intuit;

import std.stdio : writeln;

auto router = new OpenRouter("sk-or-key");
router.active("openai/gpt-4o");
router.context.system("You are a concise assistant.");

Completion first = completions(router, "Why is the sky blue?");
writeln(first.text);

Completion second = completions(router, "Summarize that in one sentence.");
writeln(second.text);
```

Call `completions(router)` with no data to send the existing context without adding a new user message:

```d
router.context.user("Give me one example.");
Completion result = completions(router);
```

Both overloads throw when no active model is set. See [Context](../CONTEXT.md) for message and compaction behavior.

## Embeddings

Embedding requests use the active model but do not modify the maintained context:

```d
router.active("openai/text-embedding-3-small");

Embedding!float vector = embeddings(router, "Hello, world!");
Embedding!float[] vectors = embeddings(
    router,
    ["Hello, world!", "Goodbye, world!"]
);
```

The backing model and provider must support embeddings.

## OpenRouter Options

OpenRouter-specific request and header options are public fields on the router:

| Field | Destination | Purpose |
| --- | --- | --- |
| `referer` | `HTTP-Referer` header | Identifies the calling application. |
| `title` | `X-OpenRouter-Title` header | Supplies the application title. |
| `categories` | `X-OpenRouter-Categories` header | Comma-separated marketplace categories. |
| `route` | Chat request | Routing strategy such as `"fallback"`. |
| `transforms` | Chat request | Message transforms such as `"middle-out"`. |
| `includeReasoning` | Chat request | Requests reasoning data when true. |
| `provider` | Chat and embedding requests | Provider preference object. |
| `plugins` | Chat requests | OpenRouter plugin array. |

```d
import std.json : JSONValue;

router.referer = "https://example.com";
router.title = "Example App";
router.categories = ["productivity"];
router.transforms = ["middle-out"];
router.includeReasoning = true;

router.provider = JSONValue.emptyObject;
router.provider["allow_fallbacks"] = JSONValue(true);
```

## Tools

The router owns a tool registry separate from endpoint registries:

```d
@Description("Return the current temperature for a city.")
int getTemperature(string city)
{
    return 21;
}

router.tools.add!getTemperature();
router.config().setToolLiability("auto");
```

Registered tools are included in completion payloads. See [Tools](../TOOLS.md) for handling returned tool calls.

## LiteLLM Catalog Access

`LiteLLM` currently supports model-catalog access only:

```d
import intuit;

auto litellm = new LiteLLM("http://localhost:4000", "proxy-key");
ModelDetails[string] catalog = litellm.catalog();
```

It requests `/v1/model/info` and maps LiteLLM metadata to shared `ModelDetails`, including context and output limits, per-token costs, modalities, and supported features.

The following operations are not implemented for `LiteLLM` and throw an exception:

- `active(modelName)`
- `config()` and `config(modelName)`
- `configs()`
- completion requests
- embedding requests

Use an `OpenAI` endpoint against a LiteLLM proxy when OpenAI-compatible completion or embedding requests are needed without router-managed state.

## Current Limitations

Router streaming is not implemented. `OpenRouter` is the only built-in router that currently supports completion and embedding requests. Router errors for an unset active model and unsupported LiteLLM operations are currently plain `Exception` values rather than specialized exception types.
