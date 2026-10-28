# OpenRouter

`OpenRouter` supports model discovery, completions, embeddings, and System One decisions. It owns conversation history and uses one active model for requests.

## Connection

```d
import intuit;

auto router = new OpenRouter("your-openrouter-key");
router.active("openai/gpt-4o");
```

The constructor accepts `key`, `url` (default `https://openrouter.ai`), and display `name` (default `"OpenRouter"`). A nonempty key uses bearer authentication. Custom deployments must expose the same routes.

| Operation | Path |
| --- | --- |
| Catalog | `GET /api/v1/models?output_modalities=all` |
| Completions | `POST /api/v1/chat/completions` |
| Embeddings | `POST /api/v1/embeddings` |
| Decisions | `POST /api/alpha/decisions` |

The base URL is used as supplied. Don't include `/api/v1` or a trailing slash for the public host.

## Models and History

Catalog IDs preserve OpenRouter slugs, such as `openai/gpt-4o`. Catalog access is lazy, and activation fetches metadata if the cache is empty. Aliases absent from the catalog are accepted.

Activation preserves messages. If the model is in the catalog, it sets `router.context.compactor.maxTokens` to the reported context length. An absent model leaves the previous limit unchanged.

Every configuration is a `SystemOneModelConfig`. Its inherited methods build ordinary OpenAI-compatible completion and embedding payloads, while its decision methods translate System One requests. The router doesn't choose `ClaudeModelConfig` or `QwenModelConfig` from a model's provider prefix.

See [Getting Started](README.md) for conversation requests and [Models](../MODELS.md) for common settings. Embedding and decision requests use the active model without changing history.

## Request Options

| Field | Destination | Purpose |
| --- | --- | --- |
| `referer` | `HTTP-Referer` header | Calling application URL. |
| `title` | `X-OpenRouter-Title` header | Application title. |
| `categories` | `X-OpenRouter-Categories` header | Categories joined with commas. |
| `route` | Completion body | Routing strategy, such as `"fallback"`. |
| `transforms` | Completion body | Message transforms, such as `"middle-out"`. |
| `includeReasoning` | Completion body | Sets `reasoning.exclude` to `false`, retaining other reasoning fields. |
| `provider` | Completion, embedding, and decision bodies | Provider preferences. |
| `plugins` | Completion body | Plugin array. |

```d
import std.json : JSONValue;

router.referer = "https://example.com";
router.title = "Example App";
router.categories = ["productivity"];
router.includeReasoning = true;
router.provider = JSONValue.emptyObject;
router.provider["allow_fallbacks"] = JSONValue(true);
```

Nonempty router options are added after model payload construction, so they can override matching model parameters. Intuit passes provider and plugin JSON through without validating the remote option schema.

## Catalog Mapping

Metadata includes descriptions, context/output limits, architecture modalities, and advertised parameters. Input/output prices are converted from USD per token to USD per million tokens. Detailed pricing tiers and other provider metadata aren't represented by `ModelDetails`.

Catalog entries can describe operations a particular account or deployment doesn't support. For decisions, choose a System One-compatible model ID and follow [Decisions](../DECISIONS.md). Streaming isn't implemented.