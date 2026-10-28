# OpenCode Zen

`Zen` supports completions through OpenCode's Chat Completions and Messages routes. It owns conversation history and selects the payload format from the active model ID. Embeddings and decisions aren't supported.

## Connection

```d
import intuit;

auto router = new Zen("your-zen-key");
ModelDetails[string] catalog = router.catalog();
```

The constructor accepts `key`, `url` (default `https://opencode.ai`), and display `name` (default `"OpenCode Zen"`). A nonempty key uses bearer authentication. Messages requests also set `anthropic-version` to `2023-06-01`.

The base URL is used as supplied. Requests append `/zen/v1/models`, `/zen/v1/chat/completions`, or `/zen/v1/messages`.

## Model Selection

The current implementation uses model ID prefixes, not catalog API-family metadata.

| Model ID | Configuration and route |
| --- | --- |
| Starts with `claude-` | `ClaudeModelConfig`, Messages API. |
| Starts with `qwen`, except `qwen3-coder` | `ClaudeModelConfig`, Messages API. |
| Starts with `gpt-`, `gemini-`, or `jev-` | Rejected when requesting a completion. |
| Other IDs, including `qwen3-coder` | Base `ModelConfig`, Chat Completions API. |

Activation loads the catalog once if needed and accepts IDs absent from it. Unsupported prefixes can therefore be activated but still fail when a completion is requested. A model appearing in the catalog doesn't guarantee that its API family is supported by this adapter.

For a supported model, select its exact ID before making requests.

```d
router.active("your-supported-model-id");
Completion result = completions(router, "Explain associative arrays.");
```

Activation preserves history and updates the compactor limit only when the catalog reports a positive context length. Changing API families doesn't translate existing history into a different message format.

## Configuration and Limitations

Use [Models](../MODELS.md) for base settings and [Anthropic](../endpoints/ANTHROPIC.md) for `ClaudeModelConfig` settings and message-translation limits. The Messages path has the same shared tool-history limitation described there.

The Chat Completions path uses the base parser, not Qwen XML extraction. Zen doesn't implement the Responses API, Gemini-native requests, decisions, embeddings, or streaming.

## Catalog Mapping

The catalog currently supplies only model IDs, names, descriptions, and positive context lengths. Prices, capabilities, modalities, and output limits aren't mapped, even if the service returns them.

See [Model Catalogs](CATALOG.md) for shared metadata and [Getting Started](README.md) for router conversation behavior.