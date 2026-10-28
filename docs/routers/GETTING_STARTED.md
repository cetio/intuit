# Getting Started with Routers

Inference-capable routers keep conversation history and use one active model. This guide uses OpenRouter. See [Quick Start](../QUICK_START.md) for installation and [Routers](README.md) for the capability comparison.

## Creating a Router

Create an OpenRouter account and supply its API key.

```d
import intuit;

auto router = new OpenRouter("your-openrouter-key");
```

The public host is used by default. Provider-specific headers, routes, and other options are covered in [OpenRouter](OPENROUTER.md).

## Selecting a Model

Set the active model before making requests. Replace the ID with a model available to your account.

```d
router.active("openai/gpt-4o");
```

Activation may fetch the catalog. It preserves existing history and can update the context compactor's token limit from model metadata.

## Completions

Each prompt is appended to the router's context. The request sends that history, and the first assistant choice is appended to it.

```d
import std.stdio : writeln;

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

Both overloads throw a plain `Exception` when no model is active. Clear history with `router.context.clear()`. Switching models doesn't clear it automatically.

## Configuring Requests

```d
ModelConfig cfg = router.config();
cfg.temperature = 0.4;
cfg.maxTokens = 1024;
```

`config()` returns the active model's cached configuration. `config(modelName)` gets or creates a configuration without activating that model. Neither operation is supported by catalog-only routers.

## Next Steps

- [Model Catalogs](CATALOG.md) for metadata and discovery.
- [Models](../MODELS.md) for request settings and structured output.
- [Context](../CONTEXT.md) for history management and compaction.
- [Responses](../RESPONSES.md) for completion fields and embedding requests.
- [Tools](../TOOLS.md) and [Decisions](../DECISIONS.md) for those workflows.
- [Router guides](README.md#implementations) for options and limitations.
