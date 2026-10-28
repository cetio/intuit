# Extending Intuit

Use a custom configuration when an existing transport works but a provider needs a different request or response shape. Implement an endpoint or router when the connection, routes, or model-selection behavior also differs.

## Custom Model Configurations

`ModelConfig` combines request settings with virtual payload builders and response parsers.

| Method | Responsibility |
| --- | --- |
| `buildCompletionPayload(input, tools)` | Build a completion request from serialized messages or content. |
| `parseCompletionResponse(json)` | Return a shared `Completion`. |
| `buildEmbeddingPayload(input)` | Build an embedding request. |
| `parseEmbeddingResponse(json)` | Return a JSON array of vectors, not the provider's response envelope. |
| `buildDecisionPayload(input, questions)` | Build a decision request. |
| `parseDecisionResponse(json, questions)` | Return a shared `Decision`, validating it against the questions. |

This configuration adds a completion setting while retaining the base serializer.

```d
import intuit;

import std.json : JSONValue;

class LocalModelConfig : ModelConfig
{
    this(string name)
    {
        super(name);
    }

    override JSONValue buildCompletionPayload(
        JSONValue input,
        ToolRegistry tools = ToolRegistry.init,
    )
    {
        JSONValue payload = super.buildCompletionPayload(input, tools);
        payload["keep_alive"] = JSONValue("5m");
        return payload;
    }
}

class LocalEndpoint : OpenAI
{
    this(string url)
    {
        super(url);
    }

    override ModelConfig config(string modelName)
    {
        foreach (cfg; _configs)
            if (cfg.name == modelName)
                return cfg;

        auto cfg = new LocalModelConfig(modelName);
        _configs ~= cfg;
        return cfg;
    }
}
```

Returning the subclass from `config()` makes high-level requests use it. This example is intended for explicitly named models. Override `available()` too if you expose discovery, because the inherited OpenAI implementation creates base `ModelConfig` instances.

For a simple extra field, prefer [ModelConfig.params](MODELS.md#completion-parameters) over a subclass. Provider-specific builders don't all merge it in the same way.

## Custom Endpoints

Implement `IEndpoint` for a provider client. It owns connection settings, a tool registry, and cached model configurations, but not conversation history.

The `_completions`, `_embeddings`, and `_decisions` methods receive a configuration and an already-built payload. They return parsed response JSON. High-level request functions perform model parsing and, for context-backed completions, history mutation and tool-policy execution.

Return a stable configuration instance for each model name. Ensure `available()` creates the same configuration subclass as `config()`. Reject unsupported operations explicitly.

The built-in HTTP helper is `package(intuit)`, so an adapter outside that package must supply its own transport. The public interfaces don't require curl. Use [Errors](ERRORS.md) as the failure contract when matching built-in behavior.

## Custom Routers

Implement `IRouter` when requests use an active model and the router owns conversation history. Supply a `Context`, tool registry, model catalog, and cached configurations.

High-level completion functions append user input and assistant responses. Don't duplicate those mutations in `_completions()`. Embedding and decision functions use the active configuration but don't update the context.

Model activation and catalog loading belong to the router implementation. Preserve history when changing models, and document how activation affects the compactor. Catalog-only implementations should reject inference operations rather than implying that catalog entries are executable.

## Tests

Test builders and parsers with fixed JSON before making live requests. Check provider errors, missing fields, unsupported operations, and preservation of raw responses. For decision adapters, test answer ordering and names, refusals, typed choices, score ranges, and probability conversion.

Use a local HTTP fixture or fake endpoint/router for request-flow tests. Keep provider credentials and live catalog access behind the `integration` version. See [Testing](../TESTING.md).
