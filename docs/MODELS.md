# Models

A `ModelConfig` stores request settings and translates provider payloads and responses. Endpoints and inference-capable routers cache configurations by model name. Changes apply to later requests for that name.

## Getting a Configuration

```d
import intuit;

auto ep = new OpenAI("http://localhost:1234");
ModelConfig cfg = ep.config("llama3");
```

`ep.config(modelName)` creates a configuration if the name isn't cached. It doesn't check whether the server offers that model. `ep.configs()` returns cached configurations without making a request. See [Endpoints](endpoints/README.md) for discovery.

On an inference-capable router, `router.config()` returns the active model's configuration. `router.config(modelName)` gets or creates another configuration without activating it. Catalog-only routers don't support either operation.

The concrete endpoint determines the configuration type:

| Endpoint | Configuration type |
| --- | --- |
| `OpenAI` | `ModelConfig` |
| `Claude` | `ClaudeModelConfig` |
| `Qwen` | `QwenModelConfig` |
| `TypeSafe` | `SystemOneModelConfig` |

OpenRouter also uses `SystemOneModelConfig`, which inherits the common completion and embedding behavior. Zen selects `ModelConfig` or `ClaudeModelConfig` by model ID. Provider guides explain the differences.

## Completion Parameters

These omission rules describe the base OpenAI-compatible payload builder. [Claude](endpoints/ANTHROPIC.md) and [Qwen](endpoints/QWEN.md) have additional rules.

| Field | Purpose | Omitted when |
| --- | --- | --- |
| `name` | Model identifier sent in each request. | Never. |
| `temperature` | Sampling temperature. | `NaN`, the default. |
| `topP` | Nucleus sampling probability. | `NaN`, the default. |
| `maxTokens` | Maximum generated tokens. | Negative, the default. |
| `stop` | Stop sequences for OpenAI-compatible requests. | Empty. |
| `seed` | Sampling seed, when supported by the server. | Zero or negative. |
| `responseSchema` | Complete response-format object. | Null. |
| `toolConfig` | Tool-choice value. | Null. |
| `params` | Additional provider request fields. | Not a JSON object. |

Assign fields directly before making a request:

```d
cfg.temperature = 0.7;
cfg.topP = 0.9;
cfg.maxTokens = 1024;
cfg.stop = ["END"];
cfg.seed = 42;

Completion result = completions(ep, cfg.name, "Write a short greeting.");
```

The base completion and embedding builders merge `params` last, so it can override generated fields:

```d
import std.json : JSONValue;

cfg.params = JSONValue.emptyObject;
cfg.params["frequency_penalty"] = JSONValue(0.2);
```

Qwen adds its own fields after the base builder runs. Claude's completion builder doesn't merge `params`. Decision builders accept `params` but write their required model, input/state, and question fields afterward. Common sampling and completion limits aren't added to decision requests.

## Structured Output

`setResponseSchema` builds the OpenAI-style `response_format` object from a schema name, JSON Schema object, and strictness flag (default `true`):

```d
import std.json : JSONValue;

JSONValue schema = JSONValue.emptyObject;
schema["type"] = JSONValue("object");
schema["properties"] = JSONValue.emptyObject;
schema["properties"]["answer"] = JSONValue.emptyObject;
schema["properties"]["answer"]["type"] = JSONValue("string");
schema["required"] = JSONValue([JSONValue("answer")]);
schema["additionalProperties"] = JSONValue(false);

cfg.setResponseSchema("answer_schema", schema);
```

The model and server must support structured output. The base builder sends `response_format`, while Claude translates the schema to `output_config.format`. Intuit doesn't validate the returned value against the schema.

Primitive shorthand nodes such as `JSONValue("string")` in schema positions are normalized to `{ "type": "string" }` before the request is sent. Strings in `enum`, `required`, descriptions, and other metadata are preserved.

Read a JSON answer with `result.json`. Invalid JSON text throws `MalformedResponseException`. Valid JSON can still violate the requested schema.

## Tool Choice

Use `setToolLiability` to control whether the model may call tools:

```d
cfg.setToolLiability("auto");
```

Accepted values are:

| Value | Behavior |
| --- | --- |
| `"auto"` | The model may call zero or more tools. |
| `"required"` | The model must call one or more tools. |
| `"none"` | The model cannot call tools. |

Any other value throws `FormatException`.

After [registering a tool](TOOLS.md#registering-a-function), use `setRequiredTool` to request that specific tool:

```d
cfg.setRequiredTool(ep.tools.get("getWeather"));
```

Passing `null` or a tool without a name clears the tool choice. These settings ask the model to produce tool calls. They don't authorize execution in your application. See [Tools](TOOLS.md) for registration, execution policy, and manual handling.

## Provider Settings

- [Anthropic](endpoints/ANTHROPIC.md) covers system prompts, thinking budgets, and Messages API translation.
- [Qwen](endpoints/QWEN.md) covers thinking controls, embedding settings, and XML tool calls.
- [TypeSafe](endpoints/TYPESAFE.md) and [OpenRouter](routers/OPENROUTER.md) cover System One decisions.
- [Extending Intuit](EXTENDING.md) covers custom payload builders and parsers.
