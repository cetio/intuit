# Models

A `ModelConfig` stores request parameters and translates completion and embedding payloads and responses. Endpoints cache one config per model name, so changes made through `config(modelName)` apply to later requests for that name.

## Getting a Configuration

```d
import intuit;

auto ep = new OpenAI("http://localhost:1234");
ModelConfig cfg = ep.config("llama3");
```

`config(modelName)` creates a configuration when the endpoint has not seen the name before. `available()` requests `/v1/models`, adds configurations for advertised models, and returns every cached configuration. `configs()` returns the cache without making a request.

The concrete endpoint determines the configuration type:

| Endpoint | Configuration type |
| --- | --- |
| `OpenAI` | `ModelConfig` |
| `Claude` | `ClaudeModelConfig` |
| `Qwen` | `QwenModelConfig` |

```d
Claude claude = new Claude("https://api.anthropic.com", "sk-ant-key");
ClaudeModelConfig cfg = cast(ClaudeModelConfig)claude.config("claude-sonnet-4-5");
```

## Common Parameters

| Field | Purpose | Omitted when |
| --- | --- | --- |
| `name` | Model identifier sent in each request. | Never. |
| `temperature` | Sampling temperature. | `NaN`, the default. |
| `topP` | Nucleus sampling probability. | `NaN`, the default. |
| `maxTokens` | Maximum generated tokens. | Negative, the default. |
| `stop` | Stop sequences for OpenAI-compatible requests. | Empty. |
| `seed` | Deterministic sampling seed. | Zero. |
| `responseSchema` | Complete response-format object. | Null. |
| `toolConfig` | Tool-choice value. | Null. |
| `params` | Additional fields merged into completion and embedding payloads. | Not a JSON object. |

Assign fields directly before making a request:

```d
ModelConfig cfg = ep.config("llama3");
cfg.temperature = 0.7;
cfg.topP = 0.9;
cfg.maxTokens = 1024;
cfg.stop = ["END"];
cfg.seed = 42;

Completion result = completions(ep, cfg.name, "Write a short greeting.");
```

Fields in `params` are merged last and can override fields built by the configuration:

```d
import std.json : JSONValue;

cfg.params = JSONValue.emptyObject;
cfg.params["frequency_penalty"] = JSONValue(0.2);
```

## Structured Output

`setResponseSchema` builds the OpenAI-style `response_format` object from a schema name, JSON Schema object, and strictness flag:

```d
import std.json : JSONValue;

JSONValue schema = JSONValue.emptyObject;
schema["type"] = JSONValue("object");
schema["properties"] = JSONValue.emptyObject;
schema["properties"]["answer"] = JSONValue.emptyObject;
schema["properties"]["answer"]["type"] = JSONValue("string");
schema["required"] = JSONValue([JSONValue("answer")]);

cfg.setResponseSchema("answer_schema", schema);
```

The model and endpoint must support this response-format shape. Intuit sends the schema but does not validate the returned value against it.

Primitive shorthand nodes such as `JSONValue("string")` in schema positions are normalized to `{ "type": "string" }` before the request is sent. Strings in `enum`, `required`, descriptions, and other metadata are preserved.

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

Use `setRequiredTool` to force a specific registered tool:

```d
Tool weather = ep.tools.get("getWeather");
cfg.setRequiredTool(weather);
```

Passing `null` or a tool without a name clears the tool choice. See [Tools](../TOOLS.md) for registration and execution.

## Claude Parameters

`ClaudeModelConfig` translates contexts to Anthropic Messages API payloads and adds these fields:

| Field | Purpose | Omitted when |
| --- | --- | --- |
| `topK` | Top-k sampling value. | Negative. |
| `stopSequences` | Anthropic stop sequences. | Empty. |
| `system` | System prompt prepended to system messages from the context. | Empty. |
| `thinkingBudget` | Enables extended thinking with the given token budget. | Negative. |

```d
ClaudeModelConfig cfg = cast(ClaudeModelConfig)claude.config("claude-sonnet-4-5");
cfg.maxTokens = 2048;
cfg.thinkingBudget = 1024;
cfg.system = "Answer concisely.";
```

System messages in a `Context` are removed from the Messages API message array and joined into the top-level `system` field. Claude tool definitions use `input_schema`, and Claude responses are translated to the shared `Completion` type.

Claude does not support embeddings through this endpoint implementation.

## Qwen Parameters

`QwenModelConfig` extends the OpenAI-compatible payload with Qwen-specific fields:

| Field | Purpose | Default or omission behavior |
| --- | --- | --- |
| `enableThinking` | Enables Qwen thinking. | `true`; `false` sends `enable_thinking: false`. |
| `topK` | Top-k sampling value. | Omitted when negative. |
| `thinkingBudget` | Thinking token budget. | Omitted when negative. |
| `chatTemplateKwargs` | Chat-template keyword arguments. | Omitted when null. |
| `mmProcessorKwargs` | Multimodal processor keyword arguments. | Omitted when null. |
| `encodingFormat` | Embedding encoding format. | `"float"`; omitted at that value. |
| `dimensions` | Requested embedding dimensions. | Omitted when zero. |

```d
Qwen qwen = new Qwen("http://localhost:1234");
QwenModelConfig cfg = cast(QwenModelConfig)qwen.config("qwen3-30b-a3b");
cfg.enableThinking = false;
cfg.topK = 40;
cfg.dimensions = 1024;
```

When a Qwen-compatible server returns tool calls embedded in `<tool_call>` XML instead of native `tool_calls`, `QwenModelConfig` extracts supported Qwen and Hermes-style forms into `Choice.toolCalls` and removes the XML block from `Choice.text`.

## Custom Configurations

Implement a `ModelConfig` subclass when a provider needs a different payload or response shape. Override the relevant methods:

| Method | Purpose |
| --- | --- |
| `buildCompletionPayload(input, tools)` | Build a completion request body. |
| `buildEmbeddingPayload(input)` | Build an embedding request body. |
| `buildDecisionPayload(input, questions)` | Build a decision request body. |
| `parseCompletionResponse(json)` | Translate a completion response to `Completion`. |
| `parseEmbeddingResponse(json)` | Extract embedding arrays from a response. |
| `parseDecisionResponse(json, questions)` | Translate a decision response to `Decision`. |
