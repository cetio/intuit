# Responses

Intuit translates provider-specific completion and embedding responses into shared D types while preserving raw JSON.

## Completion

`completions` returns a `Completion`:

```d
Completion result = completions(ep, "llama3", "Why is the sky blue?");
```

| Field | Purpose |
| --- | --- |
| `raw` | Complete JSON response returned by the endpoint. |
| `choices` | Parsed completion choices. |
| `usage` | Request latency, model name, and token accounting. |

Use `choice(index)` to access a choice by reference. The default index is zero, and an out-of-range index throws:

```d
Choice first = result.choice;
Choice second = result.choice(1);
```

Convenience accessors read common values:

```d
string text = result.text;
string secondText = result.text(1);
string reasoning = result.reasoning;
JSONValue parsed = result.json;
```

`json(index)` parses that choice's text with `std.json.parseJSON` and propagates parsing failures.

## Choice

A `Choice` represents one generated alternative:

| Field | Purpose |
| --- | --- |
| `raw` | Raw JSON for the choice, or the full response for Claude. |
| `content` | Raw OpenAI-compatible message content. |
| `text` | Extracted response text. |
| `reasoning` | Extracted reasoning or thinking text. |
| `finishReason` | Normalized reason generation ended. |
| `logProbs` | Provider log-probability data when returned. |
| `toolCalls` | Tools requested by the model. |

OpenAI-compatible parsing accepts string content and recursively extracts text and reasoning from structured arrays and objects. Claude parsing maps text, thinking, redacted-thinking, and tool-use content blocks to the same fields. Qwen parsing additionally recognizes supported XML tool-call formats.

## Finish Reasons

`FinishReason` normalizes common provider values:

| Value | Typical meaning |
| --- | --- |
| `Stop` | A normal OpenAI-compatible stop. |
| `EndTurn` | A normal Anthropic turn ending. |
| `Length` or `MaxTokens` | The token limit was reached. |
| `StopSequence` | A configured stop sequence matched. |
| `ToolCall`, `ToolCalls`, `ToolUse`, or `FunctionCall` | The model requested a tool or function. |
| `ContentFilter` or `Refusal` | Output was filtered or refused. |
| `Pause` or `PauseTurn` | The provider paused generation. |
| `Missing` | The provider reported a missing reason. |
| `Unknown` | The reason was absent or unrecognized. |

Callers should also inspect `toolCalls`; provider-specific finish reasons are not always consistent.

## Tool Calls

Each `ToolCall` contains:

| Field | Purpose |
| --- | --- |
| `id` | Provider or generated call identifier. |
| `name` | Registered function name requested by the model. |
| `arguments` | Parsed JSON arguments. |

```d
foreach (call; result.choice.toolCalls)
{
    Tool tool = ep.tools.get(call.name);
    JSONValue output = tool.impl(call.arguments);
    ctx.tool(call.id, output);
}
```

See [Tools](TOOLS.md) for a complete request-execution-response flow.

## Usage

`Completion.usage` is a `Usage` value:

| Field | Purpose |
| --- | --- |
| `modelName` | Model reported by the provider, or the requested model name. |
| `latency` | Client-observed request latency in milliseconds. |
| `promptTokens` | Prompt or input tokens. |
| `completionTokens` | Generated or output tokens. |
| `totalTokens` | Provider total, or prompt plus completion tokens. |
| `cacheHits` | Prompt tokens read from cache. |
| `cacheMisses` | Prompt tokens not read from cache. |

OpenAI-compatible responses use `prompt_tokens`, `completion_tokens`, and related token-detail fields. Claude usage includes uncached input, cache-read input, and cache-creation input in `promptTokens`, with cache-read tokens reported as hits.

Providers may omit usage fields. Missing numeric values remain zero.

## Embeddings

`embeddings` returns `Embedding!T` for one input or `Embedding!T[]` for an input array:

```d
Embedding!float embedding = embeddings(
    ep,
    "nomic-embed-text",
    "Hello, world!"
);
float[] vector = embedding.value;

Embedding!double[] embeddings = embeddings!double(
    ep,
    "nomic-embed-text",
    ["first", "second"]
);
```

`Embedding!T` aliases itself to its `value` field, so it implicitly converts to `T[]`:

```d
float[] vector = embedding;
```

The element type defaults to `float`. Numeric JSON values are cast to the requested type; null entries retain that type's default value. A response without a valid first vector returns an empty single `Embedding`, while array requests preserve the number and order of returned embedding entries.

Claude's endpoint implementation does not support embeddings and throws `EndpointException`.

## Raw Responses and Errors

Use `Completion.raw` when provider fields are not represented by the shared types. The HTTP helper also adds a numeric `latency` field to parsed response objects before model parsing.

Non-success HTTP responses and invalid JSON responses throw `EndpointException`. A completion response containing an `error` member is also rejected before choices are returned.
