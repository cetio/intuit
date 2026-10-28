# Responses

Intuit exposes shared completion and embedding types. Decision answers have their own [guide](DECISIONS.md).

## Completion

`completions` returns a `Completion`:

```d
import intuit;

auto ep = new OpenAI("http://localhost:1234");
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
if (result.choices.length > 1)
{
    Choice second = result.choice(1);
}
```

Convenience accessors read common values:

```d
import std.json : JSONValue;

string text = result.text;
string reasoning = result.reasoning;
JSONValue parsed = result.json;
```

`text(index)`, `reasoning(index)`, and `json(index)` select another choice. `json()` is appropriate only when the choice contains JSON text. Invalid JSON throws `MalformedResponseException`, preserving the response in `rawText` and the candidate text in `candidateText`. It doesn't validate a response schema.

## Choice

A `Choice` represents one generated alternative:

| Field | Purpose |
| --- | --- |
| `raw` | Raw JSON for the choice, or the full response for Claude. |
| `content` | Raw OpenAI-compatible message content. |
| `text` | Extracted response text. |
| `reasoning` | Extracted reasoning or thinking text. |
| `finishReason` | Wire reason generation ended, represented by a string-backed enum. |
| `logProbs` | Provider log-probability data when returned. |
| `toolCalls` | Tools requested by the model. |

OpenAI-compatible parsing accepts string content and recursively extracts text and reasoning from structured arrays and objects. Claude parsing maps text, thinking, redacted-thinking, and tool-use content blocks to the same fields. Qwen parsing additionally recognizes supported XML tool-call formats.

## Finish Reasons

`FinishReason` names common provider values without merging distinct wire spellings.

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
| `Unknown` | The reason was absent, non-string, or explicitly `"unknown"`. |

Unrecognized string reasons retain their wire value and aren't automatically converted to `Unknown`. Inspect `toolCalls` too, because finish reasons don't consistently establish whether a response includes calls.

## Tool Calls

Each `ToolCall` contains:

| Field | Purpose |
| --- | --- |
| `id` | Provider or generated call identifier. |
| `name` | Registered function name requested by the model. |
| `arguments` | Parsed JSON arguments. |
| `policyResult` | Execution-policy status and explanation for context-backed requests. |

Returned calls may already have been evaluated or executed. See [Tools](TOOLS.md) before invoking them. The guide covers manual-only execution, policy-managed rounds, and partial batches.

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

Providers may omit usage fields. Missing counts remain zero, while missing floating-point latency defaults to `NaN`. During automatic tool rounds, each completion has its own usage. The final returned completion doesn't aggregate all rounds.

## Embeddings

`embeddings` returns `Embedding!T` for one input or `Embedding!T[]` for an input array:

```d
Embedding!float embedding = embeddings(
    ep,
    "nomic-embed-text",
    "Hello, world!"
);
float[] vector = embedding.value;

Embedding!double[] batch = embeddings!double(
    ep,
    "nomic-embed-text",
    ["first", "second"]
);
```

`Embedding!T` aliases itself to its `value` field, so it implicitly converts to `T[]`:

```d
float[] values = embedding;
```

The element type defaults to `float`. Numeric values are cast to the requested type. Null and nonnumeric entries retain that type's default (`NaN` for floating-point types, zero for integers).

A single request reads the first returned vector, or returns an empty vector if none is usable. Batch requests preserve the order and count of returned embedding entries, not necessarily the input count. Entries without an `embedding` member are skipped, and non-array vectors become empty.

OpenRouter uses its active embedding model. This leaves conversation history unchanged.

```d
auto router = new OpenRouter("your-openrouter-key");
router.active("openai/text-embedding-3-small");
Embedding!float routerVector = embeddings(router, "Hello, world!");
```

The server and selected model must support embeddings. Claude, TypeSafe, Zen, and the catalog-only routers don't support this operation through Intuit.

## Raw Responses and Errors

Use `Completion.raw` when provider fields are not represented by the shared types. The HTTP helper also adds a numeric `latency` field to parsed response objects before model parsing.

HTTP failures and detected provider error envelopes throw `EndpointException` or a subclass. Invalid HTTP response JSON and invalid `Completion.json` text throw `MalformedResponseException`, which isn't an `EndpointException`. Other field and tool-argument parsing errors can propagate from conversion code. See [Errors](ERRORS.md) for the full distinction.
