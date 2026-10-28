---
layout: default
title: Context
nav_order: 6
permalink: /context/
---

# Context

A `Context` holds conversation messages. Keep one yourself when using an endpoint, or use the context owned by an inference-capable router.

## Building a Context

Use the role helpers to append messages:

```d
import intuit;

Context ctx;
ctx.system("You are a helpful assistant.");
ctx.user("What is D?");
ctx.assistant("D is a systems programming language.");
ctx.user("Show me a small example.");
```

Each helper returns the context by reference, so calls can be chained:

```d
ctx.system("Answer concisely.")
    .user("What is an associative array?");
```

| Member | Message type | Purpose |
| --- | --- | --- |
| `append(message)` | Any `IMessage` | Append a pre-built message. |
| `system(data)` | `SystemMessage` | Add system instructions. |
| `user(data)` | `UserMessage` | Add user content. |
| `assistant(text)` | `AssistantMessage` | Add a synthetic assistant response. |
| `assistant(text, calls)` | `AssistantMessage` | Add synthetic text and tool calls. |
| `assistant(completion, index)` | `AssistantMessage` | Preserve a completion choice and its usage. |
| `tool(data)` | `ToolMessage` | Add an unassociated tool result. |
| `tool(callId, data)` | `ToolMessage` | Add a result for a specific tool call. |
| `clear()` | None | Remove every message. |
| `length` | None | Return the message count. |

The data helpers use Intuit's JSON conversion support. Strings, numbers, booleans, arrays, string-keyed associative arrays, `JSONValue`, and supported aggregate types can be passed directly.

## Request Behavior

| Request | History behavior |
| --- | --- |
| `completions(ep, model, ctx)` | Sends the caller's context and appends the first assistant choice. |
| `completions(ep, model, text)` | Sends the supplied value without retaining the response. |
| `completions(router, data)` | Appends data as a user message, sends history, then appends the first assistant choice. |
| `completions(router)` | Sends existing history without adding another user message. |
| Embeddings and decisions | Don't append messages or run tool policy. |

Context-backed completions can perform further rounds through `toolPolicy`. The default policy doesn't allow tools; registered calls become pending, and missing tools fail. See [Tools](../tools/) for authorization and execution.

Clear history with `ctx.clear()` or `router.context.clear()`. This removes messages but retains the compactor and policy. Switching an active model doesn't clear history.

For complete conversation examples, see [Endpoints](../endpoints/getting-started/) or [Routers](../routers/getting-started/).

## Message Types

Every message implements `IMessage`, which exposes `role` and `toJSON()`.

| Type | Content |
| --- | --- |
| `SystemMessage` | JSON system instructions. |
| `UserMessage` | JSON user content. |
| `AssistantMessage` | A `Completion` choice or synthetic text and tool calls. |
| `ToolMessage` | JSON tool output with an optional originating call ID. |

`AssistantMessage` retains the complete `Completion`, selected choice index, token usage, text, and tool calls. This allows context compaction to account for usage reported by earlier requests.

All messages serialize to OpenAI-style objects. `ClaudeModelConfig` translates string system messages to the Anthropic top-level `system` field, but doesn't convert shared tool history to Anthropic content blocks. See [Anthropic](../endpoints/anthropic/#payload-translation).

## Tool Messages

A tool-result message references the originating call ID.

```d
ctx.tool("call_123", "The current temperature is 21 degrees.");
```

Use the actual ID from the assistant's tool call, not the placeholder above. The assistant message is already appended by a context-backed request. See [Tools](../tools/#manual-execution) for the complete round-trip and safe handling of policy-managed calls.

## Compaction

Assign a `Compactor` to trim history after an append exceeds a message-count or estimated token limit. A caller-created context has no compactor by default.

```d
ctx.compactor = new Compactor();
ctx.compactor.maxMessages = 20;
ctx.compactor.maxTokens = 32_000;
ctx.compactor.strategy = CompactorStrategy.Trim;
```

A zero limit disables that limit. A new `Compactor` defaults to a 250,000-token limit and no message-count limit.

The token total is estimated from stored assistant completions:

- Sum the completion tokens from every assistant message.
- Add the prompt tokens from the most recent assistant message.

Synthetic assistant messages do not contribute tokens unless their wrapped completion contains usage data.

This isn't tokenization of the current messages. New user input isn't measured before a response reports usage, and missing usage can undercount history. A compactor doesn't guarantee that a request fits the server's context window.

### Trim Strategy

`CompactorStrategy.Trim` removes the oldest non-system messages until every configured limit is satisfied. System messages are preserved. If only system messages remain and still exceed a configured message limit, no more messages can be removed.

Removing old messages may not reduce the token estimate immediately, because the latest assistant message still reports prompt usage for an earlier request.

Because messages are removed individually, trimming does not guarantee that assistant tool calls and their matching tool results remain together.

### Callback Strategy

`CompactorStrategy.Callback` delegates compaction to caller code:

```d
ctx.compactor.strategy = CompactorStrategy.Callback;
ctx.compactor.callback = (IMessage[] messages) {
    if (messages.length <= 10)
        return messages;
    return messages[$ - 10..$];
};
```

The callback runs only after a configured limit is exceeded. If it is null, the messages are left unchanged.

The example keeps only the last ten messages, including any system messages in that slice. A callback is responsible for preserving required instructions and complete tool exchanges. Its output isn't repeatedly compacted until it satisfies the limits.

Built-in routers construct a compactor automatically. Activation can update its token limit from catalog metadata without immediately compacting existing messages. Compaction occurs on the next append. See [Model Catalogs](../routers/catalog/#selecting-models) for the provider differences.

## Serialization

`ctx.toJSON()` returns an array of serialized message objects:

```d
import std.json : JSONValue;

JSONValue messages = ctx.toJSON();
```

Use the public `messages` array when individual message types or stored completion metadata are needed.

Direct changes to `messages` don't trigger compaction. Appending through the helper methods does.
