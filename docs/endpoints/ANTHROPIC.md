---
layout: default
title: Anthropic
parent: Endpoints
nav_order: 3
permalink: /endpoints/anthropic/
---

# Anthropic

`Claude` implements the Anthropic Messages API. It translates shared completion settings to Anthropic payloads and returns the same `Completion` type as the other adapters. Embeddings and decisions aren't supported.

## Connection

```d
import intuit;

auto ep = new Claude("https://api.anthropic.com", "your-api-key");
auto cfg = cast(ClaudeModelConfig)ep.config("claude-sonnet-4-5");
cfg.maxTokens = 2048;
```

The constructor accepts a base `url`, optional `key`, and display `name` (default `"Claude"`). The endpoint sends a nonempty key in `x-api-key` and sets `anthropic-version` to `2023-06-01`.

Use the host without `/v1` or a trailing slash. Discovery requests `GET /v1/models`, and completions use `POST /v1/messages`. Discovery reads the returned `data` array without following pagination.

Set `maxTokens` before requesting a completion when the server requires it. Intuit leaves it unset by default.

## Configuration

`config(modelName)` returns a cached `ClaudeModelConfig`. It supports the common `temperature`, `topP`, and `maxTokens` fields plus these settings.

| Field | Behavior |
| --- | --- |
| `topK` | Sends `top_k` when nonnegative. |
| `stopSequences` | Sends nonempty Anthropic stop sequences. Use this instead of the base `stop` field. |
| `system` | Prepends a system prompt to string system messages from the context. |
| `thinkingBudget` | Sends enabled thinking with this budget when nonnegative. |

The Messages completion builder doesn't serialize the base `seed`, `stop`, or `params` fields. Use the supported fields rather than assuming all base configuration settings apply.

## Payload Translation

String system messages are removed from the message array and joined with newlines into the top-level `system` field. Non-string system content isn't included.

`setResponseSchema` places the normalized schema under `output_config.format`. Tool-choice values map to Anthropic's `auto`, `any`, and `none` forms, and a required tool uses its name. Tool definitions use `input_schema`.

Other message objects are passed through after system extraction. In particular, the adapter doesn't translate shared OpenAI-style assistant tool calls and tool-result messages into Anthropic request content blocks. Tool registration and response parsing work, but the shared tool round-trip examples aren't a complete Anthropic conversation adapter.

## Responses

Text blocks populate `Choice.text`. Thinking blocks populate `Choice.reasoning`, while redacted thinking becomes `[redacted thinking]`. Tool-use blocks become shared `ToolCall` values.

The parser returns one choice and preserves the full response as both `Completion.raw` and `Choice.raw`. Prompt usage includes ordinary input, cache reads, and cache creation. Cache reads are reported as hits.

See [Models](../../models/) for structured output, [Responses](../../responses/) for shared fields, and [Errors](../../errors/) for failure handling.