---
layout: default
title: Qwen
parent: Endpoints
nav_order: 4
permalink: /endpoints/qwen/
---

# Qwen

`Qwen` extends the OpenAI-compatible endpoint with `QwenModelConfig`. Use it when your server needs Qwen-specific settings or returns tool calls embedded in XML.

## Connection

```d
import intuit;

auto ep = new Qwen("http://localhost:1234");
auto cfg = cast(QwenModelConfig)ep.config("qwen3-30b-a3b");
cfg.enableThinking = false;
cfg.topK = 40;
```

The constructor accepts a base `url`, optional API `key`, and display `name` (default `"Qwen"`). Authentication and request paths are inherited from [OpenAI](../openai/). Discovery and explicit configuration both create `QwenModelConfig` instances.

Replace the URL and model ID with those accepted by your server. Qwen-specific fields aren't supported by every OpenAI-compatible deployment.

## Completion Settings

| Field | Behavior |
| --- | --- |
| `enableThinking` | Defaults to `true`. Setting it to `false` sends `enable_thinking: false` only when `chatTemplateKwargs` is null. |
| `chatTemplateKwargs` | Sends `chat_template_kwargs` when non-null. Takes precedence over the `enableThinking` shortcut. |
| `topK` | Sends `top_k` when nonnegative. |
| `thinkingBudget` | Sends `thinking_budget` when nonnegative. |
| `mmProcessorKwargs` | Sends `mm_processor_kwargs` when non-null. |

The base [completion settings](../../models/#completion-parameters) are applied first. Qwen-specific settings are added afterward and can replace the corresponding fields in `params`.

## Embedding Settings

`encodingFormat` defaults to `"float"` and is omitted at that value. `dimensions` is sent only when positive. These settings affect embedding requests, not completions.

The embedding parser expects numeric arrays. Setting a non-float wire encoding doesn't add a decoder for that format, so keep the default unless your server still returns numeric vectors.

## XML Tool Calls

Native OpenAI-style `tool_calls` take precedence. When no native calls are present, the parser looks for supported `<tool_call>` blocks in the response text.

It accepts Qwen3-Coder function/parameter tags and Hermes-style JSON-in-XML forms. Extracted calls populate `Choice.toolCalls`, receive synthetic IDs, and are removed from `Choice.text`. Surrounding prose is preserved. Unrecognized blocks aren't guaranteed to produce calls.

Extraction doesn't authorize execution. Use the shared [Tools](../../tools/) workflow to register functions, evaluate policy, and handle pending calls.