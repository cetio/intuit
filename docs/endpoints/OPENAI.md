---
layout: default
title: OpenAI
parent: Endpoints
nav_order: 2
permalink: /endpoints/openai/
---

# OpenAI

`OpenAI` implements OpenAI-compatible chat completions, embeddings, and decisions. It can connect to a hosted API or a local server such as Ollama or LM Studio. The server and model determine which operations actually work.

## Connection

```d
import intuit;

auto local = new OpenAI("http://localhost:1234");
auto hosted = new OpenAI("https://api.openai.com", "your-api-key");
```

The constructor accepts `url`, an optional `key`, and an optional display `name` (default `"OpenAI"`). A nonempty key is sent as `Authorization: Bearer <key>`.

Use the server's base URL without `/v1` or a trailing slash. Intuit appends paths directly and doesn't normalize URLs. For a proxy mounted under a prefix, include that prefix in the base URL.

| Operation | Path |
| --- | --- |
| Model discovery | `GET /v1/models` |
| Chat completions | `POST /v1/chat/completions` |
| Embeddings | `POST /v1/embeddings` |
| Decisions | `POST /v1/decisions` |

Intuit doesn't start a server, download models, or load them into memory. Start the service and make the model available before sending requests.

## Models and Requests

`available()` reads model IDs from the response's `data` array and adds uncached `ModelConfig` instances. It returns all cached configurations, including manually configured names. It doesn't remove names no longer advertised by the server or follow pagination.

`config(modelName)` gets or creates a configuration without checking the server. Model IDs must match what your deployment accepts.

See [Getting Started](../getting-started/) for completions, [Models](../../models/) for request settings, and [Responses](../../responses/#embeddings) for single and batch embeddings. Decision requests have their own [guide](../../decisions/).

## Compatibility

OpenAI-compatible describes a payload shape, not identical behavior across servers. Structured output, tools, reasoning, embeddings, and decisions need support from the selected model and deployment. Sending an extra field doesn't make the server implement it.

The parser accepts ordinary message text, structured text/reasoning content, and native function tool calls. It preserves raw responses for provider fields outside the shared types. Qwen XML tool calls require the [Qwen](../qwen/) adapter.

Streaming and the OpenAI Responses API aren't implemented. `legacyCompletions(ep, payload)` still sends `/v1/completions`, but it's deprecated in favor of chat completions.