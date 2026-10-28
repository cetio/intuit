---
layout: default
title: Endpoints
nav_order: 3
has_children: true
permalink: /endpoints/
---

# Endpoints

An endpoint connects to one provider API through `IEndpoint`. Requests name a model explicitly. The endpoint caches model settings and tools, but doesn't keep conversation history.

- [Getting Started](getting-started/) makes a request and continues a conversation.
- [Models](../models/) covers shared request settings and structured output.

## Providers

| Guide | Class | Supported operations |
| --- | --- | --- |
| [OpenAI](openai/) | `OpenAI` | Completions, embeddings, and decisions when the server supports them. |
| [Anthropic](anthropic/) | `Claude` | Completions through the Messages API. |
| [Qwen](qwen/) | `Qwen` | OpenAI-compatible requests with Qwen settings and XML tool-call extraction. |
| [TypeSafe](typesafe/) | `TypeSafe` | System One decisions only. |

The interface exposes all request methods, but each provider can reject unsupported operations. Check the provider guide rather than assuming every endpoint supports every method.

## Choosing an Endpoint

Use `OpenAI` for a compatible local or hosted server, `Qwen` for its additional settings and parsing, `Claude` for Anthropic's Messages API, and `TypeSafe` for System One decisions.

Intuit doesn't host services or load models. See [Quick Start](../quick-start/) for prerequisites, [Responses](../responses/) for embedding requests, and [Decisions](../decisions/) for typed question/answer requests.
