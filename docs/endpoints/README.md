# Endpoints

An endpoint connects to one provider API through `IEndpoint`. Requests name a model explicitly. The endpoint caches model settings and tools, but doesn't keep conversation history.

- [Getting Started](GETTING_STARTED.md) makes a request and continues a conversation.
- [Models](../MODELS.md) covers shared request settings and structured output.

## Providers

| Guide | Class | Supported operations |
| --- | --- | --- |
| [OpenAI](OPENAI.md) | `OpenAI` | Completions, embeddings, and decisions when the server supports them. |
| [Anthropic](ANTHROPIC.md) | `Claude` | Completions through the Messages API. |
| [Qwen](QWEN.md) | `Qwen` | OpenAI-compatible requests with Qwen settings and XML tool-call extraction. |
| [TypeSafe](TYPESAFE.md) | `TypeSafe` | System One decisions only. |

The interface exposes all request methods, but each provider can reject unsupported operations. Check the provider guide rather than assuming every endpoint supports every method.

## Choosing an Endpoint

Use `OpenAI` for a compatible local or hosted server, `Qwen` for its additional settings and parsing, `Claude` for Anthropic's Messages API, and `TypeSafe` for System One decisions.

Intuit doesn't host services or load models. See [Quick Start](../QUICK_START.md) for prerequisites, [Responses](../RESPONSES.md) for embedding requests, and [Decisions](../DECISIONS.md) for typed question/answer requests.
