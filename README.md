# Intuit

[![DUB Package](https://img.shields.io/badge/DUB-latest-red)](https://code.dlang.org/packages/intuit)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue)](LICENSE.txt)
[![CI](https://github.com/cetio/intuit/actions/workflows/ci.yml/badge.svg)](https://github.com/cetio/intuit/actions/workflows/ci.yml)

Intuit is a library for interacting with AI LLM and embeddings endpoints, with a focus on local models. Endpoints, models, and routers are built around small interfaces so that new providers and routing strategies are straightforward to add.

An `IEndpoint` wraps one provider API. An `IRouter` sits beside one or more endpoints, maintains its own `Context`, and operates on a single active model at a time.

## Installation

```sh
dub add intuit
```

## Quick Start

```d
import intuit;

import std.stdio : writeln;

auto ep = new OpenAI("http://localhost:1234");

Context ctx;
ctx.system("You are a helpful assistant.");
ctx.user("Why is the sky blue?");

Completion result = completions(ep, "llama3", ctx);
writeln(result.text);
```

Endpoints are constructed with a base URL and an optional API key. Intuit does not normalize it; provider request paths such as `/v1/chat/completions` are appended to it. Intuit does not host endpoints — use a local server such as [Ollama](https://ollama.com/) or [LM Studio](https://lmstudio.ai/), a router, or the vendor's API directly.

## Providers

Intuit provides endpoint and model definitions for the following providers:

| Endpoint | Module |
| --- | --- |
| `OpenAI` | `intuit.provider.openai` |
| `Claude` | `intuit.provider.claude` |
| `Qwen` | `intuit.provider.qwen` |

Support means the endpoint and model definitions are provided and tested for those API styles. OpenAI-compatible endpoints work with most local servers and many hosted models, but behavior varies by model version and capabilities.

`Qwen` extends `OpenAI` with Qwen-specific model parameters and XML tool-call extraction. `Claude` implements the Anthropic Messages API and does not support embeddings.

## Routers

Routers sit beside endpoints with their own interface (`IRouter`) and request functions. A router juggles one or more endpoints internally, maintains its own `Context`, and operates on a single active model at a time. Setting the active model adjusts the maintained context's `Compactor` token limit to match the model's context window.

| Router | Description | Reference |
| --- | --- | --- |
| `OpenRouter` | OpenRouter router with dynamic model discovery. | https://openrouter.ai/ |
| `LiteLLM` | LiteLLM proxy router. Catalog access only. | https://docs.litellm.ai/ |

Router request functions omit the model parameter. When data is provided it is appended to the maintained context; when omitted, the existing context state is used:

```d
auto router = new OpenRouter("sk-my-key");
router.active("openai/gpt-4o");

Completion result = completions(router, "Why is the sky blue?");
Completion followUp = completions(router);
```

## Documentation

- [Documentation index](docs/README.md)
- [Endpoints guide](docs/endpoints/README.md)
- [Routers guide](docs/routers/README.md)
- [Context](docs/CONTEXT.md)
- [Responses](docs/RESPONSES.md)
- [Tools](docs/TOOLS.md)
- [Testing](TESTING.md)
- [Contributing](CONTRIBUTING.md)

## License

Intuit is licensed under [Apache-2.0](LICENSE.txt).
