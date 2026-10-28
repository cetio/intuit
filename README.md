# Intuit

[![DUB](https://img.shields.io/badge/DUB-red?style=for-the-badge&logo=hackthebox&logoColor=white)](https://code.dlang.org/packages/intuit)
[![Documentation](https://img.shields.io/badge/Documentation-brightgreen?style=for-the-badge&logo=readthedocs&logoColor=white)](https://mortifying.si/intuit/)

[![License](https://img.shields.io/badge/License-Apache--2.0-blue)](LICENSE.txt)
[![CI](https://github.com/cetio/intuit/actions/workflows/ci.yml/badge.svg)](https://github.com/cetio/intuit/actions/workflows/ci.yml)

Intuit is a D library for LLM completions, embeddings, and typed decision requests, with a focus on local models. It connects to provider APIs without external language bindings.

Endpoints name a model with each request and leave conversation history to the caller. Inference-capable routers keep their own history and use an active model. Both expose shared configuration and response types.

## Installation

```sh
dub add intuit
```

## Quick Start

Start an OpenAI-compatible server and load a chat model first. This example assumes a server at `http://localhost:1234` and a model ID of `llama3`. Replace them with your server's settings.

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

Use a base URL without `/v1` or a trailing slash. Intuit appends request paths directly. It doesn't start servers or load models. See [Quick Start](docs/README.md) for a complete program and a router alternative.

## Providers

Provider adapters expose different operations. Support also depends on the server and selected model.

| Endpoint | Module |
| --- | --- |
| [`OpenAI`](docs/endpoints/OPENAI.md) | `intuit.provider.openai` |
| [`Claude`](docs/endpoints/ANTHROPIC.md) | `intuit.provider.claude` |
| [`Qwen`](docs/endpoints/QWEN.md) | `intuit.provider.qwen` |
| [`TypeSafe`](docs/endpoints/TYPESAFE.md) | `intuit.provider.typesafe` |

`OpenAI` handles compatible APIs, `Qwen` adds provider settings and XML tool-call extraction, `Claude` uses Anthropic Messages, and `TypeSafe` handles System One decisions. See [Endpoints](docs/endpoints/README.md) for supported operations and limitations.

## Routers

Routers expose model catalogs through `IRouter`. OpenRouter and Zen also support inference and keep conversation history.

| Router | Requests |
| --- | --- |
| [OpenRouter](docs/routers/OPENROUTER.md) | Completions, embeddings, and System One decisions. |
| [Zen](docs/routers/ZEN.md) | Completions for supported API families. |
| [LiteLLM](docs/routers/LITELLM.md) | Catalog only. |
| [Models.dev](docs/routers/MODELSDEV.md) | Catalog only, without authentication. |

See [Routers](docs/routers/README.md) for usage and [Model Catalogs](docs/routers/CATALOG.md) for discovery. Streaming isn't implemented.

## Documentation

Guides are published at [mortifying.si/intuit](https://mortifying.si/intuit/), covering installation, endpoints, routers, model catalogs, and shared request, response, tool, and error references.

Development notes live in [Testing](TESTING.md) and [Contributing](CONTRIBUTING.md).

## License

Intuit is licensed under [Apache-2.0](LICENSE.txt).
