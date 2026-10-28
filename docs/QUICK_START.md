---
layout: default
title: Quick Start
nav_order: 2
permalink: /quick-start/
---

# Quick Start

This guide makes a completion request and continues the conversation. Use an endpoint when you want to choose the model and keep history yourself. Use an inference-capable router when you want it to own the history and active model.

## Installation

```sh
dub add intuit
```

You'll need DUB and a D compiler. Intuit doesn't host models. For the local example, start an OpenAI-compatible server such as LM Studio or Ollama, load a chat model, and enable its HTTP API.

The example assumes the server listens at `http://localhost:1234` and accepts `llama3` as a model ID. Replace both values with your deployment's settings. Ollama commonly uses port `11434`. Hosted providers need their own URL and API key.

## First Conversation

Save this as your D application's source and run it with DUB.

```d
import intuit;

import std.stdio : writeln;

void main()
{
    auto ep = new OpenAI("http://localhost:1234");
    Context ctx;
    ctx.system("Answer concisely.");
    ctx.user("What is D?");

    Completion first = completions(ep, "llama3", ctx);
    writeln(first.text);

    ctx.user("Show me a small example.");
    Completion second = completions(ep, "llama3", ctx);
    writeln(second.text);
}
```

Each request sends `ctx` and appends the first assistant choice to it. The endpoint doesn't retain history itself.

Use a base URL without `/v1` or a trailing slash. Intuit appends `/v1/chat/completions` directly. To inspect the server's advertised model IDs, use `ep.available()` as shown in [Getting Started with Endpoints](../endpoints/getting-started/#finding-a-model).

## Using a Router

A router keeps history for you. This alternative uses an OpenRouter account and a model available to that account.

```d
import intuit;

import std.stdio : writeln;

void main()
{
    auto router = new OpenRouter("your-openrouter-key");
    router.active("openai/gpt-4o");
    router.context.system("Answer concisely.");

    Completion first = completions(router, "What is D?");
    Completion second = completions(router, "How does it compare to C++?");
    writeln(first.text);
    writeln(second.text);
}
```

Select a model before requesting a completion. OpenRouter may fetch its catalog during activation. LiteLLM and Models.dev are catalog-only and can't run this example.

## Next Steps

- [Endpoints](../endpoints/) for provider clients and their supported operations.
- [Routers](../routers/) for active models and catalog access.
- [Models](../models/) for request settings and structured output.
- [Responses](../responses/) for completion fields and embedding requests.
- [Tools](../tools/) and [Decisions](../decisions/) for those workflows.
- [Errors](../errors/) if a request fails.
