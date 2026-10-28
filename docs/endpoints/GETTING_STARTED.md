---
layout: default
title: Getting Started with Endpoints
parent: Endpoints
nav_order: 1
permalink: /endpoints/getting-started/
---

# Getting Started with Endpoints

An endpoint connects to one provider API. This guide finds a model and makes a short conversation through an OpenAI-compatible server. Start with [Quick Start](../../quick-start/) if you haven't installed Intuit or prepared a server.

## Creating an Endpoint

Create an endpoint with the server's base URL and an optional API key.

```d
import intuit;

auto ep = new OpenAI("http://localhost:1234");
```

Replace the URL with your running server's address. Use the base URL without `/v1` or a trailing slash, because Intuit appends request paths directly. For hosted OpenAI, use `new OpenAI("https://api.openai.com", "your-api-key")`.

## Finding a Model

Inspect the models advertised by the server.

```d
import std.stdio : writeln;

foreach (cfg; ep.available())
    writeln(cfg.name);
```

`available()` makes a network request and adds advertised models to the endpoint's cache. It returns the entire cache, which can also contain names you configured manually.

Choose an ID accepted by your server. The examples use `llama3` as a placeholder. You can request a known model without calling `available()` first.

## A Single Request

Pass a model name and a prompt to `completions`. This request doesn't retain conversation history.

```d
Completion result = completions(ep, "llama3", "Why is the sky blue?");
writeln(result.text);
```

## Continuing a Conversation

Keep a `Context` when requests need previous messages.

```d
Context ctx;
ctx.system("You are a helpful assistant.");
ctx.user("What is D?");

Completion first = completions(ep, "llama3", ctx);
ctx.user("How does it compare to C++?");
Completion second = completions(ep, "llama3", ctx);
writeln(second.text);
```

Intuit appends the first assistant choice after each successful completion. The endpoint doesn't keep a separate copy of your history.

## Request Settings

Get the cached model configuration and change its settings before making the request.

```d
ModelConfig cfg = ep.config("llama3");
cfg.temperature = 0.7;
cfg.maxTokens = 1024;
```

The configuration is reused for later requests with that model name. Creating one doesn't verify that the server offers the model or supports every setting.

## Next Steps

- [Models](../../models/) for common settings and structured output.
- [Context](../../context/) for message types, history, and compaction.
- [Responses](../../responses/) for completion fields and embedding requests.
- [Tools](../../tools/) and [Decisions](../../decisions/) for additional request workflows.
- [Provider guides](../#providers) for authentication, formats, and supported operations.
- [Errors](../../errors/) for exception types and request timeouts.
