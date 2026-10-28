---
layout: default
title: Documentation
nav_order: 1
permalink: /
---

# Documentation

Start with [Quick Start](quick-start/) to install Intuit and make a conversation request.

## Clients

Use an endpoint to name a model with each request and keep history yourself. Use an inference-capable router to select an active model and let it keep history. Catalog-only routers supply metadata, not inference.

- [Endpoints](endpoints/) compares OpenAI, Anthropic, Qwen, and TypeSafe.
- [Routers](routers/) compares OpenRouter, Zen, LiteLLM, and Models.dev.
- [Model Catalogs](routers/catalog/) explains discovery, capabilities, limits, and prices.

## Shared Guides

| Guide | What you'll find |
| --- | --- |
| [Models](models/) | Cached settings, sampling parameters, structured output, and tool choice. |
| [Context](context/) | Conversation messages, history mutation, and compaction. |
| [Responses](responses/) | Completion choices, usage, reasoning, and single/batch embeddings. |
| [Tools](tools/) | Function registration, manual execution, policy decisions, and round limits. |
| [Decisions](decisions/) | Predicate, choice, and score requests and typed answers. |
| [Errors](errors/) | Request and parsing exceptions, timeouts, and retry considerations. |

## Development

- [Extending Intuit](extending/) covers custom configurations, endpoints, and routers.
- [Testing](https://github.com/cetio/intuit/blob/main/TESTING.md) covers the library build, offline tests, and integration prerequisites.
- [Contributing](https://github.com/cetio/intuit/blob/main/CONTRIBUTING.md) covers focused changes and documentation conventions.
