---
layout: default
title: Model Catalogs
parent: Routers
nav_order: 6
permalink: /routers/catalog/
---

# Model Catalogs

A router catalog describes models. It doesn't establish that you can run them through that router or that your account has access. LiteLLM and Models.dev expose catalogs without implementing inference.

## Reading a Catalog

`catalog()` fetches metadata on demand and returns an associative array keyed by model ID. Use IDs from that router's own catalog rather than assuming that identifiers are interchangeable.

```d
import intuit;

import std.stdio : writeln;

auto router = new ModelsDev();
ModelDetails[string] catalog = router.catalog();

foreach (id, details; catalog)
    writeln(id, " ", details.contextLength);
```

`refresh()` fetches the remote catalog again. It doesn't clear conversation history, activate a model, or replace cached model configurations. OpenRouter, LiteLLM, and Models.dev fetch again on later catalog access if their cache is empty. Zen tracks whether a catalog has loaded, even when it contains no models.

Catalog access is a network operation on a cold cache. Connection and operation timeouts apply to it too. See [Errors](../../errors/#timeouts).

## Model Details

| Field | Meaning |
| --- | --- |
| `id` | Router-specific model identifier. |
| `name` | Display name. |
| `description` | Description supplied by the catalog. |
| `contextLength` | Reported context window in tokens. |
| `maxCompletionTokens` | Reported maximum output size. |
| `inputModalities` | Reported input kinds. |
| `outputModalities` | Reported output kinds. |
| `capabilities` | Reported parameters and features. |
| `promptCost` | USD per million input tokens. |
| `completionCost` | USD per million output tokens. |

Missing numeric fields remain zero. Zero cost can mean missing pricing, not necessarily free inference. Limits and prices describe catalog metadata, not enforced request settings or a billing quote.

`Modality` is a string-backed enum with text, image, audio, video, PDF, file, embedding, decisions, rerank, speech, and transcription values. `Embedding` and `Embeddings` retain distinct wire spellings.

`ModelCapability` identifies features such as `Tools`, `ToolChoice`, `Reasoning`, `ResponseFormat`, and `StructuredOutputs`. Check modalities for input/output support. The `Audio` capability alone doesn't establish audio input or output support.

## Selecting Models

Inference-capable routers activate one model at a time. Changing it preserves history.

OpenRouter sets the context compactor's token limit when it finds metadata for the selected ID. Zen does so only when the reported context length is positive. Neither change immediately compacts existing messages. Catalog refresh alone doesn't update the active model's limit.

A catalog entry doesn't configure the request's maximum output tokens. Set that through [ModelConfig](../../models/) when needed. Compaction is an estimate based on prior reported usage, not a tokenizer or a guarantee that a request fits. See [Context](../../context/#compaction).

## Provider Mapping

- [OpenRouter](../openrouter/) preserves API model slugs and maps advertised parameters and pricing.
- [LiteLLM](../litellm/) uses proxy model names and maps its `model_info` fields.
- [Models.dev](../modelsdev/) prefixes each model ID with its catalog provider.
- [Zen](../zen/) currently maps IDs, names, descriptions, and context lengths only.

Provider guides explain omitted metadata and ID formats. Catalog capabilities don't add request methods to Intuit, and a model listed by a remote service can still use an API family the adapter doesn't support.
