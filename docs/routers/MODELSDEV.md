---
layout: default
title: Models.dev
parent: Routers
nav_order: 5
permalink: /routers/modelsdev/
---

# Models.dev

`ModelsDev` reads the public models.dev catalog without authentication. It doesn't run models. Model activation, configuration access, completions, embeddings, and decisions throw plain `Exception`.

## Reading the Catalog

```d
import intuit;

auto models = new ModelsDev();
ModelDetails[string] catalog = models.catalog();

if (auto details = "openrouter/openai/gpt-4o" in catalog)
    assert(details.id == "openrouter/openai/gpt-4o");
```

The constructor accepts `url` (default `https://models.dev`) and display `name` (default `"models.dev"`). It appends `/api.json?type=all`, including specialized model types such as decision models.

Catalog access fetches and caches metadata on demand. `refresh()` reloads it explicitly. No inference service is created by constructing this router.

## Model IDs

IDs use `<catalog-provider>/<model-id>`. For example, a model listed by the `openrouter` provider can have the ID `openrouter/openai/gpt-4o`.

The additional prefix keeps provider-specific limits and prices separate. It's a catalog identifier, not necessarily an ID you can send to an inference API. OpenRouter itself uses slugs such as `openai/gpt-4o`.

## Catalog Mapping

The adapter maps names, descriptions, context/output limits, reported modalities, and base input/output prices. Models.dev prices are already in USD per million tokens, so they aren't rescaled.

Reported `decision`, `embedding`, and `reranking` model types add the corresponding output modality. Explicit flags map tool support, reasoning, structured output, and temperature support.

Cache, audio, and reasoning prices aren't represented. Neither are pricing tiers, lifecycle flags, or provider connection settings. Missing values retain their defaults rather than being inferred.

See [Model Catalogs](../catalog/) for shared fields and [Routers](../) for inference-capable alternatives.