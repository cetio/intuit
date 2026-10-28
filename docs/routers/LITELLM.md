# LiteLLM

`LiteLLM` currently exposes the proxy's model catalog only. Model activation, configuration access, completions, embeddings, and decisions throw plain `Exception`.

## Connection

```d
import intuit;

auto router = new LiteLLM("http://localhost:4000", "your-proxy-key");
ModelDetails[string] catalog = router.catalog();
```

The constructor accepts `url` (default `http://localhost:4000`), optional `key`, and display `name` (default `"LiteLLM"`). A nonempty key uses bearer authentication.

Catalog requests append `/v1/model/info` to the supplied base URL. The cache loads on demand and can be replaced with `refresh()`.

## Catalog Mapping

IDs come from `model_name`. Display names use `model_info.key` when present and otherwise fall back to the ID.

Context length comes from `max_input_tokens`, falling back to `max_tokens`. Output limits use `max_output_tokens` with the same fallback. Prices are converted from USD per token to USD per million tokens.

Input modalities start with text, then add explicitly reported vision, audio, and PDF support. Output modalities use the reported mode and audio-output flag. Capability mapping includes tools, tool choice, response schemas, system messages, reasoning, prompt caching, and web search.

These fields describe the proxy's models. They don't enable inference in this router. See [Model Catalogs](CATALOG.md) for the shared metadata type.

## Using a Proxy for Inference

Use an [OpenAI endpoint](../endpoints/OPENAI.md) against an OpenAI-compatible LiteLLM proxy when you need completions or embeddings.

```d
auto ep = new OpenAI("http://localhost:4000", "your-proxy-key");
Context ctx;
ctx.user("Explain associative arrays.");
Completion result = completions(ep, "your-proxy-model-name", ctx);
```

The endpoint doesn't own history. Keep the `Context` yourself, and use the model name configured on the proxy.