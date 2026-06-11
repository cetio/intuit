# Intuit

[![License](https://img.shields.io/badge/License-AGPL--3-blue)](LICENSE.txt)

Intuit is a library for interacting with various AI endpoints/models, with a focus on local models. Intuit endpoints and models are highly extensible through interfaces.

## Features

- **Completions**: Text completions, with choices and easy parsing to D types.
- **Structured Output**: JSON schema (model options) as well as JSON parsing for responses.
- **Context Management**: System/user/assistant messages and preserved context via `Context`.
- **Streaming Support**: SSE streaming is available for real-time completions through `streamCompletions`.
- **Embeddings**: Embeddings with variable float sizes and SIMD math utilities.
- **Tool Use**: Native D functions as tools with automatic JSON schema generation.
- **Easy Integration**: Intuit is deliberately streamlined and adding/using new endpoints/models is straightforward.

## Usage

`dub add intuit`

### Endpoints

Intuit provides endpoint and model definitions for the following providers:

| Endpoint | Module |
|----------|--------|
| Claude | `intuit.provider.claude` |
| OpenAI | `intuit.provider.openai` |
| Qwen | `intuit.provider.qwen` |

Support does NOT mean that Intuit will work with every model from these providers or ONLY these providers, but rather that the endpoint and model definitions are provided and tested for those API styles. OpenAI endpoints are likely to work with most models, but will vary based on the model version and capabilities.

Intuit does not host endpoints, and this must be done locally via something like [Ollama](https://ollama.com/) or [LM Studio](https://lmstudio.ai/) or remotely via a router or the vendors themselves. Intuit supports providing API keys as well as base urls for endpoints.

### Creating an Endpoint

Endpoints are instantiated with a base URL and an optional API key:

```d
import intuit;

auto ep = new OpenAI("http://localhost:1234", "sk-my-key");
```

`Claude` and `Qwen` endpoints follow the same constructor signature. The base URL is normalized automatically, so trailing slashes and `/v1` suffixes are handled for you.

### Models

Fetch the models advertised by the endpoint or request one by name:

```d
IModel[] models = ep.available();

IModel model = ep.model("llama3");
OpenAIModel openaiModel = cast(OpenAIModel)model;
```

Models can be configured with chainable setters before use:

```d
openaiModel
    .temperature(0.7)
    .maxTokens(1024);
```

### Completions

Send a completion request with a string, a `Context`, or raw `JSONValue`:

```d
import intuit;

auto ep = new OpenAI("http://localhost:1234");
auto model = ep.model("llama3");

Completion result = completions(ep, model, "Why is the sky blue?");
writeln(result.text);
```

Use a `Context` to preserve conversation state. When `completions` receives a `Context`, the assistant response is appended automatically:

```d
Context ctx;
ctx.system("You are a helpful assistant.");
ctx.user("What is D?");

Completion result = completions(ep, model, ctx);
writeln(result.text);
// ctx now contains the assistant's reply
```

Structured output via JSON schema is supported by setting the model's `responseFormat`:

```d
import std.json;

JSONValue schema = JSONValue.emptyObject;
schema["type"] = JSONValue("object");
schema["properties"] = JSONValue.emptyObject;
schema["properties"]["answer"] = JSONValue("string");

auto m = cast(OpenAIModel)model;
m.responseFormat(schema);

Completion result = completions(ep, m, ctx);
JSONValue parsed = result.json;
```

### Streaming

For real-time token-by-token responses, use `streamCompletions`:

```d
CompletionStream stream = streamCompletions(ep, model, ctx);

while (!stream.complete || stream.callback !is null)
{
    try
    {
        Completion chunk = stream.next();
        write(chunk.text);
        stdout.flush();
    }
    catch (Exception ex)
        break;
}
```

`CompletionStream` is thread-safe and can also be consumed via a per-chunk `callback` or `collect()`.

### Embeddings

Request embedding vectors for a single input or an array of inputs:

```d
Embedding!float emb = embeddings(ep, model, "Hello, world!");
float[] vector = emb.value;

string[] inputs = ["Hello, world!", "Goodbye, world!"];
Embedding!float[] embs = embeddings(ep, model, inputs);
```

SIMD-accelerated utilities are provided for vector math:

```d
import intuit.response.embedding;

float similarity = cosineSimilarity(emb1.value, emb2.value);
float[] mean = normMean([emb1.value, emb2.value]);
```

### Tools

Register native D functions as tools with automatic JSON schema generation:

```d
import intuit;

@Description("Get the current weather for a location.")
string getWeather(string location)
{
    return "Sunny and 72 degrees in " ~ location;
}

auto ep = new OpenAI("http://localhost:1234");
ep.tools.add!getWeather();

Context ctx;
ctx.user("What is the weather in Paris?");

Completion result = completions(ep, model, ctx);
```

Tools marked with `autoexec = true` are invoked automatically and the conversation recurses until a text response is returned:

```d
ep.tools.add!getWeather(true);
Completion result = completions(ep, model, ctx); // loops internally
```

### Routers

Routers sit beside endpoints with their own interface (`IRouter`) and request functions. A router juggles one or more endpoints internally, maintains its own `Context`, and operates on a single active model at a time. Setting the active model adjusts the maintained context's `Compactor` token limit to match the model's context window.

`LocalRouter` wraps a single backing endpoint and exposes a static catalog of model metadata:

```d
import intuit;

auto router = new LocalRouter(new OpenAI("http://localhost:1234"));
router.active("gpt-4o");

LocalModelDetails details = router.details("gpt-4o");
LocalModelDetails[] withTools = router.filter(d => d.capabilities.canFind("tools"));
```

Router request functions omit the `model` parameter. When data is provided it is appended to the maintained context; when omitted, the existing context state is used:

```d
Completion result = completions(router, "Why is the sky blue?");

router.context.user("And why are sunsets red?");
Completion followUp = completions(router);
```

`streamCompletions` and `embeddings` are also available for routers, and all three throw if no active model has been set.

### Response Types

`Completion` exposes choices, token usage, and helper accessors:

```d
string text = result.text;           // first choice text
string text2 = result.text(1);     // second choice text
JSONValue json = result.json;       // parse first choice as JSON
Usage usage = result.usage;         // token accounting
```

`Choice` contains `text`, `reasoning`, `finishReason`, and `toolCalls` when tools are requested.

## License

Intuit is licensed under [AGPL-3.0](LICENSE.txt).
