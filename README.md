# Intuit

Intuit is a library for interacting with various AI endpoints/models, with a focus on local models. Intuit endpoints and models are highly extensible through interfaces, but primarily supports:

- OpenAI-compatible API (e.g. Ollama, Mistral, Deepseek)
- OpenAI API (e.g. ChatGPT)
- Qwen API (e.g. Qwen3, Qwen3.5)

## Features

- **Completions**: Text completions, with choices and easy parsing to D types.
- **Structured Output**: JSON schema (model options) as well as JSON parsing for responses.
- **Context Management**: System/user/assistant messages and preserved context via `Context`.
- **Streaming Support**: Real-time completions streaming (*currently not public API*)
- **Embeddings**: Embeddings with variable float sizes and SIMD math utilities.
- **Tool Use**: Native D functions as tools with automatic JSON schema generation.

## Usage

`dub add intuit`

### Endpoints

```d
import intuit.openai;

// Local LMStudio
auto endpoint = new OpenAI("http://127.0.0.1:1234");

// OpenAI API
auto openai = new OpenAI("https://api.openai.com", "your-api-key");
```

### Chat Completions

```d
import intuit.openai;
import std.json : parseJSON;

// Simple text completion
Completion result = completions(endpoint, "gpt-4o-mini", "Explain quantum computing");
string text = result.text;

// With Context
Context ctx;
ctx.system("You are a helpful assistant.")
   .user("What is the capital of France?");

Completion answer = completions(endpoint, "gpt-4o-mini", ctx);
string capital = answer.text;

// Structured JSON extraction
Completion jsonResult = completions(endpoint, "gpt-4o-mini", "Return {\"ok\":true} and nothing else.");
auto parsed = jsonResult.json;
```

### Model Configuration

```d
import intuit.openai;
import std.json : parseJSON;

auto endpoint = new OpenAI("https://api.openai.com", "your-api-key");
auto model = cast(OpenAIModel)endpoint.model("gpt-4o-mini");

// Configure model parameters
model
    .temperature(0.7)
    .maxTokens(500)
    .topP(0.9)
    .presencePenalty(0.1)
    .frequencyPenalty(0.1)
    .jsonSchema("reply", parseJSON(`{
        "type":"object",
        "properties":{"answer":{"type":"string"}},
        "required":["answer"]
    }`));

// Parameters are automatically applied to requests
Completion result = completions(endpoint, model, "Hello!");
```

### Tool Use

Intuit can register native D functions as tools and expose them to models with automatically generated JSON schemas.

```d
import intuit;

// Define functions to expose as tools
string greet(string name)
{
    return "Hello, "~name~"!";
}

int add(int a, int b)
{
    return a + b;
}

auto endpoint = new OpenAI("http://127.0.0.1:1234");
auto model = cast(OpenAIModel)endpoint.model("qwen/qwen3.5-9b");

// Register tools on the endpoint
endpoint.tools.add!greet();
endpoint.tools.add!add();

Context ctx;
ctx.user("Say hello to Bob and also compute 7 + 5");

// Manual tool execution (default)
Completion result = completions(endpoint, model, ctx);

if (result.choice.toolCalls.length > 0)
{
    foreach (call; result.choice.toolCalls)
    {
        Tool tool = endpoint.tools.get(call.name);
        JSONValue output = tool.impl(call.arguments);
        ctx.tool(call.id, output);
    }

    // Send tool results back to the model
    Completion finalResult = completions(endpoint, model, ctx);
}
```

For fully automatic tool execution, register with `autoexec = true`:

```d
endpoint.tools.add!greet(true);
endpoint.tools.add!add(true);

Context ctx;
ctx.user("Say hello to Bob and also compute 7 + 5");

// The library automatically executes tool calls and recurses until
// the model returns a plain text response.
Completion result = completions(endpoint, model, ctx);
```

### Embeddings

```d
import intuit.openai;

auto endpoint = new OpenAI("https://api.openai.com", "your-api-key");

// Single embedding
Embedding!float embedding = embeddings(endpoint, "text-embedding-3-small", "Hello, world!");
float[] vector = embedding.value;

// Batch embeddings
string[] texts = ["Hello", "World", "D Programming"];
Embedding!float[] vectors = embeddings(endpoint, "text-embedding-3-small", texts);
```

### Vector Operations

Intuit provides SIMD-optimized vector operations for working with embeddings:

```d
import intuit.response.embedding;

float[] vec1 = [1.0f, 2.0f, 3.0f];
float[] vec2 = [4.0f, 5.0f, 6.0f];

// Cosine similarity
float similarity = cosineSimilarity(vec1, vec2);

// Dot product
float dot = dotProduct(vec1, vec2);

// Euclidean distance
float distance = euclideanDistance(vec1, vec2);

// L2 norm
float norm = l2Norm(vec1);

// Normalize vector in-place
normalize(vec1);

// Normalized mean of multiple embeddings
float[][] embeddings = [[1.0f, 2.0f], [3.0f, 4.0f], [5.0f, 6.0f]];
float[] mean = normMean(embeddings);
```

## License

Intuit is licensed under [AGPL-3.0](LICENSE.txt).
