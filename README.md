# Intuit

Intuit is a small D library for interacting with OpenAI-compatible language and embedding endpoints. Intuit keeps the surface direct, JSON-native, and pragmatic, while still providing native D response types and SIMD-optimized embedding operations.

## Features

- **Chat Completions** - Generate text completions with configurable model parameters.
- **Structured Output** - Parse JSON from completion text with balanced extraction for fenced or prose-wrapped responses.
- **Embeddings** - Generate vector embeddings with support for batch processing.
- **SIMD-Optimized Vector Operations** - High-performance cosine similarity, dot product, Euclidean distance, and normalization using AVX intrinsics.
- **Model Management** - Lightweight model discovery and provider-native model configuration.
- **Type-Safe Metaprogramming** - Uses D's native types and JSON interop instead of large wrapper layers.

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
