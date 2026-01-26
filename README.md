# Intuit

Intuit is a D library for interacting with language/embedding model endpoints. Intuit provides native D structures and SIMD-optimized vector operations, with a focus on a native feel.

## Features

- **Chat Completions** - Generate text completions with configurable model parameters.
- **Embeddings** - Generate vector embeddings with support for batch processing.
- **SIMD-Optimized Vector Operations** - High-performance cosine similarity, dot product, Euclidean distance, and normalization using AVX intrinsics.
- **Model Management** - Automatic model discovery and configuration.
- **Type-Safe Metaprogramming** - Leverages D's compile-time features for efficient JSON serialization and type handling.

Currently supports both OpenAI (`intuit.openai`) and Claude (`intuit.claude`) endpoints and model configuration.

## Usage

`dub add intuit`

### Endpoints

```d
import intuit.openai;

// Local LMStudio
auto endpoint = new OpenAI("http://127.0.0.1:1234/v1/");

// OpenAI API
auto openai = new OpenAI("https://api.openai.com/v1/", "your-api-key");
```

### Chat Completions

```d
import intuit.openai;
import std.json;

// Simple text completion
Completion result = endpoint.completions("gpt-4", "Explain quantum computing");
string text = result.select!string();

// With JSONValue messages array
JSONValue messages = parseJSON(`[
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What is the capital of France?"}
]`);
Completion result = endpoint.completions("gpt-4", messages);
```

### Model Configuration

```d
import intuit.openai;

// Fetch a model
Model model = endpoint.fetch("gpt-4");

// Configure model parameters
model.temperature = 0.7;
model.maxTokens = 500;
model.topP = 0.9;
model.presencePenalty = 0.1;
model.frequencyPenalty = 0.1;

// Parameters are automatically applied to requests
Completion result = endpoint.completions("gpt-4", "Hello!");
```

### Embeddings

```d
import intuit.openai;

// Single embedding
Embedding!float embedding = endpoint.embeddings("text-embedding-ada-002", "Hello, world!");
float[] vector = embedding.value;

// Batch embeddings
string[] texts = ["Hello", "World", "D Programming"];
Embedding!float[] embeddings = endpoint.embeddings("text-embedding-ada-002", texts);
```

### Vector Operations

Intuit provides SIMD-optimized vector operations for working with embeddings:

```d
import intuit.space;

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

## Roadmap

- [x] Completions
- [x] Embeddings
- [ ] Tools
- [ ] Function interop
- [x] Streaming
- [ ] Images & Vision
- [ ] Audio
- [x] OpenAI (Qwen, Deepseek, Phi, incl.)
- [ ] Gemini
- [X] Claude


## License

Intuit is licensed under [AGPL-3.0](LICENSE.txt).
