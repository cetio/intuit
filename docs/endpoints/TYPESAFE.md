# TypeSafe

`TypeSafe` sends System One decision requests. It inherits connection handling from `OpenAI`, but rejects completions and embeddings with `EndpointException`.

## Connection

```d
import intuit;

auto ep = new TypeSafe("https://api.typesafe.ai", "your-api-key");
ModelConfig cfg = ep.config("jev-latest");
```

The URL defaults to `https://api.typesafe.ai`, the key defaults to null, and the display name defaults to `"TypeSafe"`. A nonempty key uses bearer authentication.

Discovery requests `GET /v1/models` and reads names from a top-level `models` array. It adds new configurations without clearing the cache. Decisions use `POST /v1/systemone`.

Configurations are `SystemOneModelConfig` instances. Import `intuit.provider.systemone` explicitly if you need to name that subclass. For normal requests, use the common `ModelConfig` interface and the [Decisions](../DECISIONS.md) guide.

## System One Translation

The shared question types are translated before the request.

| Shared value | System One representation |
| --- | --- |
| Request input | `state`, accepting text, an object, or an array. |
| Question array | Object keyed by question name or a generated name. |
| Predicate | `noul` question and probability. |
| Choice | A `criteria` object derived from option values and descriptions. |
| Score | A `criteria` array of level labels and descriptions. |

Unnamed questions receive internal names such as `__intuit_question_0`. Avoid explicitly naming a question with a conflicting generated name.

The builder accepts at most 255 questions, 255 choices per question, and 10 score levels per question. The shared minimums and uniqueness rules still apply. Invalid input throws `FormatException`.

The parser restores request order, original question names, typed choice values, and shared probability arrays. It preserves the original provider objects in `Decision.raw` and each answer's `raw`.

Boolean and string choices remain distinct even though System One uses string keys on the wire. Responses are validated against the original questions. Keep those questions when calling the parser directly.

## Request Settings

The configuration inherits `params` for extra request fields. Required `model`, `state`, and `questions` fields are written afterward, so `params` can't override them.

Chat sampling settings, response schemas, tool selection, and completion token limits aren't serialized into decision requests. See [Models](../MODELS.md) for the shared configuration contract.