---
layout: default
title: Decisions
nav_order: 9
permalink: /decisions/
---

# Decisions

Use `decisions` when you want typed answers to specific questions rather than a chat response. A request can estimate a probability, choose an option, or score input against labeled levels.

Decision support depends on the server and model. [OpenAI-compatible endpoints](../endpoints/openai/) use `/v1/decisions`. [TypeSafe](../endpoints/typesafe/) and [OpenRouter](../routers/openrouter/) translate questions to System One requests. Claude and Zen don't support decisions, and the catalog-only routers can't make inference requests.

## Asking a Question

This example uses TypeSafe. Replace the key and model ID with values for your account.

```d
import intuit;

import std.stdio : writeln;

auto ep = new TypeSafe("https://api.typesafe.ai", "your-api-key");

DecisionQuestion urgent;
urgent.type = DecisionType.Predicate;
urgent.name = "urgent";
urgent.instructions = "Does this ticket need immediate attention?";

Decision result = decisions(
    ep,
    "jev-latest",
    "Production is down and no customers can sign in.",
    [urgent],
);

DecisionAnswer answer = result.answer("urgent");
if (answer.type == DecisionType.Refusal)
    writeln("The model refused to answer.");
else
    writeln(answer.probability);
```

A router request omits the model argument and uses the active model.

```d
auto router = new OpenRouter("your-openrouter-key");
router.active("your-decision-model-id");

Decision routerResult = decisions(router, "A customer can't sign in.", [urgent]);
```

Decision requests don't read or append the router's conversation history. They use the input supplied to that call. Endpoint requests also leave any caller-owned context unchanged.

## Question Types

| Type | Required settings | Answer |
| --- | --- | --- |
| `Predicate` | Instructions. | `probability` between zero and one. |
| `Choice` | Instructions and at least two distinct choices. | A string or boolean in `choice`. |
| `Score` | Instructions and at least two labeled levels. | A numeric `score` from zero through the last level index. |
| `Refusal` | Not a valid request type. | The model declined to answer. |

`Predicate` is the default question type. Names are optional, but nonempty names must be unique within a request. Use names if you'll retrieve answers by name.

### Choices

```d
import std.json : JSONValue;

DecisionQuestion route;
route.type = DecisionType.Choice;
route.name = "route";
route.instructions = "Which team should handle this ticket?";
route.choices = [
    DecisionOption(JSONValue("billing"), "Payments and invoices"),
    DecisionOption(JSONValue("support"), "Technical problems"),
];
```

Choice values must be distinct strings or booleans. Boolean `true` and string `"true"` are different choices. Descriptions are optional.

### Scores

```d
DecisionQuestion severity;
severity.type = DecisionType.Score;
severity.name = "severity";
severity.instructions = "How severe is the issue?";
severity.levels = [
    DecisionLevel("Low", "A minor inconvenience"),
    DecisionLevel("High", "Work is blocked"),
];
```

Level indices start at zero. A score can be fractional, so don't assume it identifies exactly one level. Each level needs a label. Descriptions are optional.

Send several questions together by passing `[urgent, route, severity]` as the final argument.

## Reading Answers

A `Decision` exposes `answers`, `usage`, and `raw`. `answer()` returns the first answer, `answer(index)` selects by position, and `answer(name)` selects by question name. Missing names and out-of-range indices throw plain `Exception`.

Check each answer's `type` before reading its value. A refusal is an answer, not a thrown exception. Unused floating-point fields default to `NaN`.

| Answer field | Meaning |
| --- | --- |
| `name`, `type` | The question name and returned answer type. |
| `probability` | The predicate probability. |
| `choice` | The selected string or boolean as `JSONValue`. |
| `score` | The numeric score. |
| `confidence` | Optional model-reported confidence. |
| `probabilities` | Optional per-option or per-level distribution. |
| `raw` | The provider's original answer object. |

Each `DecisionProbability` has a `value` and `probability`, plus a `label` for score levels when supplied by the adapter. These are model-reported values, not a guarantee of calibration. The shared parser checks that reported probabilities are finite and between zero and one, but doesn't enforce that a distribution sums to one.

`usage` uses the shared [Usage](../responses/#usage) type. `raw` preserves the provider response, including fields not mapped into the shared types.

## Input and Validation

The base OpenAI-compatible builder accepts a string or JSON array. It sends them as `input`, with questions in an array. Use the message format required by your server when supplying an array.

System One accepts text, an object, or an array as `state`. Its additional limits and conversion rules are covered in [TypeSafe](../endpoints/typesafe/#system-one-translation).

Invalid questions throw `FormatException` before the request. Response parsing checks answer count, names, types, selected options, score ranges, and reported probabilities against the original questions. Refusals are allowed in place of the requested type. Invalid responses throw `MalformedResponseException`.

Common chat settings such as `temperature` and `maxTokens` aren't copied into decision payloads. See [Models](../models/) for extra request parameters and [Errors](../errors/) for failure handling.
