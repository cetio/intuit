---
layout: default
title: Errors
nav_order: 10
permalink: /errors/
---

# Errors

Intuit distinguishes failed requests from responses it can't parse. Catching `EndpointException` handles HTTP and transport failures, but it doesn't catch `MalformedResponseException` or `FormatException`.

## Exception Types

| Type | Parent | When it's used |
| --- | --- | --- |
| `EndpointException` | `Exception` | Non-success HTTP responses, provider error envelopes, and some unsupported operations. |
| `AuthException` | `EndpointException` | HTTP 401 or 403. |
| `RateLimitException` | `EndpointException` | HTTP 429. |
| `TransportException` | `EndpointException` | A curl transport failure, such as a failed connection. |
| `RequestTimeoutException` | `TransportException` | A curl timeout. |
| `MalformedResponseException` | `Exception` | Invalid HTTP response JSON, invalid completion JSON text, or malformed decision answers. |
| `FormatException` | `Exception` | Invalid request configuration, such as an unsupported tool-choice value or invalid decision questions. |

Not every error has a specialized type. An unset active model, a missing registry entry, an out-of-range choice or answer, and unsupported catalog-only router operations throw plain `Exception`. Argument conversion and provider-field parsing can also propagate exceptions from JSON or conversion code.

## Catching Failures

Catch subclasses before their parent when you need different handling. This example handles request and parsing failures separately.

```d
import intuit;

import std.stdio : stderr, writeln;

auto ep = new OpenAI("http://localhost:1234");

try
{
    Completion result = completions(ep, "llama3", "Say hello.");
    writeln(result.text);
}
catch (AuthException error)
{
    stderr.writeln("Check the API key and permissions.");
}
catch (RateLimitException error)
{
    stderr.writeln("The provider rejected the request with HTTP 429.");
}
catch (RequestTimeoutException error)
{
    stderr.writeln("The request timed out.");
}
catch (TransportException error)
{
    stderr.writeln("The server couldn't be reached.");
}
catch (EndpointException error)
{
    stderr.writeln(error.status, " ", error.reason);
}
catch (MalformedResponseException error)
{
    stderr.writeln("The response couldn't be parsed.");
}
```

Validate locally constructed settings separately when handling `FormatException`. The example doesn't catch every possible application or conversion error.

## Inspecting an Error

`EndpointException` exposes `method`, `route`, `status`, `reason`, and `content`. For HTTP failures, `route` is the requested URL and `content` is the response body.

A status of zero means no HTTP status is represented by that exception. Transport failures use zero, as do provider errors detected while parsing an otherwise successful response and locally rejected operations. Parser-generated routes may be labels such as `chat/completions`, not full URLs.

`TransportException.detail` contains the transport error description. `MalformedResponseException` exposes `rawText` and `candidateText`. For `Completion.json`, those hold the serialized response and the choice text that failed to parse.

Error messages and bodies can contain prompts, tool arguments, or provider details. Redact them before sharing logs. Rate-limit errors don't expose response headers or a dedicated retry delay.

## Timeouts

Endpoints and routers both expose connection and operation timeouts.

```d
import core.time : seconds;

ep.connectTimeout(5.seconds);
ep.operationTimeout(60.seconds);
```

An operation timeout applies to one HTTP operation, not an entire policy-managed tool conversation. Model activation may fetch a catalog before the completion request. Intuit doesn't set these limits for you or retry failed requests automatically.

## Retrying Requests

Decide whether a retry is safe before adding backoff. Authentication and request-format failures generally need a configuration change. A timeout doesn't establish whether the server processed the request, and repeating inference can incur another charge.

Conversation state can change before a request fails. Router completion input is appended before the HTTP request, and earlier policy rounds may already have appended assistant messages and tool results. Tool functions may have performed side effects.

Inspect the context before retrying. Don't blindly append the same user message again or rerun every tool call. See [Tools](../tools/#policy-managed-execution) for partial batches and [Context](../context/) for history management.
