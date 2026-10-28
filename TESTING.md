# Testing

The `unittest` configuration runs offline tests. It doesn't need provider credentials or a local model server. Some HTTP tests start a temporary loopback server.

## Prerequisites

Install DUB and a D compiler. The commands here use LDC (`ldc2`). DUB resolves the `unit-threaded` dependency and generates the test runner at `bin/ut.d`.

## Library Build

```sh
dub build --compiler=ldc2 --config=library
```

This builds the library without enabling unit tests.

## Offline Tests

```sh
dub run --compiler=ldc2 --config=unittest --build=unittest
```

Use this configuration for routine verification. It covers JSON conversion, model payloads and parsing, decision validation, context messages and compaction, policy decisions, tool schemas and wrappers, catalog-only restrictions, and HTTP transport behavior.

Passing tests doesn't establish live provider compatibility. In particular, parser tests and policy-decision tests don't prove a complete provider tool round-trip.

## Integration Tests

The `integration` configuration additionally enables code guarded by `version(integration)`. The `unittest` configuration makes no external requests.

```sh
dub run --compiler=ldc2 --config=integration --build=unittest
```

### Configuration

Live tests read `INTUIT_TEST_*` settings and conventional provider keys from the process environment. The library never loads dotenv files. Export variables directly, or source a trusted file:

```sh
set -a; . ./.env; set +a
```

See `.env.example` for every supported knob; it holds placeholders, not real keys. `INTUIT_TEST_ALLOW_PAID` defaults to free-only inference: `0`, unset, or empty keeps free mode, only the literal `1` enables paid inference, and any other value is a configuration error. `INTUIT_TEST_MAX_TOKENS` (256) and `INTUIT_TEST_TIMEOUT_SECONDS` (90) bound every request.

### Free defaults

- Models.dev and OpenCode Zen catalogs are public and keyless. Zen's default model is `nemotron-3.5-lightning-free`, but [OpenCode's inference docs](https://opencode.ai/v2/docs/console/api/inference) restrict free chat inference to OpenCode clients; the SDK integration test explicitly skips those free inference calls. To test Zen inference from this SDK, choose a paid model and set `ZEN_API_KEY` plus `INTUIT_TEST_ALLOW_PAID=1`.
- OpenRouter defaults are catalog-verified zero-price models: `nvidia/nemotron-3.5-lightning:free` for completions, `cohere/north-mini-code:free` for tools, and `respan/span-01-lite:free` for decisions. The free Respan decision model only accepts Noul/predicate questions, so its live smoke test checks one predicate; offline tests cover all question types. Free mode verifies exact id, modality, and zero prompt/completion pricing in the live catalog before requesting, and pins `provider.max_price` to zero. Set `INTUIT_TEST_OPENROUTER_EMBEDDING_MODEL` (for example `nvidia/nemotron-3-embed-1b:free`) to enable embedding smoke tests.
- Paid models such as `inception/mercury-decide` ($0.02/M input tokens) never run without your explicit model choice plus `INTUIT_TEST_ALLOW_PAID=1`.

Direct providers run only for explicitly configured `INTUIT_TEST_{OPENAI,QWEN,CLAUDE,TYPESAFE}_URL` plus the matching `*_MODEL` (TypeSafe uses `*_DECISION_MODEL`). OpenAI and Qwen append `/v1` to the base URL, so point them at a server root such as `http://127.0.0.1:1234`. To reach OpenRouter through the OpenAI-compatible path, use exactly `https://openrouter.ai/api`; that canonical URL activates the same catalog zero-price guard. Keys are only sent to the provider they name, and any other remote URL is your cost responsibility. Claude covers completions and model discovery only. LiteLLM runs a catalog check when `INTUIT_TEST_LITELLM_URL` is set.

The OpenAI-compatible `/v1/decisions` route is not OpenRouter's `/api/alpha/decisions`; do not point a direct provider at OpenRouter for decision tests.

### Filtering and skips

Runner arguments filter by test path: `tests.integration.routers` selects that package, a longer module prefix narrows further, and `-s` runs in one thread. Missing optional configuration prints `SKIP <scenario>: missing <ENV_NAME>` to stderr and returns; the runner captures stdout during passing tests, so skips go to stderr to stay visible. These helper-reported skips count as passes because the runner has no runtime skip mechanism. The known OpenCode free-tier restriction is also reported as a skip because third-party SDK calls are not supported for those models. Other configured scenarios that fail (auth, rate limit, transport, or parse errors) fail the test with sanitized diagnostics; nothing is retried, substituted, or downgraded to a skip. Free tiers can be rate-limited or removed by the service, so a failure may reflect service state rather than a library defect.

## Test Layout

| Path | Coverage |
| --- | --- |
| `source/tests/json.d` | Serialization, deserialization, and required fields. |
| `source/tests/model.d` | Configuration dispatch, completion parsing, and embedding conversion. |
| `source/tests/decision.d` | Shared decisions and System One translation. |
| `source/tests/context/` | Messages, compaction, and policy evaluation. |
| `source/tests/provider/` | Provider payloads, parsers, HTTP errors, and timeouts. |
| `source/tests/router/` | Catalog-only restrictions. |
| `source/tests/tool/` | Schema generation and invocation wrappers. |
| `source/tests/integration/config.d` | Offline regressions for settings parsing, skip selection, and free-price policy. |
| `source/tests/integration/routers/` | Live catalog, completion, tool, decision, and embedding router tests. |
| `source/tests/integration/providers/` | Live model listing, completion, tool, decision, and embedding provider tests. |