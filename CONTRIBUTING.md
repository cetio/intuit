# Contributing

Keep changes focused and include tests for behavior changes. Explain what changed, why it was needed, and any provider assumptions the change introduces.

## Before Making Changes

Read the relevant implementation and tests. Provider documentation isn't evidence that Intuit already supports that behavior. Check the payload builder, request route, and response parser together.

- Shared request settings live in `source/intuit/model.d`.
- Endpoints and request functions live under `source/intuit/provider/`.
- Routers and model catalogs live under `source/intuit/router/`.
- Conversation history, compaction, and policy live under `source/intuit/context/`.
- Shared response types live under `source/intuit/response/`.

See [Extending Intuit](docs/EXTENDING.md) when adding an adapter.

## Tests

Run the library build and offline tests before submitting a change.

```sh
dub build --compiler=ldc2 --config=library
dub run --compiler=ldc2 --config=unittest --build=unittest
```

Use fixed response fixtures or local HTTP servers for regression tests. Keep live provider requests behind the integration version, and don't commit keys, private prompts, or unredacted response bodies. See [Testing](TESTING.md) for integration prerequisites.

## Documentation

Update documentation when public behavior changes. Keep one authoritative explanation of each contract.

- Getting-started guides should complete a small task rather than enumerate the API.
- Shared guides own model settings, messages, responses, tools, decisions, and errors.
- Provider pages own authentication, routes, translation rules, and compatibility limits.
- Extension material belongs in the extension guide, not the first-run path.
- Use concrete sentences and small examples. Contractions are normal. Avoid em dashes and use colons and semicolons sparingly in prose.

Check examples against the code, including default values, cache behavior, exception inheritance, and tool side effects. Distinguish metadata from supported inference and requested model behavior from authorization to execute application code.

## Reporting Issues

Include a minimal example, expected and actual behavior, compiler/DUB versions, and the provider or local server involved. Supply sanitized request or response JSON when it helps reproduce a parsing problem.

For a new provider feature, describe the required wire format and how it maps to the shared types. Don't claim broad provider support from a single successful model request.