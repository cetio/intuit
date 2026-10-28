---
layout: default
title: Tools
nav_order: 8
permalink: /tools/
---

# Tools

Register D functions so a model can request them as tools. Intuit derives a schema from the signature and stores a wrapper that converts arguments and return values to JSON.

Requesting a tool and authorizing its execution are separate decisions. Context-backed completions can execute calls through a policy, or return them for manual handling.

## Registering a Function

```d
import intuit;

@Description("Get the current weather for a location.")
string getWeather(string location)
{
    return "Sunny and 72 degrees in "~location;
}

auto ep = new OpenAI("http://localhost:1234");
ep.tools.add!getWeather();
```

The tool name is the D function identifier. `Description` is optional but helps the model decide when to call the tool.

Endpoints and routers each own an independent registry through their `tools` property. Every registered tool is included in completion request payloads.

Generated schemas currently use primitive shorthand nodes. If your server requires standard JSON Schema, replace the registered schema as shown in [Schema Limitations](#schema-limitations) before sending a request.

## Supported Signatures

Schema generation supports these parameter types:

| D type | JSON Schema type |
| --- | --- |
| `string` | `string` |
| `int`, `long` | `integer` |
| `float`, `double` | `number` |
| `bool` | `boolean` |
| `T[]` | `array` with the corresponding item schema |
| `JSONValue` | `object` |

Every named generated parameter is required. Names come from the declaration, with `param0`, `param1`, and so on as fallbacks when identifiers aren't available. Default D arguments aren't inferred as optional.

Return values are converted with Intuit's `toJSON`. A `void` function returns an empty JSON object from its wrapper.

```d
int add(int left, int right)
{
    return left + right;
}

ep.tools.add!add();
```

A function whose only parameter is `JSONValue` receives the complete arguments object directly and has no generated named properties. A `JSONValue` parameter in a multi-parameter function receives its named member.

```d
import std.json : JSONValue;

JSONValue inspect(JSONValue arguments)
{
    return arguments;
}
```

Unsupported parameter types fail at compile time when the tool is registered.

## Inspecting the Registry

| Member | Purpose |
| --- | --- |
| `add!function()` | Register or replace a function by name. |
| `get(name)` | Return a registered `Tool`, or throw if absent. |
| `list()` | Return all registered tools. |
| `remove(name)` | Remove a tool by name. |

A `Tool` exposes its `name`, `description`, generated `schema`, and `impl` delegate:

```d
import std.stdio : writeln;

Tool registeredTool = ep.tools.get("getWeather");
writeln(registeredTool.schema.toPrettyString());
```

## Controlling Tool Choice

Use [model tool-choice settings](../models/#tool-choice) to request automatic selection, require a tool, or disable tool requests. These settings affect what the model returns. They don't grant execution permission.

## Manual Execution

Set `maxToolRounds` to zero to disable policy-managed execution. This example handles one returned batch manually, then sends the results back.

```d
import intuit;

import std.exception : enforce;
import std.json : JSONValue;

@Description("Greet someone by name.")
string greet(string name)
{
    return "Hello, "~name~"!";
}

OpenAI manualEndpoint = new OpenAI("http://localhost:1234");
manualEndpoint.tools.add!greet();
manualEndpoint.config("llama3").setToolLiability("auto");

Context ctx;
ctx.user("Say hello to Bob.");

Completion result = completions(manualEndpoint, "llama3", ctx, 0);
if (result.choices.length > 0 && result.choice.toolCalls.length > 0)
{
    foreach (call; result.choice.toolCalls)
    {
        enforce(call.name == "greet", "Unexpected tool requested.");
        Tool tool = manualEndpoint.tools.get(call.name);
        JSONValue output = tool.impl(call.arguments);
        ctx.tool(call.id, output);
    }
    result = completions(manualEndpoint, "llama3", ctx, 0);
}
```

The first completion already appends the assistant's calls to `ctx`. Each `ctx.tool(call.id, output)` adds the matching result. Don't append the assistant call a second time.

The follow-up can request another batch. Inspect its calls before treating its text as a final answer, and bound any application-managed loop.

This example's function is harmless. For application tools, validate arguments and authorize side effects before calling `impl`. Calling the wrapper directly bypasses `toolPolicy`. Missing tool names and argument conversion failures throw.

## Policy-Managed Execution

Every context has a `toolPolicy`. It defaults to no rules, so a registered call becomes `Pending` instead of running. A call to an unregistered name becomes `Failed`. Add allowed names to execute registered functions automatically.

```d
Context automaticContext;
automaticContext.toolPolicy.allow("greet");
automaticContext.toolPolicy.deny("deleteFile");
automaticContext.user("Say hello to Bob.");

Completion automatic = completions(manualEndpoint, "llama3", automaticContext, 3);
```

For a router, configure `router.context.toolPolicy` and call `completions(router, prompt, 3)`. Only context-backed completions run policy evaluation. An endpoint request with a plain string or raw JSON returns tool calls without executing them.

### Policy Decisions

Deny-list matches take precedence over allow-list matches. If neither matches, Intuit calls the optional evaluator. Without an evaluator, `eval()` returns `None`, which completion handling exposes as `Pending`.

```d
automaticContext.toolPolicy.evaluator = (Tool tool) {
    return tool.name == "greet"
        ? ToolPolicyStatus.Allowed
        : ToolPolicyStatus.Pending;
};
```

The evaluator receives the registered `Tool`, not the call arguments. Use manual handling when approval depends on the supplied values.

### Returned Outcomes

Inspect `call.policyResult.status` and `call.policyResult.message`.

| Status | What happened |
| --- | --- |
| `Allowed` | The function ran and its result was appended. |
| `Denied` | The function didn't run. A rejection result was appended. |
| `Pending` | No result was appended. The call needs manual resolution. |
| `Failed` | Lookup, policy evaluation, or execution failed. An execution failure appends an error result, while lookup and evaluator failures don't. |

Intuit evaluates the entire batch even if a call is pending or fails. It then returns that completion if any calls need attention. Allowed calls in the same batch may already have run. Don't replay every call or append results that already exist.

For failed calls, inspect the message and context before retrying. A function can perform a side effect and then throw. Denied calls should remain denied unless your application deliberately changes that decision.

### Round Limits

`maxToolRounds` defaults to `8`. It limits automatically handled tool batches, not individual calls or total elapsed time.

When all calls in a batch are resolved and another round is available, Intuit sends the updated history again. At the limit it returns the last tool-call completion after evaluating that batch, without requesting a final assistant answer. Setting the limit to zero skips policy evaluation and marks returned calls `Pending`. Negative values throw.

Automatic rounds are kept in the context. Inspect stored `AssistantMessage` values to review earlier calls and their outcomes. See [Errors](../errors/#retrying-requests) before retrying a failed conversation.

## Provider Formats

`ModelConfig` emits OpenAI-compatible function tools:

- top-level tool type `function`
- function `name`, optional `description`, and `parameters`
- tool choice as a string or named function object

`ClaudeModelConfig` translates definitions to Anthropic's shape with `name`, optional `description`, and `input_schema`. It parses `tool_use` blocks into shared calls, but doesn't translate shared tool history back into Anthropic request blocks. See the [Anthropic limitations](../endpoints/anthropic/#payload-translation).

`QwenModelConfig` accepts native OpenAI-compatible calls and also extracts supported Qwen custom XML and Hermes JSON-in-XML tool calls. Extracted calls receive synthetic IDs.

## Schema Limitations

Generated schemas describe the supported primitive and array types. They don't infer optional parameters, defaults, nested structs, enums, constraints, or per-parameter descriptions. Primitive property and array-item nodes use shorthand strings such as `"string"` rather than full `{ "type": "string" }` nodes.

For a server that requires standard JSON Schema, replace the registered tool's schema with an explicit one. The registry returns the same mutable `Tool` instance used to build requests.

```d
import std.json : parseJSON;

ep.tools.get("getWeather").schema = parseJSON(`{
    "type": "object",
    "properties": {"location": {"type": "string"}},
    "required": ["location"],
    "additionalProperties": false
}`);
```

`ToolRegistry` doesn't accept a pre-built `Tool`. You can customize a registered tool's `schema`, `description`, or `impl`, but keeping them consistent is your responsibility. A function taking one `JSONValue` can handle a custom argument object and perform its own validation.
