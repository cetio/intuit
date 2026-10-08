# Tools

Intuit turns native D functions into model-visible tools. `ToolRegistry` derives a JSON schema from function parameters and stores a delegate that converts JSON arguments, invokes the function, and converts its return value to JSON.

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

Every generated parameter is required. Parameter names come from the function declaration, so compile with parameter identifiers available.

Return values are converted with Intuit's `toJSON`. A `void` function returns an empty JSON object from its wrapper.

```d
int add(int left, int right)
{
    return left + right;
}

ep.tools.add!add();
```

A function whose only parameter is `JSONValue` receives the complete arguments object directly. A `JSONValue` parameter in a multi-parameter function receives its named member.

```d
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
Tool tool = ep.tools.get("getWeather");
writeln(tool.schema.toPrettyString());
```

## Controlling Tool Choice

Configure tool choice on the model configuration:

```d
ModelConfig cfg = ep.config("llama3");
cfg.setToolLiability("auto");
```

| Value | Behavior |
| --- | --- |
| `"auto"` | The model may call zero or more tools. |
| `"required"` | The model must call one or more tools. |
| `"none"` | The model cannot call tools. |

Force one tool with `setRequiredTool`:

```d
cfg.setRequiredTool(ep.tools.get("getWeather"));
```

Passing `null` clears the tool choice. Provider and model support varies; Intuit serializes the requested setting but cannot force an incompatible model to honor it.

## Tool Execution Policy

A `Context` has a `toolPolicy` that can allow or deny named tools or delegate decisions:

```d
router.context.toolPolicy.allow("getWeather");
router.context.toolPolicy.deny("deleteFile");

router.context.toolPolicy.evaluator = (Tool tool) {
    if (tool.name == "lookupAccount")
        return ToolPolicyStatus.Allowed;
    return ToolPolicyStatus.Pending;
};
```

A deny-list match takes precedence over an allow-list match; list matches take precedence over the evaluator. If no list matches and the evaluator is null, `eval` returns `ToolPolicyStatus.None`.

Context-backed router and endpoint completions process tool batches on a best-effort basis. `Allowed` tools run and append their result; `Denied` tools are not run and append a policy-rejection result. Unregistered tools and policy/execution errors are marked `Failed`. A `None` policy decision is exposed as `Pending` on the `ToolCall`, as is an evaluator that explicitly returns `Pending`. Inspect each call's `policyResult.status` and `policyResult.message`. If any call is `Pending` or `Failed`, the batch is still fully evaluated, then completion returns so the caller can resolve those calls manually. When automatic policy rounds finish, inspect the stored assistant message in the context to review prior call results.

Context-backed completions accept `maxToolRounds` (default `8`) to limit automatic policy-managed tool rounds. For example, use `completions(router, "Question", 3)` or `completions(ep, "model", context, 3)`. Setting it to zero skips policy execution and marks returned calls `Pending`. Reaching the round limit returns the last tool-call completion with its per-call results attached.

## Executing Tool Calls

Without a policy decision, Intuit returns requested calls for application code to handle. Look up each call by name, invoke its wrapper with the parsed arguments, and append the result to the context:

```d
import intuit;

@Description("Greet someone by name.")
string greet(string name)
{
    return "Hello, "~name~"!";
}

OpenAI ep = new OpenAI("http://localhost:1234");
ep.tools.add!greet();
ep.config("llama3").setToolLiability("auto");

Context ctx;
ctx.user("Say hello to Bob.");

Completion result = completions(ep, "llama3", ctx);
foreach (call; result.choice.toolCalls)
{
    Tool tool = ep.tools.get(call.name);
    JSONValue output = tool.impl(call.arguments);
    ctx.tool(call.id, output);
}

Completion finalResult = completions(ep, "llama3", ctx);
```

Passing `ctx` to the first completion automatically appends an `AssistantMessage` containing the tool calls. Each `ctx.tool(call.id, output)` then creates the matching tool-result message. The final request therefore contains the assistant calls and their results.

Validate tool names and arguments before invoking side-effecting application code. `get(name)` throws a plain `Exception` when a model requests an unregistered name, and argument conversion throws when values do not match the D signature.

## Provider Shapes

`ModelConfig` emits OpenAI-compatible function tools:

- top-level tool type `function`
- function `name`, optional `description`, and `parameters`
- tool choice as a string or named function object

`ClaudeModelConfig` translates definitions to Anthropic's shape with `name`, optional `description`, and `input_schema`. Claude `tool_use` blocks are translated back to shared `ToolCall` values.

`QwenModelConfig` accepts native OpenAI-compatible calls and also extracts supported Qwen custom XML and Hermes JSON-in-XML tool calls. Synthetic IDs are assigned when the source format does not provide one.

## Schema Limitations

Generated schemas currently describe only the supported primitive and array types. They do not infer optional parameters, default values, nested structs, enums, constraints, or per-parameter descriptions. Construct a `Tool` directly when a richer schema or custom invocation delegate is required:

```d
JSONValue schema = JSONValue.emptyObject;
schema["type"] = JSONValue("object");
schema["properties"] = JSONValue.emptyObject;

Tool custom = new Tool(
    "custom",
    "Handle a custom payload.",
    schema,
    (JSONValue arguments) {
        return arguments;
    }
);
```

`ToolRegistry` does not currently expose an overload for adding a pre-built `Tool`; direct construction is useful for custom registries or future endpoint implementations, while the public registry API registers functions through `add`.
