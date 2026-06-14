module tests.tool.manual;

import intuit;
import unit_threaded;

import std.json : JSONValue;

string greet(string name)
{
    return "Hello, "~name~"!";
}

@Name("manual tool call round-trip yields final text response")
unittest
{
    OpenAI endpoint = new OpenAI("http://127.0.0.1:1234");

    endpoint.tools.add!greet();

    Context ctx;
    ctx.user("Say hello to Bob");

    Completion result = completions(endpoint, "gemma-4-e4b", ctx);
    assert(result.choice.toolCalls.length > 0, "Model should make a tool call");
    assert(result.choice.toolCalls[0].name == "greet", "Model should call greet");

    Tool tool = endpoint.tools.get("greet");
    JSONValue toolArgs = JSONValue.emptyObject;
    toolArgs["name"] = JSONValue("Bob");
    JSONValue toolResult = tool.impl(toolArgs);
    assert(toolResult.str == "Hello, Bob!", "Tool should execute correctly");

    ctx.tool(result.choice.toolCalls[0].id, toolResult);
    Completion finalResult = completions(endpoint, "gemma-4-e4b", ctx);
    assert(finalResult.choice.text.length > 0, "Model should provide final response");
}
