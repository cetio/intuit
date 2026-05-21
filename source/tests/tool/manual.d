module tests.tool.manual;

import intuit;
import std.json : JSONValue;

string greet(string name)
{
    return "Hello, "~name~"!";
}

unittest
{
    OpenAI endpoint = new OpenAI("http://127.0.0.1:1234");
    OpenAIModel model = cast(OpenAIModel)endpoint.model("google/gemma-4-e4b");

    endpoint.tools.add!greet();

    Context ctx;
    ctx.user("Say hello to Bob");

    Completion result = completions(endpoint, model, ctx);
    assert(result.choice.toolCalls.length > 0, "Model should make a tool call");
    assert(result.choice.toolCalls[0].name == "greet", "Model should call greet");

    Tool tool = endpoint.tools.get("greet");
    JSONValue toolResult = tool.impl(result.choice.toolCalls[0].arguments);
    assert(toolResult.str == "Hello, Bob!", "Tool should execute correctly");

    ctx.tool(result.choice.toolCalls[0].id, toolResult);
    Completion finalResult = completions(endpoint, model, ctx);
    assert(finalResult.choice.text.length > 0, "Model should provide final response");
}
