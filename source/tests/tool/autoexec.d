module tests.tool.autoexec;

import intuit;

string greet(string name)
{
    return "Hello, "~name~"!";
}

unittest
{
    OpenAI endpoint = new OpenAI("http://127.0.0.1:1234");
    OpenAIModel model = cast(OpenAIModel)endpoint.model("qwen/qwen3.5-9b");

    endpoint.tools.add!greet(true);

    Context ctx;
    ctx.user("Say hello to Bob");

    Completion result = completions(endpoint, model, ctx);

    assert(result.choice.text.length > 0, "Autoexec should recurse and return final text response");
    assert(ctx.length >= 3, "Context should contain user, assistant with tool call, and tool result");
}
