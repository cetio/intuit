module tests.tool.autoexec;

import intuit;
import unit_threaded;

string greet(string name)
{
    return "Hello, "~name~"!";
}

@Name("auto-executed tool call recurses to final text response")
unittest
{
    OpenAI endpoint = new OpenAI("http://127.0.0.1:1234");
    endpoint.tools.add!greet(true);

    Context ctx;
    ctx.user("Say hello to Bob");

    Completion result = completions(endpoint, "gemma-4-e4b", ctx);

    assert(result.choice.text.length > 0, "Autoexec should recurse and return final text response");
    assert(ctx.length >= 3, "Context should contain user, assistant with tool call, and tool result");
}
