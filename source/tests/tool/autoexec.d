module tests.tool.autoexec;

import intuit;
import unit_threaded;

string greet(string name)
{
    return "Hello, "~name~"!";
}

version(integration)
{
    import tests.common;

    @Name("auto-executed tool call recurses to final text response") @Serial
    unittest
    {
        testOnce((endpoint, model) {
            endpoint.tools.add!greet(true);

            Context ctx;
            ctx.user("Say hello to Bob");

            Completion result = completions(endpoint, model, ctx);

            assert(result.choice.text.length > 0, "Autoexec should recurse and return final text response");
            assert(ctx.length >= 3, "Context should contain user, assistant with tool call, and tool result");
        });
    }
}
