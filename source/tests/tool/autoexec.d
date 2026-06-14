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
        testOnce((endpoint, modelName) {
            endpoint.tools.add!greet(true);

            Context ctx;
            ctx.user("Say hello to Bob");

            Completion result = completions(endpoint, modelName, ctx);

            result.choice.text.length.shouldBeGreaterThan(0);
            ctx.length.shouldBeGreaterThan(2);
        });
    }
}
