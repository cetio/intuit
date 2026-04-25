module checks.context;

import intuit;

unittest
{
    Context ctx;
    auto ref chained = ctx
        .system("system")
        .user("user")
        .assistant("assistant")
        .tool("tool");

    assert(chained.length == 4);
    assert(ctx.messages[0]["role"].str == "system");
    assert(ctx.messages[1]["role"].str == "user");
    assert(ctx.messages[2]["role"].str == "assistant");
    assert(ctx.messages[3]["role"].str == "tool");
}

unittest
{
    Context ctx;
    ctx.system("hello").clear();
    assert(ctx.length == 0);
}
