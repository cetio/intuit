module tests.tool;

import intuit;
import std.conv : to;
import std.json : JSONValue, JSONType;

string testAdd(int a, int b)
{
    return (a + b).to!string;
}

string testGreet(string name)
{
    return "Hello, " ~ name ~ "!";
}

int testMultiply(int a, int b)
{
    return a * b;
}

bool testIsPositive(int value)
{
    return value > 0;
}

double testDivide(double a, double b)
{
    return a / b;
}

string testCombine(string first, string second, string third)
{
    return first ~ " " ~ second ~ " " ~ third;
}

string testNoParams()
{
    return "no params";
}

unittest
{
    ToolRegistry registry;

    registry.add!testAdd();

    Tool[] tools = registry.list();
    assert(tools.length == 1);
    assert(tools[0].name == "testAdd");
}

unittest
{
    ToolRegistry registry;

    registry.add!testGreet();

    Tool tool = registry.get("testGreet");
    assert(tool.name == "testGreet");

    JSONValue args = JSONValue.emptyObject;
    args["param0"] = JSONValue("World");
    JSONValue result = tool.impl(args);
    assert(result.str == "Hello, World!");
}

unittest
{
    ToolRegistry registry;

    registry.add!testMultiply();

    Tool tool = registry.get("testMultiply");
    assert(tool.name == "testMultiply");

    JSONValue args = JSONValue.emptyObject;
    args["param0"] = JSONValue(3);
    args["param1"] = JSONValue(4);
    JSONValue result = tool.impl(args);
    assert(result.integer == 12);
}

unittest
{
    ToolRegistry registry;

    registry.add!testIsPositive();

    Tool tool = registry.get("testIsPositive");
    assert(tool.name == "testIsPositive");

    JSONValue args = JSONValue.emptyObject;
    args["param0"] = JSONValue(5);
    JSONValue result = tool.impl(args);
    assert(result.type == JSONType.true_);
}

unittest
{
    ToolRegistry registry;

    registry.add!testDivide();

    Tool tool = registry.get("testDivide");
    assert(tool.name == "testDivide");

    JSONValue args = JSONValue.emptyObject;
    args["param0"] = JSONValue(10.0);
    args["param1"] = JSONValue(2.0);
    JSONValue result = tool.impl(args);
    assert(result.floating == 5.0);
}

unittest
{
    ToolRegistry registry;

    registry.add!testCombine();

    Tool tool = registry.get("testCombine");
    assert(tool.name == "testCombine");

    JSONValue args = JSONValue.emptyObject;
    args["param0"] = JSONValue("Hello");
    args["param1"] = JSONValue("World");
    args["param2"] = JSONValue("Today");
    JSONValue result = tool.impl(args);
    assert(result.str == "Hello World Today");
}

unittest
{
    ToolRegistry registry;

    registry.add!testNoParams();

    Tool tool = registry.get("testNoParams");
    assert(tool.name == "testNoParams");

    JSONValue args = JSONValue.emptyObject;
    JSONValue result = tool.impl(args);
    assert(result.str == "no params");
}

unittest
{
    ToolRegistry registry;

    registry.add!testAdd();
    assert(registry.get("testAdd").name == "testAdd");

    registry.remove("testAdd");
    try
    {
        registry.get("testAdd");
        assert(false, "Should have thrown exception");
    }
    catch (Exception e)
    {
        // Expected
    }
}

unittest
{
    ToolRegistry registry;

    registry.add!testGreet();

    Tool[] tools = registry.list();
    assert(tools.length == 1);
    assert(tools[0].name == "testGreet");
}

unittest
{
    ToolRegistry registry;
    registry.add!testGreet();
    assert(!registry.get("testGreet").autoexec, "autoexec should default to false");
}

unittest
{
    ToolRegistry registry;
    registry.add!testGreet(true);
    assert(registry.get("testGreet").autoexec, "autoexec should be true when passed");
}

private class AutoexecModel : IModel
{
    string _name;
    int callCount;

    this(string name)
    {
        _name = name;
    }

    override ref string name()
        => _name;
    override ref string owner()
    {
        static string empty;
        return empty;
    }

    override JSONValue completionsJSON(JSONValue input, ToolRegistry tools = ToolRegistry.init)
    {
        JSONValue ret = JSONValue.emptyObject;
        ret["model"] = JSONValue(_name);
        ret["input"] = input;
        return ret;
    }

    override JSONValue embeddingsJSON(JSONValue input)
    {
        return JSONValue.emptyObject;
    }

    override Completion parseCompletions(JSONValue response)
    {
        Completion ret;
        Choice choice;

        if (callCount == 0)
        {
            choice.text = "";
            JSONValue args = JSONValue.emptyObject;
            args["param0"] = JSONValue("Bob");
            choice.toolCalls ~= ToolCall("call_1", "testGreet", args);
            callCount++;
        }
        else
        {
            choice.text = "Hello, Bob!";
        }

        ret.choices = [choice];
        return ret;
    }

    override JSONValue parseEmbeddings(JSONValue response)
    {
        return JSONValue.emptyArray;
    }
}

private class AutoexecEndpoint : IEndpoint
{
    string _name = "autoexec";
    string _url = "autoexec://";
    string _key;
    ToolRegistry _tools;

    override ref string name() => _name;
    override ref string url() => _url;
    override ref string key() => _key;
    override ref ToolRegistry tools() => _tools;

    override IModel[] available() { return null; }
    override IModel model(string name) { return new AutoexecModel(name); }

    override JSONValue _completions(IModel model, JSONValue payload)
    {
        JSONValue ret = JSONValue.emptyObject;
        ret["model"] = JSONValue("autoexec-model");
        return ret;
    }

    override JSONValue _embeddings(IModel model, JSONValue payload)
    {
        return JSONValue.emptyObject;
    }
}

unittest
{
    auto endpoint = new AutoexecEndpoint;
    auto model = cast(AutoexecModel)endpoint.model("test");

    endpoint.tools().add!testGreet(true);

    Context ctx;
    ctx.user("Say hello to Bob");

    Completion result = completions(endpoint, model, ctx);

    assert(result.choice.text == "Hello, Bob!", "Autoexec should recurse and return final text");
    assert(ctx.length >= 3, "Context should contain user, assistant with tool call, and tool messages");
}

unittest
{
    auto endpoint = new AutoexecEndpoint;
    auto model = cast(AutoexecModel)endpoint.model("test");

    endpoint.tools().add!testGreet();

    Context ctx;
    ctx.user("Say hello to Bob");

    Completion result = completions(endpoint, model, ctx);

    assert(result.choice.toolCalls.length > 0, "Non-autoexec should return tool calls");
}

unittest
{
    auto endpoint = new OpenAI("http://127.0.0.1:1234");
    auto model = cast(OpenAIModel)endpoint.model("qwen/qwen3.5-9b");

    ToolRegistry registry;
    registry.add!testGreet();

    endpoint.tools() = registry;

    Context ctx;
    ctx.user("Say hello to Bob");

    Completion result = completions(endpoint, model, ctx);

    assert(result.choice.toolCalls.length > 0, "Model should have made a tool call");
    assert(result.choice.toolCalls[0].name == "testGreet", "Tool name should be testGreet");

    Tool tool = registry.get("testGreet");
    JSONValue toolResult = tool.impl(result.choice.toolCalls[0].arguments);

    assert(toolResult.str == "Hello, Bob!", "Tool should execute correctly");

    ctx.tool(result.choice.toolCalls[0].id, toolResult);
    Completion finalResult = completions(endpoint, model, ctx);

    assert(finalResult.choice.text.length > 0, "Model should provide final response");
}
