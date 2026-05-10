module tests.tool.execution;

import intuit;
import std.json : JSONValue;

string greet(string name)
{
    return "Hello, "~name~"!";
}

int add(int a, int b)
{
    return a + b;
}

string soleJSONValue(JSONValue data)
{
    return data["param0"].str;
}

string mixedJSONValue(string name, JSONValue extra)
{
    return name~":"~extra["key"].str;
}

unittest
{
    ToolRegistry registry;
    registry.add!greet();

    JSONValue args = JSONValue.emptyObject;
    args["param0"] = JSONValue("World");
    JSONValue result = registry.get("greet").impl(args);

    assert(result.str == "Hello, World!");
}

unittest
{
    ToolRegistry registry;
    registry.add!add();

    JSONValue args = JSONValue.emptyObject;
    args["param0"] = JSONValue(3);
    args["param1"] = JSONValue(4);
    JSONValue result = registry.get("add").impl(args);

    assert(result.integer == 7);
}

unittest
{
    ToolRegistry registry;
    registry.add!soleJSONValue();

    JSONValue args = JSONValue.emptyObject;
    args["param0"] = JSONValue("from whole object");
    JSONValue result = registry.get("soleJSONValue").impl(args);

    assert(result.str == "from whole object");
}

unittest
{
    ToolRegistry registry;
    registry.add!mixedJSONValue();

    JSONValue args = JSONValue.emptyObject;
    args["param0"] = JSONValue("Alice");
    args["param1"] = JSONValue.emptyObject;
    args["param1"]["key"] = JSONValue("value");
    JSONValue result = registry.get("mixedJSONValue").impl(args);

    assert(result.str == "Alice:value");
}
