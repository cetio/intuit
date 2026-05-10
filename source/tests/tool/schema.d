module tests.tool.schema;

import intuit;
import std.json : JSONValue;

string greet(string name)
{
    return "Hello, "~name~"!";
}

string multiParam(int a, string b, bool c)
{
    return b;
}

string soleJSONValue(JSONValue data)
{
    return data.toString();
}

string mixedJSONValue(string name, JSONValue extra)
{
    return name;
}

unittest
{
    ToolRegistry registry;
    registry.add!greet();

    Tool tool = registry.get("greet");
    assert(tool.name == "greet");

    JSONValue schema = tool.schema;
    assert(schema["type"].str == "object");
    assert(schema["properties"]["param0"].str == "string");
    assert(schema["required"].array.length == 1);
    assert(schema["required"][0].str == "param0");
}

unittest
{
    ToolRegistry registry;
    registry.add!multiParam();

    JSONValue schema = registry.get("multiParam").schema;
    assert(schema["properties"]["param0"].str == "integer");
    assert(schema["properties"]["param1"].str == "string");
    assert(schema["properties"]["param2"].str == "boolean");
    assert(schema["required"].array.length == 3);
}

unittest
{
    ToolRegistry registry;
    registry.add!soleJSONValue();

    JSONValue schema = registry.get("soleJSONValue").schema;
    assert("properties" !in schema || schema["properties"].object.length == 0,
        "Sole JSONValue param should not generate schema properties");
}

unittest
{
    ToolRegistry registry;
    registry.add!mixedJSONValue();

    JSONValue schema = registry.get("mixedJSONValue").schema;
    assert(schema["properties"]["param0"].str == "string");
    assert(schema["properties"]["param1"].str == "object");
    assert(schema["required"].array.length == 2);
}
