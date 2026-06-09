module tests.tool.schema;

import intuit;
import unit_threaded;

import std.json : JSONValue;

string greet(string name)
{
    return "Hello, "~name~"!";
}

@Description("Looks up the current weather for a city.")
string weather(string city)
{
    return city;
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

string concat(string[] parts)
{
    string ret;
    foreach (part; parts)
        ret ~= part;
    return ret;
}

int sum(int[] nums)
{
    int ret;
    foreach (num; nums)
        ret += num;
    return ret;
}

@Name("single string parameter schema")
unittest
{
    ToolRegistry registry;
    registry.add!greet();

    Tool tool = registry.get("greet");
    assert(tool.name == "greet");

    JSONValue schema = tool.schema;
    assert(schema["type"].str == "object");
    assert(schema["properties"]["name"].str == "string");
    assert(schema["required"].array.length == 1);
    assert(schema["required"][0].str == "name");
}

@Name("Description UDA populates tool description")
unittest
{
    ToolRegistry registry;
    registry.add!weather();

    Tool tool = registry.get("weather");
    assert(tool.description == "Looks up the current weather for a city.");
}

@Name("tool without Description UDA has empty description")
unittest
{
    ToolRegistry registry;
    registry.add!greet();

    assert(registry.get("greet").description.length == 0);
}

@Name("multiple typed parameters schema")
unittest
{
    ToolRegistry registry;
    registry.add!multiParam();

    JSONValue schema = registry.get("multiParam").schema;
    assert(schema["properties"]["a"].str == "integer");
    assert(schema["properties"]["b"].str == "string");
    assert(schema["properties"]["c"].str == "boolean");
    assert(schema["required"].array.length == 3);
}

@Name("sole JSONValue parameter omits properties")
unittest
{
    ToolRegistry registry;
    registry.add!soleJSONValue();

    JSONValue schema = registry.get("soleJSONValue").schema;
    assert("properties" !in schema || schema["properties"].object.length == 0,
        "Sole JSONValue param should not generate schema properties");
}

@Name("mixed string and JSONValue parameters schema")
unittest
{
    ToolRegistry registry;
    registry.add!mixedJSONValue();

    JSONValue schema = registry.get("mixedJSONValue").schema;
    assert(schema["properties"]["name"].str == "string");
    assert(schema["properties"]["extra"].str == "object");
    assert(schema["required"].array.length == 2);
}

@Name("string array parameter schema")
unittest
{
    ToolRegistry registry;
    registry.add!concat();

    JSONValue schema = registry.get("concat").schema;
    assert(schema["properties"]["parts"]["type"].str == "array");
    assert(schema["properties"]["parts"]["items"].str == "string");
    assert(schema["required"].array.length == 1);
}

@Name("int array parameter schema")
unittest
{
    ToolRegistry registry;
    registry.add!sum();

    JSONValue schema = registry.get("sum").schema;
    assert(schema["properties"]["nums"]["type"].str == "array");
    assert(schema["properties"]["nums"]["items"].str == "integer");
    assert(schema["required"].array.length == 1);
}
