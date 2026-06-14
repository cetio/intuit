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
    tool.name.should == "greet";

    JSONValue schema = tool.schema;
    schema["type"].str.should == "object";
    schema["properties"]["name"].str.should == "string";
    schema["required"].array.length.should == 1;
    schema["required"][0].str.should == "name";
}

@Name("Description UDA populates tool description")
unittest
{
    ToolRegistry registry;
    registry.add!weather();

    Tool tool = registry.get("weather");
    tool.description.should == "Looks up the current weather for a city.";
}

@Name("tool without Description UDA has empty description")
unittest
{
    ToolRegistry registry;
    registry.add!greet();

    registry.get("greet").description.length.should == 0;
}

@Name("multiple typed parameters schema")
unittest
{
    ToolRegistry registry;
    registry.add!multiParam();

    JSONValue schema = registry.get("multiParam").schema;
    schema["properties"]["a"].str.should == "integer";
    schema["properties"]["b"].str.should == "string";
    schema["properties"]["c"].str.should == "boolean";
    schema["required"].array.length.should == 3;
}

@Name("sole JSONValue parameter omits properties")
unittest
{
    ToolRegistry registry;
    registry.add!soleJSONValue();

    JSONValue schema = registry.get("soleJSONValue").schema;
    if ("properties" in schema)
        schema["properties"].object.length.should == 0;
}

@Name("mixed string and JSONValue parameters schema")
unittest
{
    ToolRegistry registry;
    registry.add!mixedJSONValue();

    JSONValue schema = registry.get("mixedJSONValue").schema;
    schema["properties"]["name"].str.should == "string";
    schema["properties"]["extra"].str.should == "object";
    schema["required"].array.length.should == 2;
}

@Name("string array parameter schema")
unittest
{
    ToolRegistry registry;
    registry.add!concat();

    JSONValue schema = registry.get("concat").schema;
    schema["properties"]["parts"]["type"].str.should == "array";
    schema["properties"]["parts"]["items"].str.should == "string";
    schema["required"].array.length.should == 1;
}

@Name("int array parameter schema")
unittest
{
    ToolRegistry registry;
    registry.add!sum();

    JSONValue schema = registry.get("sum").schema;
    schema["properties"]["nums"]["type"].str.should == "array";
    schema["properties"]["nums"]["items"].str.should == "integer";
    schema["required"].array.length.should == 1;
}
