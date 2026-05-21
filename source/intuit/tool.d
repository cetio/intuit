module intuit.tool;

import conductor.serialize : toJSON;
import std.conv : to;
import std.json : JSONValue, JSONType;
import std.traits : Parameters, ReturnType;

class Tool
{
    string name;
    JSONValue schema;
    JSONValue delegate(JSONValue) impl;
    bool autoexec;

    this(string name, JSONValue schema, JSONValue delegate(JSONValue) impl, bool autoexec = false)
    {
        this.name = name;
        this.schema = schema;
        this.impl = impl;
        this.autoexec = autoexec;
    }
}

struct ToolRegistry
{
    private Tool[string] tools;

    void add(alias F)(bool autoexec = false)
        if (__traits(compiles, &F))
    {
        tools[__traits(identifier, F)] = new Tool(
            __traits(identifier, F),
            generateSchema!F(),
            generateWrapper!F(),
            autoexec
        );
    }

    void remove(string name)
    {
        tools.remove(name);
    }

    Tool get(string name)
    {
        if (name in tools)
            return tools[name];
        throw new Exception("Tool not found: "~name);
    }

    Tool[] list()
    {
        Tool[] result;
        foreach (tool; tools)
            result ~= tool;
        return result;
    }

private:
static:
    JSONValue typeSchema(T)()
    {
        static if (is(T == JSONValue))
            return JSONValue("object");
        else static if (is(T == string))
            return JSONValue("string");
        else static if (is(T == int) || is(T == long))
            return JSONValue("integer");
        else static if (is(T == float) || is(T == double))
            return JSONValue("number");
        else static if (is(T == bool))
            return JSONValue("boolean");
        else static if (is(T : E[], E))
        {
            JSONValue arr = JSONValue.emptyObject;
            arr["type"] = JSONValue("array");
            arr["items"] = typeSchema!E();
            return arr;
        }
        else
            static assert(false, "Unsupported parameter type: "~T.stringof);
    }

    JSONValue generateSchema(alias F)()
    {
        JSONValue schema = JSONValue.emptyObject;
        schema["type"] = JSONValue("object");
        schema["properties"] = JSONValue.emptyObject;
        schema["required"] = JSONValue.emptyArray;

        alias ParamTypes = Parameters!F;
        static foreach (i, T; ParamTypes)
        {
            static if (is(T == JSONValue))
            {
                static if (ParamTypes.length > 1)
                    schema["properties"]["param"~i.to!string] = typeSchema!T();
            }
            else
                schema["properties"]["param"~i.to!string] = typeSchema!T();

            static if (!is(T == JSONValue) || ParamTypes.length > 1)
                schema["required"].array ~= JSONValue("param"~i.to!string);
        }

        return schema;
    }

    JSONValue delegate(JSONValue) generateWrapper(alias F)()
    {
        enum ParamDecl = () {
            string ret = "";
            static foreach (I, T; Parameters!F)
            {
                static if (is(T == JSONValue) && Parameters!F.length == 1)
                    ret ~= "JSONValue param"~I.to!string~" = args;";
                else static if (is(T == JSONValue))
                    ret ~= "JSONValue param"~I.to!string~" = args[\"param"~I.to!string~"\"];";
                else
                    ret ~= T.stringof~" param"~I.to!string~" = parseValue!("~T.stringof~")(args[\"param"~I.to!string~"\"]);";
            }
            return ret;
        }();

        enum ArgDecl = () {
            string ret = "";
            static foreach (I, T; Parameters!F)
            {
                ret ~= "param"~I.to!string;
                if (I < cast(ptrdiff_t)Parameters!F.length - 1)
                    ret ~= ", ";
            }
            return ret;
        }();

        static if (is(ReturnType!F == void))
        {
            mixin("return (JSONValue args) {
                "~ParamDecl~";
                F("~ArgDecl~");
                return JSONValue.emptyObject;
            };");
        }
        else
        {
            mixin("return (JSONValue args) {
                "~ParamDecl~";
                return F("~ArgDecl~").toJSON();
            };");
        }
    }

    T parseValue(T)(JSONValue value)
    {
        static if (is(T == JSONValue))
            return value;
        else static if (is(T == string))
            return value.str;
        else static if (is(T == byte) || is(T == short) || is(T == int) || is(T == long) ||
            is(T == ubyte) || is(T == ushort) || is(T == uint) || is(T == ulong))
        {
            if (value.type == JSONType.integer)
                return cast(T)value.integer;
            else
                return cast(T)value.floating;
        }
        else static if (is(T == float) || is(T == double))
        {
            if (value.type == JSONType.integer)
                return cast(T)value.integer;
            else
                return cast(T)value.floating;
        }
        else static if (is(T == bool))
            return value.type == JSONType.true_;
        else static if (is(T : E[], E))
        {
            E[] ret;
            foreach (item; value.array)
                ret ~= parseValue!E(item);
            return ret;
        }
        else
            static assert(false, "Unsupported parameter type: "~T.stringof);
    }

}
