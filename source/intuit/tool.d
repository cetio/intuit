module intuit.tool;

import conductor.http : toJSON;
import std.conv : to;
import std.json : JSONValue, JSONType;
import std.traits : Parameters, ReturnType;
import std.meta : staticMap, Filter;
import std.typecons : Tuple;

class Tool
{
    string name;
    JSONValue schema;
    JSONValue delegate(JSONValue) execute;

    this(string name, JSONValue schema, JSONValue delegate(JSONValue) execute)
    {
        this.name = name;
        this.schema = schema;
        this.execute = execute;
    }
}

struct ToolRegistry
{
    private Tool[string] tools;

    void add(alias func)()
    {
        static assert(__traits(compiles, &func), "Function must be callable");
        enum functionName = __traits(identifier, func);
        JSONValue schema = generateSchema!func();
        JSONValue delegate(JSONValue) wrapper = generateWrapper!func();
        tools[functionName] = new Tool(functionName, schema, wrapper);
    }

    void remove(string name)
    {
        tools.remove(name);
    }

    Tool get(string name)
    {
        if (name in tools)
            return tools[name];
        throw new Exception("Tool not found: " ~ name);
    }

    Tool[] list()
    {
        Tool[] result;
        foreach (tool; tools)
            result ~= tool;
        return result;
    }

private:
    JSONValue generateSchema(alias func)()
    {
        JSONValue schema = JSONValue.emptyObject;
        schema["type"] = JSONValue("object");
        JSONValue properties = JSONValue.emptyObject;
        string[] required;

        alias ParamTypes = Parameters!func;
        static foreach (i, Type; ParamTypes)
        {
            static if (is(Type == string))
                properties["arg" ~ i.to!string] = JSONValue("string");
            else static if (is(Type == int) || is(Type == long))
                properties["arg" ~ i.to!string] = JSONValue("integer");
            else static if (is(Type == float) || is(Type == double))
                properties["arg" ~ i.to!string] = JSONValue("number");
            else static if (is(Type == bool))
                properties["arg" ~ i.to!string] = JSONValue("boolean");
            else
                static assert(false, "Unsupported parameter type: " ~ Type.stringof);

            required ~= "arg" ~ i.to!string;
        }

        schema["properties"] = properties;
        JSONValue requiredArray = JSONValue.emptyArray;
        foreach (req; required)
            requiredArray.array ~= JSONValue(req);
        schema["required"] = requiredArray;

        return schema;
    }

    JSONValue delegate(JSONValue) generateWrapper(alias func)()
    {
        alias ParamTypes = Parameters!func;
        
        static if (ParamTypes.length == 0)
        {
            return delegate(JSONValue args) {
                static if (is(ReturnType!func == void))
                {
                    func();
                    return JSONValue.emptyObject;
                }
                else
                {
                    return func().toJSON();
                }
            };
        }
        else
        {
            return delegate(JSONValue args) {
                return impl!func(args);
            };
        }
    }

    JSONValue impl(alias func)(JSONValue args)
    {
        alias ParamTypes = Parameters!func;
        alias RetType = ReturnType!func;

        static if (ParamTypes.length == 0)
        {
            static if (is(RetType == void))
            {
                func();
                return JSONValue.emptyObject;
            }
            else
            {
                return func().toJSON();
            }
        }
        else
        {
            enum paramDecls = () {
                string result = "";
                static foreach (i, Type; ParamTypes)
                {
                    result ~= Type.stringof ~ " p" ~ i.to!string ~ ";";
                }
                return result;
            }();
            
            enum paramAssigns = () {
                string result = "";
                static foreach (i, Type; ParamTypes)
                {
                    result ~= "p" ~ i.to!string ~ " = extractParam!" ~ Type.stringof ~ "(args, \"arg" ~ i.to!string ~ "\");";
                }
                return result;
            }();
            
            enum callArgs = () {
                string result = "";
                static foreach (i, Type; ParamTypes)
                {
                    if (i > 0) result ~= ", ";
                    result ~= "p" ~ i.to!string;
                }
                return result;
            }();

            enum callCode = `
                ` ~ paramDecls ~ `
                ` ~ paramAssigns ~ `
                static if (is(RetType == void)) {
                    func(` ~ callArgs ~ `);
                    return JSONValue.emptyObject;
                } else {
                    return func(` ~ callArgs ~ `).toJSON();
                }
            `;

            mixin(callCode);
        }
    }

    T extractParam(T)(JSONValue args, string name)
    {
        static if (is(T == string))
            return args[name].str;
        else static if (is(T == int))
            return cast(int) args[name].integer;
        else static if (is(T == long))
            return cast(long) args[name].integer;
        else static if (is(T == float))
            return cast(float) args[name].floating;
        else static if (is(T == double))
            return args[name].floating;
        else static if (is(T == bool))
            return args[name].type == JSONType.true_;
        else
            static assert(false, "Unsupported parameter type: " ~ T.stringof);
    }
}
