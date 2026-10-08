module intuit.json;

import std.conv : to;
import std.json : JSONType, JSONValue;
import std.traits;
import std.typecons : Nullable;

struct JSONName
{
    string value;
}

struct JSONRequired { }

JSONValue toJSON(T)(T value)
{
    static if (is(T == JSONValue))
        return value;
    else static if (is(T == typeof(null)))
        return JSONValue(null);
    else static if (is(T == enum))
    {
        static if (is(OriginalType!T == string))
            return JSONValue(cast(string)value);
        else
            return JSONValue(cast(OriginalType!T)value);
    }
    else static if (is(T == string))
        return value is null ? JSONValue(null) : JSONValue(value);
    else static if (__traits(compiles, JSONValue(value)))
        return JSONValue(value);
    else static if (isStaticArray!T || isDynamicArray!T)
    {
        static if (is(T : E[], E) && (is(E == char) || is(E == immutable(char))))
            return value is null ? JSONValue(null) : JSONValue(value);
        else
        {
            JSONValue ret = JSONValue.emptyArray;
            foreach (element; value)
                ret.array ~= element.toJSON();
            return ret;
        }
    }
    else static if (isAssociativeArray!T)
    {
        static assert(is(KeyType!T == string), "Only string-keyed AAs are supported for JSON.");
        JSONValue ret = JSONValue.emptyObject;
        foreach (string key, ValueType!T element; value)
            ret[key] = element.toJSON();
        return ret;
    }
    else static if (isInstanceOf!(Nullable, T))
    {
        if (value.isNull)
            return JSONValue(null);
        return value.get.toJSON();
    }
    else static if (isAggregateType!T)
    {
        JSONValue ret = JSONValue.emptyObject;
        static foreach (string FIELD; FieldNameTuple!T)
        {{
            static if (__traits(compiles, typeof(__traits(getMember, value, FIELD))))
            {{
                alias TYPE = typeof(__traits(getMember, value, FIELD));
                static if (!is(TYPE == void) && __traits(compiles, toJSON(__traits(getMember, value, FIELD))))
                {{
                    enum NAME = JsonFieldName!(T, FIELD);
                    ret[NAME] = toJSON(__traits(getMember, value, FIELD));
                }}
            }}
        }}
        return ret;
    }
    else static if (isPointer!T)
    {
        if (value == null)
            return JSONValue(null);
        return JSONValue(cast(size_t)value);
    }
    else
        static assert(false, "Cannot serialize type "~T.stringof~" to JSON.");
}

T fromJSON(T)(JSONValue json)
{
    static if (is(T == JSONValue))
        return json;
    else static if (is(T == typeof(null)))
        return null;
    else static if (is(T == string))
    {
        if (json.type == JSONType.null_)
            return null;
        if (json.type != JSONType.string)
            throw new Exception("Expected string for "~T.stringof~", got "~json.type.to!string);
        return json.str;
    }
    else static if (is(T == bool))
    {
        if (json.type == JSONType.true_)
            return true;
        if (json.type == JSONType.false_)
            return false;
        throw new Exception("Expected bool for "~T.stringof~", got "~json.type.to!string);
    }
    else static if (isIntegral!T)
    {
        switch (json.type)
        {
        case JSONType.integer:
            return cast(T)json.integer;

        case JSONType.uinteger:
            return cast(T)json.uinteger;

        case JSONType.string:
            if (json.str is null)
                throw new Exception("Expected integral for "~T.stringof~", got null string");
            return json.str.to!T;

        default:
            throw new Exception("Expected integral for "~T.stringof~", got "~json.type.to!string);
        }
    }
    else static if (isFloatingPoint!T)
    {
        switch (json.type)
        {
        case JSONType.float_:
            return cast(T)json.floating;

        case JSONType.integer:
            return cast(T)json.integer;

        case JSONType.uinteger:
            return cast(T)json.uinteger;

        case JSONType.string:
            if (json.str is null)
                throw new Exception("Expected float for "~T.stringof~", got null string");
            return json.str.to!T;
            
        default:
            throw new Exception("Expected float for "~T.stringof~", got "~json.type.to!string);
        }
    }
    else static if (is(T == enum))
    {
        static if (is(OriginalType!T == string))
        {
            if (json.type != JSONType.string)
                throw new Exception("Expected string for enum "~T.stringof);
            static foreach (MEMBER; EnumMembers!T)
            {
                if (json.str == MEMBER)
                    return MEMBER;
            }
            throw new Exception("Invalid "~T.stringof~" value: "~json.str);
        }
        else
            return cast(T)fromJSON!(OriginalType!T)(json);
    }
    else static if (isStaticArray!T)
    {
        if (json.type != JSONType.array)
            throw new Exception("Expected array for "~T.stringof);
        if (json.array.length != T.length)
            throw new Exception("Array length mismatch for "~T.stringof);

        T ret = T.init;
        static foreach (i; 0..T.length)
            ret[i] = fromJSON!(typeof(ret[0]))(json.array[i]);
        return ret;
    }
    else static if (isDynamicArray!T)
    {
        static if (is(T : E[], E) && (is(E == char) || is(E == immutable(char))))
        {
            if (json.type == JSONType.null_)
                return null;
            if (json.type != JSONType.string)
                throw new Exception("Expected string for "~T.stringof);
            return json.str;
        }
        else
        {
            if (json.type == JSONType.null_)
                return null;
            if (json.type != JSONType.array)
                throw new Exception("Expected array for "~T.stringof);

            T ret = new typeof(T.init[0])[](json.array.length);
            foreach (i; 0..json.array.length)
                ret[i] = fromJSON!(typeof(T.init[0]))(json.array[i]);
            return ret;
        }
    }
    else static if (isAssociativeArray!T)
    {
        static assert(is(KeyType!T == string), "Only string-keyed AAs are supported for JSON.");
        if (json.type == JSONType.null_)
            return null;
        if (json.type != JSONType.object)
            throw new Exception("Expected object for "~T.stringof);

        T ret;
        foreach (string key, JSONValue element; json.object)
            ret[key] = fromJSON!(ValueType!T)(element);
        return ret;
    }
    else static if (isInstanceOf!(Nullable, T))
    {
        if (json.type == JSONType.null_)
            return T.init;
        return T(fromJSON!(TemplateArgsOf!T[0])(json));
    }
    else static if (isAggregateType!T)
    {
        if (json.type == JSONType.null_)
        {
            static if (is(T == class))
                return null;
            else
                throw new Exception("Expected object for "~T.stringof);
        }
        if (json.type != JSONType.object)
            throw new Exception("Expected object for "~T.stringof);

        static if (is(T == class))
        {
            static assert(
                __traits(compiles, new T()),
                "Cannot deserialize JSON to class "~T.stringof~" without a default constructor.",
            );
            T ret = new T();
        }
        else
            T ret = T.init;

        static foreach (string FIELD; FieldNameTuple!T)
        {{
            static if (__traits(compiles, typeof(__traits(getMember, ret, FIELD))))
            {{
                alias TYPE = typeof(__traits(getMember, ret, FIELD));
                static if (!is(TYPE == void) && __traits(compiles, fromJSON!TYPE(JSONValue.init)))
                {{
                    enum NAME = JsonFieldName!(T, FIELD);
                    enum REQUIRED = IsRequiredField!(T, FIELD);
                    if (NAME in json)
                        __traits(getMember, ret, FIELD) = fromJSON!TYPE(json[NAME]);
                    else static if (REQUIRED)
                        throw new Exception("Missing required field '"~NAME~"' in "~T.stringof);
                }}
            }}
        }}
        return ret;
    }
    else static if (isPointer!T)
    {
        if (json.type == JSONType.null_)
            return T.init;
        if (json.type == JSONType.integer)
            return cast(T)json.integer;
        if (json.type == JSONType.uinteger)
            return cast(T)json.uinteger;
        throw new Exception("Expected integer for pointer type "~T.stringof);
    }
    else
        static assert(false, "Cannot deserialize JSON to type "~T.stringof);
}

private:

template JsonFieldName(T, string FIELD)
{
    enum JsonFieldName = () {
        alias ATTRS = getUDAs!(__traits(getMember, T, FIELD), JSONName);
        static if (ATTRS.length > 0)
            return ATTRS[0].value;
        else
            return FIELD;
    }();
}

template IsRequiredField(T, string FIELD)
{
    enum IsRequiredField = hasUDA!(__traits(getMember, T, FIELD), JSONRequired);
}
