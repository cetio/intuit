module intuit.context;

import conductor.http : toJSON;
import std.json;

enum Role : string
{
    System = "system",
    User = "user",
    Assistant = "assistant",
    Tool = "tool"
}

struct Context
{
    private JSONValue _messages = JSONValue.emptyArray;

    ref Context append(T)(Role role, T data)
    {
        JSONValue msg = JSONValue.emptyObject;
        msg["role"] = JSONValue(role);
        msg["content"] = data.toJSON();
        _messages.array ~= msg;
        return this;
    }

    ref Context system(T)(T data)
    {
        return append(Role.System, data);
    }

    ref Context user(T)(T data)
    {
        return append(Role.User, data);
    }

    ref Context assistant(T)(T data)
    {
        return append(Role.Assistant, data);
    }

    ref Context tool(T)(T data)
    {
        return append(Role.Tool, data);
    }

    ref Context clear()
    {
        _messages = JSONValue.emptyArray;
        return this;
    }

    ref JSONValue messages()
        => _messages;

    size_t length() const
        => _messages.array.length;
}
