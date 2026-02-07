module intuit.context;

import intuit.utils;
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

    void append(T)(Role role, T data)
    {
        JSONValue msg = JSONValue.emptyObject;
        msg["role"] = JSONValue(role);
        msg["content"] = data.toJSON();
        _messages.array ~= msg;
    }

    void clear()
    {
        _messages = JSONValue.emptyArray;
    }

    ref JSONValue messages()
        => _messages;

    size_t length() const
        => _messages.array.length;
}
