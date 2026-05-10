module intuit.context;

import conductor.http : toJSON;
import intuit.response;
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

    ref Context assistant(T)(T data, ToolCall[] toolCalls)
    {
        JSONValue msg = JSONValue.emptyObject;
        msg["role"] = JSONValue(Role.Assistant);

        JSONValue content = data.toJSON();
        if (toolCalls.length == 0 || content.type != JSONType.string || content.str.length > 0)
            msg["content"] = content;

        if (toolCalls.length > 0)
        {
            JSONValue callsArray = JSONValue.emptyArray;
            foreach (tc; toolCalls)
            {
                JSONValue call = JSONValue.emptyObject;
                call["id"] = JSONValue(tc.id);
                call["type"] = JSONValue("function");
                JSONValue func = JSONValue.emptyObject;
                func["name"] = JSONValue(tc.name);
                func["arguments"] = JSONValue(tc.arguments.toString());
                call["function"] = func;
                callsArray.array ~= call;
            }
            msg["tool_calls"] = callsArray;
        }
        _messages.array ~= msg;
        return this;
    }

    ref Context tool(T)(T data)
    {
        return append(Role.Tool, data);
    }

    ref Context tool(T)(string toolCallId, T data)
    {
        JSONValue msg = JSONValue.emptyObject;
        msg["role"] = JSONValue(Role.Tool);
        msg["tool_call_id"] = JSONValue(toolCallId);
        msg["content"] = data.toJSON();
        _messages.array ~= msg;
        return this;
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
