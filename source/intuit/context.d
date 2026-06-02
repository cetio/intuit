/// Conversation context builder and message management.
module intuit.context;

import conductor.serialize : toJSON;
import intuit.response;
import std.json;

/// Represents the role of a message in a conversation.
enum Role : string
{
    System = "system",
    User = "user",
    Assistant = "assistant",
    Tool = "tool"
}

/// Mutable conversation context that accumulates messages for LLM requests.
struct Context
{
    private JSONValue _messages = JSONValue.emptyArray;

    /**
     * Appends a message with the given role and data.
     *
     * Params:
     *  role = The role of the message.
     *  data = The message content.
     *
     * Returns:
     *  A reference to this context for chaining.
     */
    ref Context append(T)(Role role, T data)
    {
        JSONValue msg = JSONValue.emptyObject;
        msg["role"] = JSONValue(role);
        msg["content"] = data.toJSON();
        _messages.array ~= msg;
        return this;
    }

    /// Appends a system message.
    ref Context system(T)(T data)
    {
        return append(Role.System, data);
    }

    /// Appends a user message.
    ref Context user(T)(T data)
    {
        return append(Role.User, data);
    }

    /// Appends an assistant message without tool calls.
    ref Context assistant(T)(T data)
    {
        return append(Role.Assistant, data);
    }

    /**
     * Appends an assistant message with optional tool calls.
     *
     * Params:
     *  data = The assistant's text content.
     *  toolCalls = Tool calls issued by the assistant.
     *
     * Returns:
     *  A reference to this context for chaining.
     */
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
            foreach (toolCall; toolCalls)
            {
                JSONValue call = JSONValue.emptyObject;
                call["id"] = JSONValue(toolCall.id);
                call["type"] = JSONValue("function");
                JSONValue func = JSONValue.emptyObject;
                func["name"] = JSONValue(toolCall.name);
                func["arguments"] = JSONValue(toolCall.arguments.toString());
                call["function"] = func;
                callsArray.array ~= call;
            }
            msg["tool_calls"] = callsArray;
        }
        _messages.array ~= msg;
        return this;
    }

    /// Appends a tool result message.
    ref Context tool(T)(T data)
    {
        return append(Role.Tool, data);
    }

    /**
     * Appends a tool result message tied to a specific tool call.
     *
     * Params:
     *  toolCallId = The id of the tool call this result belongs to.
     *  data = The tool result content.
     *
     * Returns:
     *  A reference to this context for chaining.
     */
    ref Context tool(T)(string toolCallId, T data)
    {
        JSONValue msg = JSONValue.emptyObject;
        msg["role"] = JSONValue(Role.Tool);
        msg["tool_call_id"] = JSONValue(toolCallId);
        msg["content"] = data.toJSON();
        _messages.array ~= msg;
        return this;
    }

    /// Clears all messages from the context.
    ref Context clear()
    {
        _messages = JSONValue.emptyArray;
        return this;
    }

    /// Gets the underlying messages array.
    ref JSONValue messages()
        => _messages;

    /// Gets the number of messages in the context.
    size_t length() const
        => _messages.array.length;
}
