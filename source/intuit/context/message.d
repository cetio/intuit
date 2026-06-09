/// Typed conversation message objects shared across contexts and models.
module intuit.context.message;

import intuit.response;
import conductor.serialize : toJSON;

import std.json : JSONValue;

/// Represents the role of a message in a conversation.
enum Role : string
{
    System = "system",
    User = "user",
    Assistant = "assistant",
    Tool = "tool"
}

/// Interface implemented by every conversation message.
interface IMessage
{
    /// Gets the role of the message.
    Role role();

    /// Serializes the message into an OpenAI-style JSON object.
    JSONValue toJSON();
}

/// A system instruction message.
class SystemMessage : IMessage
{
    private JSONValue _content;

    /**
     * Constructs a SystemMessage.
     *
     * Params:
     *  content = The message content as JSON.
     */
    this(JSONValue content)
    {
        _content = content;
    }

    override Role role()
        => Role.System;

    /// Gets the message content.
    ref JSONValue content()
        => _content;

    override JSONValue toJSON()
    {
        JSONValue ret = JSONValue.emptyObject;
        ret["role"] = JSONValue(Role.System);
        ret["content"] = _content;
        return ret;
    }
}

/// A user message.
class UserMessage : IMessage
{
    private JSONValue _content;

    /**
     * Constructs a UserMessage.
     *
     * Params:
     *  content = The message content as JSON.
     */
    this(JSONValue content)
    {
        _content = content;
    }

    override Role role()
        => Role.User;

    /// Gets the message content.
    ref JSONValue content()
        => _content;

    override JSONValue toJSON()
    {
        JSONValue ret = JSONValue.emptyObject;
        ret["role"] = JSONValue(Role.User);
        ret["content"] = _content;
        return ret;
    }
}

/// An assistant turn, wrapping the completion that produced it.
class AssistantMessage : IMessage
{
    private Completion _completion;
    private size_t _index;

    /**
     * Constructs an AssistantMessage from a completion.
     *
     * Params:
     *  completion = The completion produced by the model.
     *  index = The choice index this message represents.
     */
    this(Completion completion, size_t index = 0)
    {
        _completion = completion;
        _index = index;
    }

    /**
     * Constructs a synthetic AssistantMessage from raw text and tool calls.
     *
     * Used when no real completion is available, such as manually scripted turns.
     *
     * Params:
     *  text = The assistant text content.
     *  toolCalls = Tool calls issued by the assistant.
     */
    this(string text, ToolCall[] toolCalls = null)
    {
        Choice choice;
        choice.text = text;
        choice.toolCalls = toolCalls;
        _completion.choices = [choice];
        _index = 0;
    }

    override Role role()
        => Role.Assistant;

    /// Gets the underlying completion.
    ref Completion completion()
        => _completion;

    /// Gets the token usage reported for this turn.
    Usage usage()
        => _completion.usage;

    /// Gets the assistant text content of the selected choice.
    string text()
        => _index < _completion.choices.length ? _completion.choices[_index].text : null;

    /// Gets the tool calls issued by the selected choice.
    ToolCall[] toolCalls()
        => _index < _completion.choices.length ? _completion.choices[_index].toolCalls : null;

    override JSONValue toJSON()
    {
        JSONValue ret = JSONValue.emptyObject;
        ret["role"] = JSONValue(Role.Assistant);

        ToolCall[] calls = toolCalls();
        JSONValue content = JSONValue(text());
        if (calls.length == 0 || content.str.length > 0)
            ret["content"] = content;

        if (calls.length > 0)
        {
            JSONValue callsArray = JSONValue.emptyArray;
            foreach (call; calls)
            {
                JSONValue entry = JSONValue.emptyObject;
                entry["id"] = JSONValue(call.id);
                entry["type"] = JSONValue("function");
                JSONValue func = JSONValue.emptyObject;
                func["name"] = JSONValue(call.name);
                func["arguments"] = JSONValue(call.arguments.toString());
                entry["function"] = func;
                callsArray.array ~= entry;
            }
            ret["tool_calls"] = callsArray;
        }
        return ret;
    }
}

/// A tool result message tied to a prior tool call.
class ToolMessage : IMessage
{
    private string _toolCallId;
    private JSONValue _content;

    /**
     * Constructs a ToolMessage tied to a tool call.
     *
     * Params:
     *  toolCallId = The id of the originating tool call.
     *  content = The tool result content as JSON.
     */
    this(string toolCallId, JSONValue content)
    {
        _toolCallId = toolCallId;
        _content = content;
    }

    /**
     * Constructs a ToolMessage with no originating call id.
     *
     * Params:
     *  content = The tool result content as JSON.
     */
    this(JSONValue content)
    {
        _content = content;
    }

    override Role role()
        => Role.Tool;

    /// Gets the originating tool call id.
    ref string toolCallId()
        => _toolCallId;

    /// Gets the tool result content.
    ref JSONValue content()
        => _content;

    override JSONValue toJSON()
    {
        JSONValue ret = JSONValue.emptyObject;
        ret["role"] = JSONValue(Role.Tool);
        if (_toolCallId.length > 0)
            ret["tool_call_id"] = JSONValue(_toolCallId);
        ret["content"] = _content;
        return ret;
    }
}
