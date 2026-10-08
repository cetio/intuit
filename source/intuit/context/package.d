/// Conversation context built from typed messages with optional compaction.
module intuit.context;

public import intuit.context.message;
public import intuit.context.compactor;
public import intuit.context.policy;

import intuit.json : toJSON;
import intuit.response;
import intuit.tool : Tool, ToolRegistry;

import std.json : JSONValue;

/// Mutable conversation context that accumulates typed messages for LLM requests.
struct Context
{
    IMessage[] messages;
    /// The compactor applied after each append. Disabled if set to null.
    Compactor compactor;
    ToolPolicy toolPolicy;

    /**
     * Appends a pre-built message, compacting afterwards if a compactor is set.
     *
     * Params:
     *  message = The message to append.
     *
     * Returns:
     *  A reference to this context for chaining.
     */
    ref Context append(IMessage message)
    {
        messages ~= message;
        if (compactor !is null)
            messages = compactor.compact(messages);
        return this;
    }

    /// Appends a system message.
    ref Context system(T)(T data)
        => append(new SystemMessage(data.toJSON()));

    /// Appends a user message.
    ref Context user(T)(T data)
        => append(new UserMessage(data.toJSON()));

    /// Appends an assistant message from raw text.
    ref Context assistant(string data)
        => append(new AssistantMessage(data));

    /**
     * Appends an assistant message from text and tool calls.
     *
     * Params:
     *  data = The assistant text content.
     *  toolCalls = Tool calls issued by the assistant.
     *
     * Returns:
     *  A reference to this context for chaining.
     */
    ref Context assistant(string data, ToolCall[] toolCalls)
        => append(new AssistantMessage(data, toolCalls));

    /**
     * Appends an assistant message that wraps a completion.
     *
     * Params:
     *  completion = The completion produced by the model.
     *  index = The choice index to represent.
     *
     * Returns:
     *  A reference to this context for chaining.
     */
    ref Context assistant(Completion completion, size_t index = 0)
        => append(new AssistantMessage(completion, index));

    /// Appends a tool result message.
    ref Context tool(T)(T data)
        => append(new ToolMessage(data.toJSON()));

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
        => append(new ToolMessage(toolCallId, data.toJSON()));

    /// Serializes all messages into a JSONValue array.
    JSONValue toJSON()
    {
        JSONValue ret = JSONValue.emptyArray;
        foreach (message; messages)
            ret.array ~= message.toJSON();
        return ret;
    }

    /// Clears all messages from the context.
    ref Context clear()
    {
        messages = null;
        return this;
    }

    /// Gets the number of messages in the context.
    size_t length() const
        => messages.length;

    package(intuit) void executePolicy(ToolRegistry registry, ToolCall[] calls)
    {
        foreach (i; 0..calls.length)
        {
            calls[i].policyResult.status = ToolPolicyStatus.None;
            calls[i].policyResult.message = null;

            Tool registeredTool;
            try
                registeredTool = registry.get(calls[i].name);
            catch (Exception e)
            {
                calls[i].policyResult.status = ToolPolicyStatus.Failed;
                calls[i].policyResult.message = e.msg;
                continue;
            }

            ToolPolicyStatus decision;
            try
                decision = toolPolicy.eval(registeredTool);
            catch (Exception e)
            {
                calls[i].policyResult.status = ToolPolicyStatus.Failed;
                calls[i].policyResult.message = "Tool policy evaluation failed: "~e.msg;
                continue;
            }

            JSONValue output;
            final switch (decision)
            {
                case ToolPolicyStatus.None:
                    calls[i].policyResult.status = ToolPolicyStatus.Pending;
                    calls[i].policyResult.message = "No policy rule matched.";
                    continue;
                case ToolPolicyStatus.Pending:
                    calls[i].policyResult.status = ToolPolicyStatus.Pending;
                    calls[i].policyResult.message = "Policy evaluation is pending.";
                    continue;
                case ToolPolicyStatus.Failed:
                    calls[i].policyResult.status = ToolPolicyStatus.Failed;
                    calls[i].policyResult.message = "Tool policy failed to resolve the tool.";
                    continue;
                case ToolPolicyStatus.Allowed:
                    calls[i].policyResult.status = ToolPolicyStatus.Allowed;
                    try
                        output = registeredTool.impl(calls[i].arguments);
                    catch (Exception e)
                    {
                        calls[i].policyResult.status = ToolPolicyStatus.Failed;
                        calls[i].policyResult.message = "Tool execution failed: "~e.msg;
                        output = JSONValue(calls[i].policyResult.message);
                    }
                    break;
                case ToolPolicyStatus.Denied:
                    calls[i].policyResult.status = ToolPolicyStatus.Denied;
                    calls[i].policyResult.message = "Tool execution denied by policy.";
                    output = JSONValue(calls[i].policyResult.message);
                    break;
            }
            tool(calls[i].id, output);
        }
    }
}
