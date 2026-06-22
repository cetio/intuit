/// Completion response types.
module intuit.response.completion;

import std.json : JSONValue, JSONType, parseJSON;

/// Reason why a completion generation finished.
enum FinishReason : string
{
    Missing = "missing",
    Length = "length",
    Max_Tokens = "max_tokens",
    ContentFilter = "content_filter",
    Refusal = "refusal",
    ToolCall = "tool_call",
    ToolUse = "tool_use",
    FunctionCall = "function_call",
    Pause = "pause",
    PauseTurn = "pause_turn",
    Stop = "stop",
    EndTurn = "end_turn",
    StopSequence = "stop_sequence",
    Unknown = "unknown"
}

/// Represents a single tool call requested by the model.
struct ToolCall
{
    /// Unique identifier for the tool call.
    string id;
    /// Name of the tool to invoke.
    string name;
    /// Arguments to pass to the tool as JSON.
    JSONValue arguments;
}

/// A single choice within a completion response.
struct Choice
{
    /// The raw JSON value for this choice.
    JSONValue raw;
    /// The raw content JSON value.
    JSONValue content;
    /// Extracted text content.
    string text;
    /// Extracted reasoning content.
    string reasoning;
    /// Why generation stopped for this choice.
    FinishReason finishReason;
    /// Optional log probability data.
    JSONValue logProbs;
    /// Tool calls requested in this choice.
    ToolCall[] toolCalls;
}

/// Token accounting reported by the endpoint for a completion.
struct Usage
{
    /// Tokens consumed by the prompt.
    size_t promptTokens;
    /// Tokens generated in the completion.
    size_t completionTokens;
    /// Total tokens billed for the request.
    size_t totalTokens;
}

/**
 * Parses token usage from a raw completion response.
 *
 * Accepts both OpenAI-style (`prompt_tokens`/`completion_tokens`) and
 * Anthropic-style (`input_tokens`/`output_tokens`) field names. The total
 * is derived from the prompt and completion counts when not reported.
 *
 * Params:
 *  json = The raw JSON response.
 *
 * Returns:
 *  The parsed Usage, zeroed when no usage block is present.
 */
Usage parseUsage(JSONValue json)
{
    Usage ret;
    if ("usage" !in json || json["usage"].type != JSONType.object)
        return ret;

    JSONValue usage = json["usage"];
    ret.promptTokens = usageField(usage, "prompt_tokens", "input_tokens");
    ret.completionTokens = usageField(usage, "completion_tokens", "output_tokens");
    ret.totalTokens = usageField(usage, "total_tokens", null);
    if (ret.totalTokens == 0)
        ret.totalTokens = ret.promptTokens + ret.completionTokens;
    return ret;
}

/// Reads an integral token count from a usage object under either key.
private size_t usageField(JSONValue usage, string primary, string fallback)
{
    foreach (key; [primary, fallback])
    {
        if (key is null || key !in usage)
            continue;

        JSONValue value = usage[key];
        if (value.type == JSONType.integer)
            return cast(size_t)value.integer;
        if (value.type == JSONType.uinteger)
            return cast(size_t)value.uinteger;
    }
    return 0;
}

/// Parsed result from a completions request.
struct Completion
{
    /// The raw JSON response.
    JSONValue raw;
    /// All generated choices.
    Choice[] choices;
    /// Token accounting for the request, when reported by the endpoint.
    Usage usage;

    /**
     * Gets a choice by index.
     *
     * Params:
     *  index = The choice index.
     *
     * Returns:
     *  The Choice at the given index.
     *
     * Throws:
     *  Exception if the index is out of range.
     */
    ref inout(Choice) choice(size_t index = 0) inout
    {
        if (index >= choices.length)
            throw new Exception("Completion choice index is out of range.");
        return choices[index];
    }

    /// Gets the text of the first choice.
    string text() const
        => text(0);

    /// Gets the text of the choice at `index`.
    string text(size_t index = 0) const
        => choice(index).text;

    /// Gets the reasoning text of the first choice.
    string reasoning() const
        => reasoning(0);

    /// Gets the reasoning text of the choice at `index`.
    string reasoning(size_t index = 0) const
        => choice(index).reasoning;

    /// Parses the first choice's text as JSON.
    JSONValue json() const
        => json(0);

    /// Parses the choice at `index`'s text as JSON.
    JSONValue json(size_t index = 0) const
        => text(index).parseJSON();
}
