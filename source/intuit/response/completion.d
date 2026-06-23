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

/// Token accounting and metadata reported by the endpoint for a completion.
struct Usage
{
    /// Resolved or requested model name.
    string modelName;
    /// Total request latency in milliseconds.
    float latency;
    /// Tokens read from cache.
    size_t cacheHits;
    /// Tokens not read from cache.
    size_t cacheMisses;
    /// Tokens consumed by the prompt.
    size_t promptTokens;
    /// Tokens generated in the completion.
    size_t completionTokens;
    /// Total tokens billed for the request.
    size_t totalTokens;
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
