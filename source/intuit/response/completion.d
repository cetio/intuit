/// Completion response types and streaming support.
module intuit.response.completion;

import core.atomic;
import core.thread;
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

/**
 * Thread-safe completion stream consumer.
 *
 * This is the frontend for all Server-Sent Events (SSE) streaming
 * completions. Endpoint implementations feed parsed SSE chunks into
 * a CompletionStream, and callers consume them via `next()`, `collect()`,
 * or the per-chunk `callback` delegate.
 */
class CompletionStream
{
private:
    Completion[] _completions;
    shared size_t _length;
    shared size_t _index;
    shared bool _writer;

public:
    /// Internal delegate to start the stream.
    void delegate(CompletionStream) _commence;

    /// Accumulated metadata JSON.
    JSONValue json;
    /// Model name for the stream.
    string model;
    /// True when the stream has finished.
    bool complete;
    /// If set, an exception thrown by the background worker.
    Exception error;
    /// Callback invoked for each completion chunk.
    void delegate(Completion) callback;

    /**
     * Constructs a CompletionStream.
     *
     * Params:
     *  model = The model name.
     *  callback = Delegate called for each chunk.
     */
    this(string model, void delegate(Completion) callback)
    {
        this.model = model;
        this.callback = callback;
        this._completions = null;
        this._length = 0;
        this._index = 0;
        this._writer = false;
        this.complete = false;
        this.json = JSONValue.emptyObject;
    }

    /**
     * Gets the next available completion chunk, blocking until available.
     *
     * Returns:
     *  The next Completion chunk.
     *
     * Throws:
     *  Exception if the stream has not been initialized.
     */
    Completion next()
    {
        if (_commence is null)
            throw new Exception("Stream not initialized");

        if (error !is null)
            throw error;

        while (atomicLoad!(MemoryOrder.acq)(_writer))
            Thread.yield();

        size_t cur = atomicFetchAdd!(MemoryOrder.seq)(_index, 1);

        while (cur >= atomicLoad!(MemoryOrder.acq)(_length))
        {
            if (complete)
                throw new Exception("Stream exhausted");
            Thread.yield();
        }

        atomicFence!(MemoryOrder.acq);
        return _completions[cur];
    }

    /**
     * Collects `count` completion chunks.
     *
     * Params:
     *  count = Number of chunks to collect.
     *
     * Returns:
     *  An array of collected Completion chunks.
     */
    Completion[] collect(size_t count)
    {
        Completion[] ret;
        foreach (i; 0..count)
            ret ~= next();
        return ret;
    }

    /// Begins streaming using the internal commence delegate.
    void begin()
    {
        _commence(this);
    }

    /**
     * Sets the commence delegate and starts streaming.
     *
     * Params:
     *  cb = The delegate that drives the stream.
     */
    void commence(void delegate(CompletionStream) cb)
    {
        _commence = cb;
        begin();
    }

    /**
     * Pushes a new completion chunk into the stream.
     *
     * Params:
     *  val = The completion chunk to append.
     */
    void update(Completion val)
    {
        atomicStore!(MemoryOrder.rel)(_writer, true);
        atomicFence!(MemoryOrder.rel);

        _completions ~= val;
        atomicFetchAdd!(MemoryOrder.rel)(_length, 1);

        atomicFence!(MemoryOrder.rel);
        atomicStore!(MemoryOrder.rel)(_writer, false);
    }
}
