/// Completion response types and streaming support.
module intuit.response.completion;

import core.atomic;
import core.thread;
import std.json : JSONValue, parseJSON;

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

/// Parsed result from a completions request.
struct Completion
{
    /// The raw JSON response.
    JSONValue raw;
    /// All generated choices.
    Choice[] choices;

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

/// Thread-safe completion stream consumer.
class CompletionStream
{
private:
    Completion[] _completions;
    shared size_t _length;
    shared size_t _index;
    shared bool _writer;

package:
    /// Internal delegate to start the stream.
    void delegate(CompletionStream) _commence;

public:
    /// Accumulated metadata JSON.
    JSONValue json;
    /// Model name for the stream.
    string model;
    /// True when the stream has finished.
    bool complete;
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
