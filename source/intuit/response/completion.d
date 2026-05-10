module intuit.response.completion;

import core.atomic;
import core.thread;
import std.json : JSONValue, parseJSON;

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

struct ToolCall
{
    string id;
    string name;
    JSONValue arguments;
}

struct Choice
{
    JSONValue raw;
    JSONValue content;
    string text;
    string reasoning;
    FinishReason finishReason;
    JSONValue logProbs;
    ToolCall[] toolCalls;
}

struct Completion
{
    JSONValue raw;
    Choice[] choices;

    ref inout(Choice) choice(size_t index = 0) inout
    {
        if (index >= choices.length)
            throw new Exception("Completion choice index is out of range.");
        return choices[index];
    }

    string text() const
        => text(0);

    string text(size_t index = 0) const
        => choice(index).text;

    string reasoning() const
        => reasoning(0);

    string reasoning(size_t index = 0) const
        => choice(index).reasoning;

    JSONValue json() const
        => json(0);

    JSONValue json(size_t index = 0) const
        => text(index).parseJSON();
}

class CompletionStream
{
private:
    Completion[] _completions;
    shared size_t _length;
    shared size_t _index;
    shared bool _writer;

package:
    void delegate(CompletionStream) _commence;

public:
    JSONValue json;
    string model;
    bool complete;
    void delegate(Completion) callback;

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
                return _completions[atomicLoad!(MemoryOrder.acq)(_length) - 1];
            Thread.yield();
        }

        atomicFence!(MemoryOrder.acq);
        return _completions[cur];
    }

    Completion[] collect(size_t count)
    {
        Completion[] ret;
        foreach (i; 0..count)
            ret ~= next();
        return ret;
    }

    void begin()
    {
        _commence(this);
    }

    void commence(void delegate(CompletionStream) cb)
    {
        _commence = cb;
        begin();
    }

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
