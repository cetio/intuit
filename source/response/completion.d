module intuit.response.completion;

import std.variant;
import std.json;
import std.conv : to;
import core.atomic;
import core.thread;
import core.exception;

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

struct Choice
{
    Variant data;
    FinishReason finishReason;
    float logProbs;
    
    string text()
    {
        if (data.type == typeid(string))
            return data.get!string;
        else if (data.type == typeid(string[2]))
            return data.get!(string[2])[0];

        throw new Exception("Completion does not have text available.");
    }

    string reasoning()
    {
        if (data.type == typeid(string[2]))
            return data.get!(string[2])[1];

        throw new Exception("Completion does not have reasoning available.");
    }
}

struct Completion
{
    Choice[] choices;

    T select(T)(size_t index = 0)
        if (is(T == string))
        => choices[index].text;

    T select(T)(size_t index = 0)
        if (is(T == JSONValue))
        => choices[index].toJSON();
}

class CompletionStream
{
private:
    Completion[] completions;
    shared size_t length;
    shared size_t index;
    shared bool writer;

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
        this.completions = null;
        this.length = 0;
        this.index = 0;
        this.writer = false;
        this.complete = false;
        this.json = JSONValue.emptyObject;
    }

    Completion next()
    {
        if (_commence is null)
            throw new Exception("Stream not initialized");

        while (atomicLoad!(MemoryOrder.acq)(writer))
            Thread.yield();

        size_t cur = atomicFetchAdd!(MemoryOrder.seq)(index, 1);

        while (cur >= atomicLoad!(MemoryOrder.acq)(length))
        {
            if (complete)
                return completions[atomicLoad!(MemoryOrder.acq)(length) - 1];
            Thread.yield();
        }

        atomicFence!(MemoryOrder.acq);
        return completions[cur];
    }

    Completion[] collect(size_t count)
    {
        Completion[] ret;
        foreach (i; 0 .. count)
            ret ~= next();
        return ret;
    }

    void begin()
    {
        _commence(this);
    }

    void update(Completion val)
    {
        atomicStore!(MemoryOrder.rel)(writer, true);
        atomicFence!(MemoryOrder.rel);

        completions ~= val;
        atomicFetchAdd!(MemoryOrder.rel)(length, 1);

        atomicFence!(MemoryOrder.rel);
        atomicStore!(MemoryOrder.rel)(writer, false);
    }
}
