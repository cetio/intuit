module intuit.response.completion;

import intuit.error : CompletionParseError;
import std.json : JSONType, JSONValue, parseJSON;
import std.string : strip;

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
    JSONValue raw;
    JSONValue content;
    string text;
    string reasoning;
    FinishReason finishReason;
    JSONValue logProbs;
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

    string text(size_t index = 0) const
    {
        return choice(index).text;
    }

    string reasoning(size_t index = 0) const
    {
        return choice(index).reasoning;
    }

    JSONValue parsedJSON(size_t index = 0) const
    {
        return parseStructuredJSON(text(index));
    }
}

private:

JSONValue parseStructuredJSON(string raw)
{
    string text = raw.strip;
    if (text.length == 0)
        throw new CompletionParseError("Completion did not contain parseable text.", raw, null);

    string candidate;
    string parseError;
    foreach (size_t i, char c; text)
    {
        if (c != '{' && c != '[')
            continue;

        candidate = balancedSlice(text, i);
        if (candidate.length == 0)
            continue;

        try
        {
            return parseJSON(candidate);
        }
        catch (Exception e)
        {
            parseError = e.msg;
        }
    }

    if (candidate.length == 0)
        throw new CompletionParseError("Completion did not contain JSON.", raw, null);

    throw new CompletionParseError("Completion response was not valid JSON: "~parseError, raw, candidate);
}

string balancedSlice(string raw, size_t start)
{
    char[] stack;
    bool inString;
    bool escaped;

    char opener = raw[start];
    if (opener != '{' && opener != '[')
        return null;

    stack ~= opener == '{' ? '}' : ']';
    foreach (size_t i; start + 1..raw.length)
    {
        char c = raw[i];
        if (inString)
        {
            if (escaped)
            {
                escaped = false;
                continue;
            }

            if (c == '\\')
            {
                escaped = true;
                continue;
            }

            if (c == '"')
                inString = false;
            continue;
        }

        if (c == '"')
        {
            inString = true;
            continue;
        }

        if (c == '{')
        {
            stack ~= '}';
            continue;
        }

        if (c == '[')
        {
            stack ~= ']';
            continue;
        }

        if (c == '}' || c == ']')
        {
            if (stack.length == 0 || c != stack[$-1])
                return null;

            stack = stack[0..$-1];
            if (stack.length == 0)
                return raw[start..i + 1].idup;
        }
    }

    return null;
}
