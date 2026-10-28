/// Completion response types.
module intuit.response.completion;

import intuit.context.policy.tool : ToolPolicyResult;
import intuit.exception : EndpointException, MalformedResponseException;
import intuit.json : fromJSON;

import std.algorithm.searching : canFind;
import std.json : JSONValue, JSONType, parseJSON;
import std.string : toLower;

/// Reason why a completion generation finished.
enum FinishReason : string
{
    Missing = "missing",
    Length = "length",
    MaxTokens = "max_tokens",
    ContentFilter = "content_filter",
    Refusal = "refusal",
    ToolCall = "tool_call",
    ToolCalls = "tool_calls",
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
    /// Result of tool policy evaluation for this call.
    ToolPolicyResult policyResult;
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
    {
        string candidate = text(index);
        JSONValue ret;
        try
            ret = candidate.parseJSON();
        catch (Exception)
        {
            throw new MalformedResponseException(
                "Completion text is not valid JSON.",
                raw.toString(),
                candidate,
            );
        }
        return ret;
    }
}

package(intuit):
Completion parseCompletionResponse(JSONValue json, string modelName = null)
{
    Completion ret;
    ret.raw = json;
    // Throws if the JSON response contains an exception field.
    if ("error" in json)
    {
        if (json["error"].type == JSONType.string)
            throw new EndpointException("POST", "chat/completions", 0, "error", json["error"].str);
        else if (json["error"].type == JSONType.object && "message" in json["error"])
            throw new EndpointException(
                "POST",
                "chat/completions",
                0,
                "error",
                json["error"]["message"].str,
            );
        else
            throw new EndpointException("POST", "chat/completions", 0, "error", json.toPrettyString());
    }

    ret.usage.modelName = "model" in json && json["model"].type == JSONType.string
        ? json["model"].str : modelName;
    if ("latency" in json
        && (json["latency"].type == JSONType.float_ || json["latency"].type == JSONType.integer))
        ret.usage.latency = fromJSON!float(json["latency"]);

    // Reads integral token counts, including provider-specific fallback keys.
    if ("usage" in json && json["usage"].type == JSONType.object)
    {
        JSONValue usage = json["usage"];
        ret.usage.promptTokens = "prompt_tokens" in usage
            ? fromJSON!size_t(usage["prompt_tokens"])
            : ("input_tokens" in usage ? fromJSON!size_t(usage["input_tokens"]) : 0);
        ret.usage.completionTokens = "completion_tokens" in usage
            ? fromJSON!size_t(usage["completion_tokens"])
            : ("output_tokens" in usage ? fromJSON!size_t(usage["output_tokens"]) : 0);
        ret.usage.totalTokens = "total_tokens" in usage
            ? fromJSON!size_t(usage["total_tokens"]) : 0;
        if (ret.usage.totalTokens == 0)
            ret.usage.totalTokens = ret.usage.promptTokens + ret.usage.completionTokens;

        if ("prompt_tokens_details" in usage
            && usage["prompt_tokens_details"].type == JSONType.object
            && "cached_tokens" in usage["prompt_tokens_details"])
            ret.usage.cacheHits = fromJSON!size_t(usage["prompt_tokens_details"]["cached_tokens"]);
        ret.usage.cacheMisses = ret.usage.promptTokens - ret.usage.cacheHits;
    }

    if ("choices" in json && json["choices"].type == JSONType.array)
    {
        foreach (entry; json["choices"].array)
        {
            Choice choice;
            choice.raw = entry;
            JSONValue msg = ("message" in entry)
                ? entry["message"]
                : (("delta" in entry) ? entry["delta"] : JSONValue.init);

            // Extracts text, reasoning, and tool calls from a message JSON object.
            if (msg.type == JSONType.object)
            {
                if ("content" in msg && !msg["content"].isNull)
                {
                    choice.content = msg["content"];
                    parseContent(msg["content"], choice.text, choice.reasoning, false);
                }
                if ("reasoning" in msg && !msg["reasoning"].isNull)
                    parseContent(msg["reasoning"], choice.text, choice.reasoning, true);
                if ("reasoning_content" in msg && !msg["reasoning_content"].isNull)
                    parseContent(msg["reasoning_content"], choice.text, choice.reasoning, true);

                if ("tool_calls" in msg && msg["tool_calls"].type == JSONType.array)
                {
                    foreach (toolCall; msg["tool_calls"].array)
                    {
                        ToolCall call;
                        call.id = ("id" in toolCall) ? toolCall["id"].str : "";
                        if ("function" in toolCall && toolCall["function"].type == JSONType.object)
                        {
                            JSONValue func = toolCall["function"];
                            call.name = ("name" in func) ? func["name"].str : "";
                            if ("arguments" in func && func["arguments"].type == JSONType.string)
                                call.arguments = parseJSON(func["arguments"].str);
                        }
                        choice.toolCalls ~= call;
                    }
                }
            }
            else if ("text" in entry && entry["text"].type == JSONType.string)
            {
                choice.content = entry["text"];
                choice.text = entry["text"].str;
            }

            // Maps a JSON finish reason string to the FinishReason enum.
            choice.finishReason = "finish_reason" in entry
                && entry["finish_reason"].type == JSONType.string
                ? cast(FinishReason)entry["finish_reason"].str : FinishReason.Unknown;

            if ("logprobs" in entry && !entry["logprobs"].isNull)
                choice.logProbs = entry["logprobs"];
            else if ("log_probs" in entry && !entry["log_probs"].isNull)
                choice.logProbs = entry["log_probs"];

            ret.choices ~= choice;
        }
    }
    return ret;
}

private:
/// Recursively extracts text or reasoning content from a JSON value.
void parseContent(
    JSONValue value,
    ref string text,
    ref string reasoning,
    bool reasoningMode
)
{
    switch (value.type)
    {
    case JSONType.string:
        // Appends text to a target string if non-empty.
        if (value.str.length > 0)
        {
            if (reasoningMode)
                reasoning ~= value.str;
            else
                text ~= value.str;
        }
        return;

    case JSONType.array:
        foreach (part; value.array)
            parseContent(part, text, reasoning, reasoningMode);
        return;

    case JSONType.object:
        bool nextReasoning = reasoningMode;
        if ("type" in value && value["type"].type == JSONType.string)
        {
            // Checks if a content type string indicates reasoning content.
            string kind = value["type"].str.toLower;
            nextReasoning = kind.canFind("reason")
                || kind.canFind("think")
                || kind.canFind("summary");
        }

        if ("summary" in value && !value["summary"].isNull)
            parseContent(value["summary"], text, reasoning, true);
        if ("text" in value && !value["text"].isNull)
            parseContent(value["text"], text, reasoning, nextReasoning);
        if ("content" in value && !value["content"].isNull)
            parseContent(value["content"], text, reasoning, nextReasoning);
        if ("value" in value && !value["value"].isNull)
            parseContent(value["value"], text, reasoning, nextReasoning);
        return;

    default:
        return;
    }
}
