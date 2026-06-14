/// Qwen model implementation with Qwen-specific parameters.
module intuit.provider.qwen.model;

import intuit.error : EndpointError;
import intuit.model;
import intuit.response;
import intuit.tool;
import conductor.serialize : toJSON;

import std.algorithm.searching : canFind;
import std.conv : to;
import std.json : JSONValue, JSONType, parseJSON;
import std.math : isNaN;
import std.regex;
import std.string : indexOf, toLower, strip;

/// Qwen-compatible model with Qwen-specific parameters.
class QwenModel : ModelConfig
{
public:
    /// Whether thinking is enabled.
    bool enableThinking = true;
    /// Top-k sampling value.
    long topK = -1;
    /// Extra body parameters.
    JSONValue extraBody;
    /// Whether extraBody has been set.
    bool hasExtraBody;
    /// Chat template keyword arguments.
    JSONValue chatTemplateKwargs;
    /// Whether chatTemplateKwargs has been set.
    bool hasChatTemplateKwargs;
    /// Multimodal processor keyword arguments.
    JSONValue mmProcessorKwargs;
    /// Whether mmProcessorKwargs has been set.
    bool hasMmProcessorKwargs;
    /// Thinking budget.
    long thinkingBudget = -1;
    /// Embedding encoding format.
    string encodingFormat = "float";
    /// Embedding dimensions.
    long dimensions = 0;

    /**
     * Constructs a QwenModel.
     *
     * Params:
     *  name = The model name.
     */
    this(string name)
    {
        super(name);
    }

    /**
     * Builds the completions request payload including Qwen-specific parameters.
     *
     * Params:
     *  input = The input messages or raw content.
     *  tools = Registered tools to include.
     *
     * Returns:
     *  The JSON payload for the completions endpoint.
     */
    override JSONValue buildPayload(JSONValue input, ToolRegistry tools = ToolRegistry.init)
    {
        JSONValue ret = super.buildPayload(input, tools);

        if (topK >= 0)
            ret["top_k"] = JSONValue(topK);
        if (thinkingBudget >= 0)
            ret["thinking_budget"] = JSONValue(thinkingBudget);
        if (hasChatTemplateKwargs)
            ret["chat_template_kwargs"] = chatTemplateKwargs;
        else if (!enableThinking)
            ret["enable_thinking"] = JSONValue(false);
        if (hasMmProcessorKwargs)
            ret["mm_processor_kwargs"] = mmProcessorKwargs;
        if (hasExtraBody)
            ret["extra_body"] = extraBody;

        return ret;
    }

    /**
     * Builds the embeddings request payload from configured parameters.
     *
     * Params:
     *  input = The input data to embed.
     *
     * Returns:
     *  The JSON payload for the embeddings endpoint.
     */
    override JSONValue buildEmbeddingsPayload(JSONValue input)
    {
        JSONValue ret = super.buildEmbeddingsPayload(input);
        if (encodingFormat != "float")
            ret["encoding_format"] = JSONValue(encodingFormat);
        if (dimensions > 0)
            ret["dimensions"] = JSONValue(dimensions);
        return ret;
    }

    /**
     * Parses a raw completions JSON response into a Completion struct.
     *
     * Params:
     *  json = The raw JSON response.
     *
     * Returns:
     *  The parsed Completion.
     */
    override Completion parseResponse(JSONValue json)
    {
        Completion ret;
        ret.raw = json;
        checkError(json);
        ret.usage = parseUsage(json);

        if ("choices" in json && json["choices"].type == JSONType.array)
        {
            foreach (entry; json["choices"].array)
            {
                Choice choice;
                choice.raw = entry;
                JSONValue msg = ("message" in entry)
                    ? entry["message"]
                    : (("delta" in entry) ? entry["delta"] : JSONValue.init);
                parseMessage(choice, msg);
                choice.finishReason = parseFinishReason(
                    "finish_reason" in entry ? entry["finish_reason"] : JSONValue.init
                );

                if ("logprobs" in entry && !entry["logprobs"].isNull)
                    choice.logProbs = entry["logprobs"];
                else if ("log_probs" in entry && !entry["log_probs"].isNull)
                    choice.logProbs = entry["log_probs"];

                ret.choices ~= choice;
            }
        }
        return ret;
    }

    /**
     * Parses a raw embeddings JSON response into a JSON array of embeddings.
     *
     * Params:
     *  json = The raw JSON response.
     *
     * Returns:
     *  A JSON array containing the embedding vectors.
     */
    override JSONValue parseEmbeddingsResponse(JSONValue json)
    {
        checkError(json);

        JSONValue ret = JSONValue.emptyArray;
        if ("data" in json && json["data"].type == JSONType.array)
        {
            foreach (item; json["data"].array)
            {
                if ("embedding" in item)
                    ret.array ~= item["embedding"];
            }
        }
        return ret;
    }

private:
    /// Extracts text, reasoning, and tool calls from a message JSON object.
    static void parseMessage(ref Choice choice, JSONValue message)
    {
        if (message.type != JSONType.object)
            return;

        if ("content" in message && !message["content"].isNull)
        {
            choice.content = message["content"];
            parseContent(message["content"], choice.text, choice.reasoning, false);
        }

        if ("reasoning" in message && !message["reasoning"].isNull)
            parseContent(message["reasoning"], choice.text, choice.reasoning, true);

        if ("tool_calls" in message && message["tool_calls"].type == JSONType.array)
        {
            foreach (toolCall; message["tool_calls"].array)
            {
                ToolCall tc;
                tc.id = ("id" in toolCall) ? toolCall["id"].str : "";
                if ("function" in toolCall && toolCall["function"].type == JSONType.object)
                {
                    JSONValue func = toolCall["function"];
                    tc.name = ("name" in func) ? func["name"].str : "";
                    if ("arguments" in func && func["arguments"].type == JSONType.string)
                        tc.arguments = func["arguments"].str.parseJSON();
                }
                choice.toolCalls ~= tc;
            }
        }

        // Fallback: extract XML tool calls embedded in content text.
        // Qwen2.5+ and Qwen3 models sometimes emit tool calls as XML tags
        // inside the message content instead of a structured tool_calls array.
        if (choice.toolCalls.length == 0 && choice.text.length > 0)
            extractXmlToolCalls(choice);
    }

    /// Attempts to extract tool calls from XML tags embedded in choice text.
    /// Supports both Qwen3-Coder custom XML and Qwen2.5+/Qwen3 JSON-in-XML formats.
    static void extractXmlToolCalls(ref Choice choice)
    {
        if (!choice.text.canFind("<tool_call>"))
            return;

        // Try Qwen3-Coder custom XML: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
        auto customCalls = parseCustomXmlToolCalls(choice.text);
        if (customCalls.length > 0)
        {
            choice.toolCalls = customCalls;
            choice.text = stripXmlToolCalls(choice.text);
            return;
        }

        // Try JSON-in-XML: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        auto jsonCalls = parseJsonInXmlToolCalls(choice.text);
        if (jsonCalls.length > 0)
        {
            choice.toolCalls = jsonCalls;
            choice.text = stripXmlToolCalls(choice.text);
        }
    }

    /// Parses Qwen3-Coder custom XML tool calls from text.
    static ToolCall[] parseCustomXmlToolCalls(string text)
    {
        ToolCall[] ret;
        if (!text.canFind("<function="))
            return ret;

        // Match <tool_call>...</tool_call> blocks (complete or trailing)
        auto toolCallRe = regex(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", "s");
        auto funcRe = regex(r"<function=(.*?)</function>|<function=(.*)$", "s");
        auto paramRe = regex(r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)", "s");

        foreach (tcMatch; matchAll(text, toolCallRe))
        {
            string block = tcMatch.captures[1];
            if (block.length == 0)
                block = tcMatch.captures[2];
            if (block.length == 0)
                continue;

            foreach (funcMatch; matchAll(block, funcRe))
            {
                string funcBlock = funcMatch.captures[1];
                if (funcBlock.length == 0)
                    funcBlock = funcMatch.captures[2];
                if (funcBlock.length == 0)
                    continue;

                ptrdiff_t gt = funcBlock.indexOf(">");
                if (gt < 0)
                    continue;

                string funcName = funcBlock[0..gt].strip;
                string paramsStr = funcBlock[gt + 1..$];

                JSONValue args = JSONValue.emptyObject;
                foreach (paramMatch; matchAll(paramsStr, paramRe))
                {
                    string paramBlock = paramMatch.captures[1];
                    if (paramBlock.length == 0)
                        continue;

                    ptrdiff_t pgt = paramBlock.indexOf(">");
                    if (pgt < 0)
                        continue;

                    string paramName = paramBlock[0..pgt].strip;
                    string paramValue = paramBlock[pgt + 1..$].strip;
                    args[paramName] = tryConvertValue(paramValue);
                }

                ToolCall tc;
                tc.id = generateToolCallId(ret.length);
                tc.name = funcName;
                tc.arguments = args;
                ret ~= tc;
            }
        }
        return ret;
    }

    /// Parses JSON-in-XML tool calls (Hermes-style used by Qwen2.5/Qwen3).
    static ToolCall[] parseJsonInXmlToolCalls(string text)
    {
        ToolCall[] ret;
        // Match <tool_call>{...}</tool_call> with optional whitespace
        auto re = regex(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", "s");
        foreach (m; matchAll(text, re))
        {
            try
            {
                JSONValue json = parseJSON(m.captures[1]);
                ToolCall tc;
                tc.id = generateToolCallId(ret.length);
                if ("name" in json)
                    tc.name = json["name"].str;
                else if ("function" in json)
                    tc.name = json["function"].str;

                if ("arguments" in json && json["arguments"].type == JSONType.object)
                    tc.arguments = json["arguments"];
                else if ("parameters" in json && json["parameters"].type == JSONType.object)
                    tc.arguments = json["parameters"];

                if (tc.name.length > 0)
                    ret ~= tc;
            }
            catch (Exception)
            {
                // Skip malformed JSON blocks.
            }
        }
        return ret;
    }

    /// Removes XML tool-call blocks from text, preserving any leading/trailing prose.
    static string stripXmlToolCalls(string text)
    {
        auto re = regex(r"<tool_call>.*?</tool_call>", "s");
        string ret = replaceAll(text, re, "");
        ret = replaceAll(ret, regex(r"<tool_call>.*$", "s"), "");
        return ret.strip;
    }

    /// Generates a synthetic tool call ID for XML-extracted calls.
    static string generateToolCallId(size_t index)
    {
        import std.format : format;
        return format("call_%016x", index);
    }

    /// Converts a string value to the most appropriate JSON type.
    static JSONValue tryConvertValue(string value)
    {
        string trimmed = value.strip;
        if (trimmed.length == 0)
            return JSONValue("");

        // Try integer
        try
        {
            long num = trimmed.to!long;
            if (num.to!string == trimmed)
                return JSONValue(num);
        }
        catch (Exception)
        {
        }

        // Try float
        try
        {
            double num = trimmed.to!double;
            if (num.to!string == trimmed || (num.to!string~"f") == trimmed)
                return JSONValue(num);
        }
        catch (Exception)
        {
        }

        // Try boolean
        if (trimmed == "true" || trimmed == "True")
            return JSONValue(true);
        if (trimmed == "false" || trimmed == "False")
            return JSONValue(false);

        // Try null
        if (trimmed == "null" || trimmed == "None")
            return JSONValue(null);

        // Try JSON object/array
        try
        {
            return parseJSON(trimmed);
        }
        catch (Exception)
        {
        }

        // Fallback to string
        return JSONValue(trimmed);
    }

    /// Recursively extracts text or reasoning content from a JSON value.
    static void parseContent(JSONValue value, ref string text, ref string reasoning, bool reasoningMode)
    {
        switch (value.type)
        {
        case JSONType.string:
            appendText(reasoningMode ? reasoning : text, value.str);
            return;

        case JSONType.array:
            foreach (part; value.array)
                parseContent(part, text, reasoning, reasoningMode);
            return;

        case JSONType.object:
            bool nextReasoning = reasoningMode;
            if ("type" in value && value["type"].type == JSONType.string)
                nextReasoning = isReasoningType(value["type"].str);

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

    /// Checks if a content type string indicates reasoning content.
    static bool isReasoningType(string raw)
    {
        string kind = raw.toLower;
        return kind.canFind("reason")
            || kind.canFind("think")
            || kind.canFind("summary");
    }

    /// Appends text to a target string if non-empty.
    static void appendText(ref string target, string value)
    {
        if (value.length > 0)
            target ~= value;
    }

    /// Maps a JSON finish reason string to the FinishReason enum.
    static FinishReason parseFinishReason(JSONValue value)
    {
        if (value.type != JSONType.string)
            return FinishReason.Unknown;

        switch (value.str)
        {
        case "missing":
            return FinishReason.Missing;
        case "length":
            return FinishReason.Length;
        case "max_tokens":
            return FinishReason.Max_Tokens;
        case "content_filter":
            return FinishReason.ContentFilter;
        case "refusal":
            return FinishReason.Refusal;
        case "tool_call":
            return FinishReason.ToolCall;
        case "tool_use":
            return FinishReason.ToolUse;
        case "function_call":
            return FinishReason.FunctionCall;
        case "pause":
            return FinishReason.Pause;
        case "pause_turn":
            return FinishReason.PauseTurn;
        case "stop":
            return FinishReason.Stop;
        case "end_turn":
            return FinishReason.EndTurn;
        case "stop_sequence":
            return FinishReason.StopSequence;
        default:
            return FinishReason.Unknown;
        }
    }

    /// Throws if the JSON response contains an error field.
    static void checkError(JSONValue json)
    {
        if ("error" !in json)
            return;

        if (json["error"].type == JSONType.string)
            throw new Exception(json["error"].str);
        else if (json["error"].type == JSONType.object && "message" in json["error"])
            throw new EndpointError("POST", "chat/completions", 0, "error", json["error"]["message"].str);
        else
            throw new EndpointError("POST", "chat/completions", 0, "error", json.toPrettyString());
    }
}
