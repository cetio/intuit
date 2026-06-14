/// Qwen model implementation with Qwen-specific parameters.
module intuit.provider.qwen.model;

import intuit.error : EndpointError;
import intuit.model;
import intuit.response;
import intuit.tool;

import std.algorithm.searching : canFind;
import std.conv : to;
import std.format : format;
import std.json : JSONType, JSONValue, parseJSON;
import std.math : isNaN;
import std.regex;
import std.string : indexOf, strip, toLower;

/// Qwen-compatible model with XML tool-call extraction fallback.
class QwenModelConfig : ModelConfig
{
    /// Whether thinking is enabled.
    bool enableThinking = true;
    /// Top-k sampling value.
    long topK = -1;
    /// Chat template keyword arguments.
    JSONValue chatTemplateKwargs;
    /// Multimodal processor keyword arguments.
    JSONValue mmProcessorKwargs;
    /// Thinking budget.
    long thinkingBudget = -1;
    /// Embedding encoding format.
    string encodingFormat = "float";
    /// Embedding dimensions.
    long dimensions = 0;

    /**
     * Constructs a QwenModelConfig.
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
        if (!chatTemplateKwargs.isNull)
            ret["chat_template_kwargs"] = chatTemplateKwargs;
        else if (!enableThinking)
            ret["enable_thinking"] = JSONValue(false);
        if (!mmProcessorKwargs.isNull)
            ret["mm_processor_kwargs"] = mmProcessorKwargs;

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
        Completion ret = super.parseResponse(json);
        foreach (ref choice; ret.choices)
        {
            if (choice.toolCalls.length == 0 && choice.text.length > 0)
                extractXmlToolCalls(choice);
        }
        return ret;
    }

private:
    /// Attempts to extract tool calls from XML tags embedded in choice text.
    /// Supports both Qwen3-Coder custom XML and Qwen2.5+/Qwen3 JSON-in-XML formats.
    static void extractXmlToolCalls(ref Choice choice)
    {
        if (!choice.text.canFind("<tool_call>"))
            return;

        ToolCall[] customCalls = parseCustomXmlToolCalls(choice.text);
        if (customCalls.length > 0)
        {
            choice.toolCalls = customCalls;
            choice.text = stripXmlToolCalls(choice.text);
            return;
        }

        ToolCall[] jsonCalls = parseJsonInXmlToolCalls(choice.text);
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

        Regex toolCallRe = ctRegex!(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", "s");
        Regex funcRe = ctRegex!(r"<function=(.*?)</function>|<function=(.*)$", "s");
        Regex paramRe = ctRegex!(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)", "s"
        );

        foreach (tcMatch; text.matchAll(toolCallRe))
        {
            string block = tcMatch.captures[1];
            if (block.length == 0)
                block = tcMatch.captures[2];
            if (block.length == 0)
                continue;

            foreach (funcMatch; block.matchAll(funcRe))
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
                foreach (paramMatch; paramsStr.matchAll(paramRe))
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

                ToolCall call;
                call.id = generateToolCallId(ret.length);
                call.name = funcName;
                call.arguments = args;
                ret ~= call;
            }
        }
        return ret;
    }

    /// Parses JSON-in-XML tool calls (Hermes-style used by Qwen2.5/Qwen3).
    static ToolCall[] parseJsonInXmlToolCalls(string text)
    {
        ToolCall[] ret;
        // Match content between <tool_call> and </tool_call>
        Regex re = ctRegex!(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", "s");
        foreach (m; text.matchAll(re))
        {
            try
            {
                JSONValue json = parseJSON(m.captures[1]);
                ToolCall call;
                call.id = generateToolCallId(ret.length);
                if ("name" in json)
                    call.name = json["name"].str;
                else if ("function" in json)
                    call.name = json["function"].str;

                if ("arguments" in json && json["arguments"].type == JSONType.object)
                    call.arguments = json["arguments"];
                else if ("parameters" in json && json["parameters"].type == JSONType.object)
                    call.arguments = json["parameters"];

                if (call.name.length > 0)
                    ret ~= call;
            }
            catch (Exception) { }
        }
        return ret;
    }

    /// Removes XML tool-call blocks from text, preserving any leading/trailing prose.
    static string stripXmlToolCalls(string text)
    {
        return text.replaceAll(ctRegex(r"<tool_call>.*?</tool_call>", "s"), "")
            .replaceAll(ctRegex(r"<tool_call>.*$", "s"), "")
            .strip();
    }

    /// Generates a synthetic tool call ID for XML-extracted calls.
    static string generateToolCallId(size_t index)
    {
        return format("call_%016x", index);
    }

    /// Converts a string value to the most appropriate JSON type.
    static JSONValue tryConvertValue(string value)
    {
        string trimmed = value.strip;
        if (trimmed.length == 0)
            return JSONValue("");

        try
        {
            long num = trimmed.to!long;
            if (num.to!string == trimmed)
                return JSONValue(num);
        }
        catch (Exception) { }

        try
        {
            double num = trimmed.to!double;
            if (num.to!string == trimmed || (num.to!string~"f") == trimmed)
                return JSONValue(num);
        }
        catch (Exception) { }

        if (trimmed == "true" || trimmed == "True")
            return JSONValue(true);
        if (trimmed == "false" || trimmed == "False")
            return JSONValue(false);
        if (trimmed == "null" || trimmed == "None")
            return JSONValue(null);

        try
        {
            return parseJSON(trimmed);
        }
        catch (Exception) { }

        return JSONValue(trimmed);
    }
}
