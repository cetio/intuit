/// Qwen model implementation with Qwen-specific parameters.
module intuit.qwen.model;

import std.algorithm.searching : canFind;
import intuit.error : EndpointError;
import intuit.model;
import intuit.response;
import intuit.tool;
import conductor.serialize : toJSON;
import std.conv : to;
import std.json : JSONValue, JSONType, parseJSON;
import std.math : isNaN;
import std.regex;
import std.string : indexOf, toLower, strip;

/// Qwen-compatible model with standard OpenAI and Qwen-specific parameters.
class QwenModel : IModel
{
private:
    string _name;
    string _owner;

    double _temperature = double.nan;
    double _topP = double.nan;
    long _maxTokens = -1;
    string[] _stop;
    double _presencePenalty = double.nan;
    double _frequencyPenalty = double.nan;
    long _n = 1;
    long[long] _logitBias;
    long _seed = 0;
    string _encodingFormat = "float";
    long _dimensions = 0;
    JSONValue _responseFormat;
    bool _hasResponseFormat;
    JSONValue _toolChoice;
    bool _hasToolChoice;

    // Qwen-specific parameters
    bool _enableThinking = true;
    long _topK = -1;
    JSONValue _extraBody;
    bool _hasExtraBody;
    JSONValue _chatTemplateKwargs;
    bool _hasChatTemplateKwargs;
    JSONValue _mmProcessorKwargs;
    bool _hasMmProcessorKwargs;
    long _thinkingBudget = -1;

public:
    /**
     * Constructs a QwenModel.
     *
     * Params:
     *  name = The model name.
     *  owner = The model owner, if known.
     */
    this(string name, string owner = null)
    {
        this._name = name;
        this._owner = owner;
    }

    override ref string name()
        => _name;

    override ref string owner()
        => _owner;

    override string toString() const
        => "QwenModel("~_name~", "~_owner~")";

    // Standard OpenAI parameters (chainable and accessor properties)

    /// Gets or sets the sampling temperature.
    ref double temperature()
        => _temperature;

    /// ditto
    QwenModel temperature(double val)
    {
        _temperature = val;
        return this;
    }

    /// Gets or sets the nucleus sampling probability.
    ref double topP()
        => _topP;

    /// ditto
    QwenModel topP(double val)
    {
        _topP = val;
        return this;
    }

    /// Gets or sets the maximum number of tokens to generate.
    ref long maxTokens()
        => _maxTokens;

    /// ditto
    QwenModel maxTokens(long val)
    {
        _maxTokens = val;
        return this;
    }

    /// Gets or sets the stop sequences.
    ref string[] stop()
        => _stop;

    /// ditto
    QwenModel stop(string[] val)
    {
        _stop = val;
        return this;
    }

    /// Gets or sets the presence penalty.
    ref double presencePenalty()
        => _presencePenalty;

    /// ditto
    QwenModel presencePenalty(double val)
    {
        _presencePenalty = val;
        return this;
    }

    /// Gets or sets the frequency penalty.
    ref double frequencyPenalty()
        => _frequencyPenalty;

    /// ditto
    QwenModel frequencyPenalty(double val)
    {
        _frequencyPenalty = val;
        return this;
    }

    /// Gets or sets the number of completions to generate.
    ref long n()
        => _n;

    /// ditto
    QwenModel n(long val)
    {
        _n = val;
        return this;
    }

    /// Gets or sets the logit bias map.
    ref long[long] logitBias()
        => _logitBias;

    /// ditto
    QwenModel logitBias(long[long] val)
    {
        _logitBias = val;
        return this;
    }

    /// Gets or sets the random seed.
    ref long seed()
        => _seed;

    /// ditto
    QwenModel seed(long val)
    {
        _seed = val;
        return this;
    }

    /// Gets or sets the embedding encoding format.
    ref string encodingFormat()
        => _encodingFormat;

    /// ditto
    QwenModel encodingFormat(string val)
    {
        _encodingFormat = val;
        return this;
    }

    /// Gets or sets the embedding dimensions.
    ref long dimensions()
        => _dimensions;

    /// ditto
    QwenModel dimensions(long val)
    {
        _dimensions = val;
        return this;
    }

    /// Sets the response format JSON.
    QwenModel responseFormat(JSONValue val)
    {
        _responseFormat = val;
        _hasResponseFormat = true;
        return this;
    }

    /// Sets the tool_choice parameter.
    QwenModel toolChoice(JSONValue val)
    {
        _toolChoice = val;
        _hasToolChoice = true;
        return this;
    }

    /// ditto
    QwenModel toolChoice(string val)
    {
        _toolChoice = JSONValue(val);
        _hasToolChoice = true;
        return this;
    }

    /// Forces the model to call a specific tool.
    QwenModel forceTool(string toolName)
    {
        JSONValue choice = JSONValue.emptyObject;
        choice["type"] = JSONValue("function");
        JSONValue func = JSONValue.emptyObject;
        func["name"] = JSONValue(toolName);
        choice["function"] = func;
        _toolChoice = choice;
        _hasToolChoice = true;
        return this;
    }

    /// Enables JSON object response mode.
    QwenModel jsonMode()
    {
        JSONValue format = JSONValue.emptyObject;
        format["type"] = JSONValue("json_object");
        return responseFormat(format);
    }

    /**
     * Enables JSON schema response mode.
     *
     * Params:
     *  name = The schema name.
     *  schema = The JSON schema object.
     *  strict = Whether to enforce strict schema adherence.
     *
     * Returns:
     *  A reference to this model for chaining.
     */
    QwenModel jsonSchema(string name, JSONValue schema, bool strict = true)
    {
        JSONValue format = JSONValue.emptyObject;
        format["type"] = JSONValue("json_schema");

        JSONValue spec = JSONValue.emptyObject;
        spec["name"] = JSONValue(name);
        spec["schema"] = schema;
        spec["strict"] = JSONValue(strict);
        format["json_schema"] = spec;
        return responseFormat(format);
    }

    // Qwen-specific parameters (chainable and accessor properties)

    /// Gets or sets whether thinking is enabled.
    ref bool enableThinking()
        => _enableThinking;

    /// ditto
    QwenModel enableThinking(bool val)
    {
        _enableThinking = val;
        return this;
    }

    /// Gets or sets the top-k sampling value.
    ref long topK()
        => _topK;

    /// ditto
    QwenModel topK(long val)
    {
        _topK = val;
        return this;
    }

    /// Gets or sets extra body parameters.
    ref JSONValue extraBody()
        => _extraBody;

    /// ditto
    QwenModel extraBody(JSONValue val)
    {
        _extraBody = val;
        _hasExtraBody = true;
        return this;
    }

    /// Gets or sets chat template keyword arguments.
    ref JSONValue chatTemplateKwargs()
        => _chatTemplateKwargs;

    /// ditto
    QwenModel chatTemplateKwargs(JSONValue val)
    {
        _chatTemplateKwargs = val;
        _hasChatTemplateKwargs = true;
        return this;
    }

    /// Gets or sets multimodal processor keyword arguments.
    ref JSONValue mmProcessorKwargs()
        => _mmProcessorKwargs;

    /// ditto
    QwenModel mmProcessorKwargs(JSONValue val)
    {
        _mmProcessorKwargs = val;
        _hasMmProcessorKwargs = true;
        return this;
    }

    /// Gets or sets the thinking budget.
    ref long thinkingBudget()
        => _thinkingBudget;

    /// ditto
    QwenModel thinkingBudget(long val)
    {
        _thinkingBudget = val;
        return this;
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
    override JSONValue completionsJSON(JSONValue input, ToolRegistry tools = ToolRegistry.init)
    {
        JSONValue ret = JSONValue.emptyObject;
        ret["model"] = JSONValue(_name);

        // Standard OpenAI parameters
        if (_maxTokens >= 0) ret["max_tokens"] = JSONValue(_maxTokens);
        if (!isNaN(_temperature)) ret["temperature"] = JSONValue(_temperature);
        if (!isNaN(_topP)) ret["top_p"] = JSONValue(_topP);
        if (_stop.length > 0)
        {
            JSONValue arr = JSONValue.emptyArray;
            foreach (s; _stop) arr.array ~= JSONValue(s);
            ret["stop"] = arr;
        }
        if (!isNaN(_presencePenalty)) ret["presence_penalty"] = JSONValue(_presencePenalty);
        if (!isNaN(_frequencyPenalty)) ret["frequency_penalty"] = JSONValue(_frequencyPenalty);
        if (_n > 1) ret["n"] = JSONValue(_n);
        if (_logitBias.length > 0)
        {
            JSONValue bias = JSONValue.emptyObject;
            foreach (k, v; _logitBias) bias[k.to!string] = JSONValue(v);
            ret["logit_bias"] = bias;
        }
        if (_seed > 0) ret["seed"] = JSONValue(_seed);
        if (_hasResponseFormat) ret["response_format"] = _responseFormat;
        if (_hasToolChoice) ret["tool_choice"] = _toolChoice;

        // Qwen-specific parameters
        if (_topK >= 0) ret["top_k"] = JSONValue(_topK);
        if (_thinkingBudget >= 0) ret["thinking_budget"] = JSONValue(_thinkingBudget);

        // enable_thinking can be at top level (Alibaba Cloud) or in chat_template_kwargs (vLLM)
        // We'll add it to chat_template_kwargs if that's being used, otherwise at top level
        if (_hasChatTemplateKwargs)
            ret["chat_template_kwargs"] = _chatTemplateKwargs;
        else if (!_enableThinking)
            ret["enable_thinking"] = JSONValue(false);

        if (_hasMmProcessorKwargs) ret["mm_processor_kwargs"] = _mmProcessorKwargs;
        if (_hasExtraBody) ret["extra_body"] = _extraBody;

        // Tools support
        Tool[] toolList = tools.list();
        if (toolList.length > 0)
        {
            JSONValue toolsArray = JSONValue.emptyArray;
            foreach (tool; toolList)
            {
                JSONValue toolObj = JSONValue.emptyObject;
                toolObj["type"] = JSONValue("function");
                JSONValue functionObj = JSONValue.emptyObject;
                functionObj["name"] = JSONValue(tool.name);
                if (tool.description.length > 0)
                    functionObj["description"] = JSONValue(tool.description);
                functionObj["parameters"] = tool.schema;
                toolObj["function"] = functionObj;
                toolsArray.array ~= toolObj;
            }
            ret["tools"] = toolsArray;
        }

        if (input.type == JSONType.array)
            ret["messages"] = input;
        else
        {
            ret["messages"] = JSONValue.emptyArray;
            JSONValue msg = JSONValue.emptyObject;
            msg["role"] = JSONValue("user");
            msg["content"] = input;
            ret["messages"].array ~= msg;
        }
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
    override JSONValue embeddingsJSON(JSONValue input)
    {
        JSONValue ret = JSONValue.emptyObject;
        ret["model"] = JSONValue(_name);
        if (_encodingFormat != "float") ret["encoding_format"] = JSONValue(_encodingFormat);
        if (_dimensions > 0) ret["dimensions"] = JSONValue(_dimensions);
        ret["input"] = input;
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
    override Completion parseCompletions(JSONValue json)
    {
        Completion ret;
        ret.raw = json;
        checkError(json);

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
    override JSONValue parseEmbeddings(JSONValue json)
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
