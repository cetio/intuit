/// Anthropic Claude model implementation with Messages API parameters.
module intuit.claude.model;

import intuit.error : EndpointError;
import intuit.model;
import intuit.response;
import intuit.tool;
import conductor.serialize : toJSON;

import std.algorithm.searching : canFind;
import std.conv : to;
import std.json : JSONValue, JSONType, parseJSON;
import std.math : isNaN;
import std.string : toLower;

/// Anthropic Claude model with Messages API parameters.
class ClaudeModel : IModel
{
private:
    string _name;
    string _owner;

    double _temperature = double.nan;
    double _topP = double.nan;
    long _topK = -1;
    long _maxTokens = -1;
    string[] _stopSequences;
    string _system;
    long _thinkingBudget = -1;
    JSONValue _toolChoice;
    bool _hasToolChoice;
    JSONValue _responseFormat;
    bool _hasResponseFormat;

public:
    /**
     * Constructs a ClaudeModel.
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
        => "ClaudeModel("~_name~", "~_owner~")";

    /// Gets or sets the sampling temperature.
    ref double temperature()
        => _temperature;

    /// ditto
    ClaudeModel temperature(double val)
    {
        _temperature = val;
        return this;
    }

    /// Gets or sets the nucleus sampling probability.
    ref double topP()
        => _topP;

    /// ditto
    ClaudeModel topP(double val)
    {
        _topP = val;
        return this;
    }

    /// Gets or sets the top-k sampling parameter.
    ref long topK()
        => _topK;

    /// ditto
    ClaudeModel topK(long val)
    {
        _topK = val;
        return this;
    }

    /// Gets or sets the maximum number of tokens to generate.
    ref long maxTokens()
        => _maxTokens;

    /// ditto
    ClaudeModel maxTokens(long val)
    {
        _maxTokens = val;
        return this;
    }

    /// Gets or sets the stop sequences.
    ref string[] stopSequences()
        => _stopSequences;

    /// ditto
    ClaudeModel stopSequences(string[] val)
    {
        _stopSequences = val;
        return this;
    }

    /// Gets or sets the system prompt.
    ref string system()
        => _system;

    /// ditto
    ClaudeModel system(string val)
    {
        _system = val;
        return this;
    }

    /// Gets or sets the thinking budget in tokens.
    ref long thinkingBudget()
        => _thinkingBudget;

    /// ditto
    ClaudeModel thinkingBudget(long val)
    {
        _thinkingBudget = val;
        return this;
    }

    /// Sets the tool_choice parameter.
    ClaudeModel toolChoice(JSONValue val)
    {
        _toolChoice = val;
        _hasToolChoice = true;
        return this;
    }

    /// ditto
    ClaudeModel toolChoice(string val)
    {
        _toolChoice = JSONValue(val);
        _hasToolChoice = true;
        return this;
    }

    /// Forces the model to call a specific tool.
    ClaudeModel forceTool(string toolName)
    {
        JSONValue choice = JSONValue.emptyObject;
        choice["type"] = JSONValue("tool");
        choice["name"] = JSONValue(toolName);
        _toolChoice = choice;
        _hasToolChoice = true;
        return this;
    }

    /// Sets the response format to JSON.
    ClaudeModel responseFormat(JSONValue val)
    {
        _responseFormat = val;
        _hasResponseFormat = true;
        return this;
    }

    /// Sets the response format to JSON with a schema.
    ClaudeModel jsonSchema(string name, JSONValue schema, bool strict = true)
    {
        JSONValue format = JSONValue.emptyObject;
        format["type"] = JSONValue("json");

        JSONValue spec = JSONValue.emptyObject;
        spec["name"] = JSONValue(name);
        spec["schema"] = schema;
        spec["strict"] = JSONValue(strict);
        format["json"] = spec;
        return responseFormat(format);
    }

    /**
     * Builds the completions request payload for Claude Messages API.
     *
     * Params:
     *  input = The input messages or raw content.
     *  tools = Registered tools to include.
     *
     * Returns:
     *  The JSON payload for the messages endpoint.
     */
    override JSONValue completionsJSON(JSONValue input, ToolRegistry tools = ToolRegistry.init)
    {
        JSONValue ret = JSONValue.emptyObject;
        ret["model"] = JSONValue(_name);

        if (_maxTokens >= 0)
            ret["max_tokens"] = JSONValue(_maxTokens);
        if (!isNaN(_temperature))
            ret["temperature"] = JSONValue(_temperature);
        if (!isNaN(_topP))
            ret["top_p"] = JSONValue(_topP);
        if (_topK >= 0)
            ret["top_k"] = JSONValue(_topK);
        if (_stopSequences.length > 0)
        {
            JSONValue arr = JSONValue.emptyArray;
            foreach (s; _stopSequences)
                arr.array ~= JSONValue(s);
            ret["stop_sequences"] = arr;
        }
        if (_thinkingBudget >= 0)
        {
            JSONValue thinking = JSONValue.emptyObject;
            thinking["type"] = JSONValue("enabled");
            thinking["budget_tokens"] = JSONValue(_thinkingBudget);
            ret["thinking"] = thinking;
        }
        if (_hasResponseFormat)
            ret["response_format"] = _responseFormat;
        if (_hasToolChoice)
            ret["tool_choice"] = _toolChoice;

        Tool[] toolList = tools.list();
        if (toolList.length > 0)
        {
            JSONValue toolsArray = JSONValue.emptyArray;
            foreach (tool; toolList)
            {
                JSONValue toolObj = JSONValue.emptyObject;
                toolObj["name"] = JSONValue(tool.name);
                if (tool.description.length > 0)
                    toolObj["description"] = JSONValue(tool.description);
                toolObj["input_schema"] = tool.schema;
                toolsArray.array ~= toolObj;
            }
            ret["tools"] = toolsArray;
        }

        string systemPrompt = _system;

        if (input.type == JSONType.array)
        {
            JSONValue[] remainingMessages;
            foreach (msg; input.array)
            {
                if ("role" in msg && msg["role"].type == JSONType.string && msg["role"].str == "system")
                {
                    if ("content" in msg && msg["content"].type == JSONType.string)
                    {
                        if (systemPrompt.length > 0)
                            systemPrompt ~= "\n" ~ msg["content"].str;
                        else
                            systemPrompt = msg["content"].str;
                    }
                }
                else
                {
                    remainingMessages ~= msg;
                }
            }
            ret["messages"] = JSONValue(remainingMessages);
        }
        else
        {
            ret["messages"] = JSONValue.emptyArray;
            JSONValue msg = JSONValue.emptyObject;
            msg["role"] = JSONValue("user");
            msg["content"] = input;
            ret["messages"].array ~= msg;
        }

        if (systemPrompt.length > 0)
            ret["system"] = JSONValue(systemPrompt);

        return ret;
    }

    /**
     * Builds the embeddings request payload.
     *
     * Claude does not support embeddings through the Messages API.
     *
     * Throws:
     *  EndpointError because Claude has no embeddings endpoint.
     */
    override JSONValue embeddingsJSON(JSONValue input)
    {
        throw new EndpointError("POST", "embeddings", 0, "not supported", "Claude does not support embeddings.");
    }

    /**
     * Parses a raw Claude Messages API JSON response into a Completion struct.
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
        ret.usage = parseUsage(json);

        Choice choice;
        choice.raw = json;

        if ("content" in json && json["content"].type == JSONType.array)
        {
            foreach (block; json["content"].array)
            {
                if (block.type != JSONType.object)
                    continue;

                if ("type" in block && block["type"].type == JSONType.string)
                {
                    string blockType = block["type"].str;
                    if (blockType == "text" && "text" in block)
                        choice.text ~= block["text"].str;
                    else if (blockType == "thinking" && "thinking" in block)
                        choice.reasoning ~= block["thinking"].str;
                    else if (blockType == "redacted_thinking")
                        choice.reasoning ~= "[redacted thinking]";
                    else if (blockType == "tool_use")
                    {
                        ToolCall tc;
                        tc.id = ("id" in block) ? block["id"].str : "";
                        tc.name = ("name" in block) ? block["name"].str : "";
                        if ("input" in block)
                            tc.arguments = block["input"];
                        choice.toolCalls ~= tc;
                    }
                }
            }
        }

        choice.finishReason = parseFinishReason(
            "stop_reason" in json ? json["stop_reason"] : JSONValue.init
        );

        ret.choices ~= choice;
        return ret;
    }

    /**
     * Parses a raw embeddings JSON response.
     *
     * Claude does not support embeddings.
     *
     * Throws:
     *  EndpointError because Claude has no embeddings endpoint.
     */
    override JSONValue parseEmbeddings(JSONValue json)
    {
        throw new EndpointError("POST", "embeddings", 0, "not supported", "Claude does not support embeddings.");
    }

private:
    /// Maps a JSON finish reason string to the FinishReason enum.
    static FinishReason parseFinishReason(JSONValue value)
    {
        if (value.type != JSONType.string)
            return FinishReason.Unknown;

        switch (value.str)
        {
        case "end_turn":
            return FinishReason.EndTurn;
        case "max_tokens":
            return FinishReason.Max_Tokens;
        case "stop_sequence":
            return FinishReason.StopSequence;
        case "tool_use":
            return FinishReason.ToolUse;
        case "content_filter":
            return FinishReason.ContentFilter;
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
            throw new EndpointError("POST", "messages", 0, "error", json["error"]["message"].str);
        else
            throw new EndpointError("POST", "messages", 0, "error", json.toPrettyString());
    }
}
