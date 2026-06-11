/// Anthropic Claude model implementation with Messages API parameters.
module intuit.provider.claude.model;

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

public:
    /// Sampling temperature.
    double temperature = double.nan;
    /// Nucleus sampling probability.
    double topP = double.nan;
    /// Top-k sampling parameter.
    long topK = -1;
    /// Maximum tokens to generate.
    long maxTokens = -1;
    /// Stop sequences.
    string[] stopSequences;
    /// System prompt.
    string system;
    /// Thinking budget in tokens.
    long thinkingBudget = -1;
    /// Tool choice JSON.
    JSONValue toolChoice;
    /// Whether toolChoice has been set.
    bool hasToolChoice;
    /// Response format JSON.
    JSONValue responseFormat;
    /// Whether responseFormat has been set.
    bool hasResponseFormat;

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

    override string name()
        => _name;

    override string owner()
        => _owner;

    override string toString() const
        => "ClaudeModel("~_name~", "~_owner~")";

    /// Forces the model to call a specific tool.
    void forceTool(string toolName)
    {
        JSONValue choice = JSONValue.emptyObject;
        choice["type"] = JSONValue("tool");
        choice["name"] = JSONValue(toolName);
        toolChoice = choice;
        hasToolChoice = true;
    }

    /// Enables JSON object response mode.
    void jsonMode()
    {
        JSONValue format = JSONValue.emptyObject;
        format["type"] = JSONValue("json");
        responseFormat = format;
        hasResponseFormat = true;
    }

    /**
     * Enables JSON schema response mode.
     *
     * Params:
     *  name = The schema name.
     *  schema = The JSON schema object.
     *  strict = Whether to enforce strict schema adherence.
     */
    void jsonSchema(string name, JSONValue schema, bool strict = true)
    {
        JSONValue format = JSONValue.emptyObject;
        format["type"] = JSONValue("json");

        JSONValue spec = JSONValue.emptyObject;
        spec["name"] = JSONValue(name);
        spec["schema"] = schema;
        spec["strict"] = JSONValue(strict);
        format["json"] = spec;
        responseFormat = format;
        hasResponseFormat = true;
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

        if (maxTokens >= 0)
            ret["max_tokens"] = JSONValue(maxTokens);
        if (!isNaN(temperature))
            ret["temperature"] = JSONValue(temperature);
        if (!isNaN(topP))
            ret["top_p"] = JSONValue(topP);
        if (topK >= 0)
            ret["top_k"] = JSONValue(topK);
        if (stopSequences.length > 0)
        {
            JSONValue arr = JSONValue.emptyArray;
            foreach (s; stopSequences)
                arr.array ~= JSONValue(s);
            ret["stop_sequences"] = arr;
        }
        if (thinkingBudget >= 0)
        {
            JSONValue thinking = JSONValue.emptyObject;
            thinking["type"] = JSONValue("enabled");
            thinking["budget_tokens"] = JSONValue(thinkingBudget);
            ret["thinking"] = thinking;
        }
        if (hasResponseFormat)
            ret["response_format"] = responseFormat;
        if (hasToolChoice)
            ret["tool_choice"] = toolChoice;

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

        string systemPrompt = system;

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
