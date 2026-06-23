/// Anthropic Claude model implementation with Messages API parameters.
module intuit.provider.claude.model;

import intuit.exception : EndpointException;
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
class ClaudeModelConfig : ModelConfig
{
public:
    /// Top-k sampling parameter.
    long topK = -1;
    /// Stop sequences.
    string[] stopSequences;
    /// System prompt.
    string system;
    /// Thinking budget in tokens.
    long thinkingBudget = -1;

    /**
     * Constructs a ClaudeModelConfig.
     *
     * Params:
     *  name = The model name.
     */
    this(string name)
    {
        super(name);
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
    override JSONValue buildPayload(JSONValue input, ToolRegistry tools = ToolRegistry.init)
    {
        JSONValue ret = JSONValue.emptyObject;
        ret["model"] = JSONValue(name);

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
            JSONValue stops = JSONValue.emptyArray;
            foreach (seq; stopSequences)
                stops.array ~= JSONValue(seq);
            ret["stop_sequences"] = stops;
        }
        if (thinkingBudget >= 0)
        {
            JSONValue thinking = JSONValue.emptyObject;
            thinking["type"] = JSONValue("enabled");
            thinking["budget_tokens"] = JSONValue(thinkingBudget);
            ret["thinking"] = thinking;
        }
        if (!responseSchema.isNull)
            ret["response_format"] = responseSchema;
        if (!toolConfig.isNull)
            ret["tool_choice"] = toolConfig;

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
                            systemPrompt ~= "\n"~msg["content"].str;
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
            JSONValue message = JSONValue.emptyObject;
            message["role"] = JSONValue("user");
            message["content"] = input;
            ret["messages"].array ~= message;
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
     *  EndpointException because Claude has no embeddings endpoint.
     */
    override JSONValue buildEmbeddingsPayload(JSONValue input)
    {
        throw new EndpointException("POST", "embeddings", 0, "not supported", "Claude does not support embeddings.");
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
    override Completion parseResponse(JSONValue json)
    {
        Completion ret;
        ret.raw = json;
        checkException(json);

        if ("model" in json && json["model"].type == JSONType.string)
            ret.usage.modelName = json["model"].str;
        else
            ret.usage.modelName = name;

        if ("latency" in json && json["latency"].type == JSONType.float_)
            ret.usage.latency = cast(float)json["latency"].floating;
        else if ("latency" in json && json["latency"].type == JSONType.integer)
            ret.usage.latency = cast(float)json["latency"].integer;

        if ("usage" in json && json["usage"].type == JSONType.object)
        {
            JSONValue usage = json["usage"];
            size_t inputTokens = readUint(usage, "input_tokens", null);
            size_t cacheReadTokens = readUint(usage, "cache_read_input_tokens", null);
            size_t cacheCreationTokens = readUint(usage, "cache_creation_input_tokens", null);
            ret.usage.promptTokens = inputTokens + cacheReadTokens + cacheCreationTokens;
            ret.usage.completionTokens = readUint(usage, "output_tokens", null);
            ret.usage.totalTokens = ret.usage.promptTokens + ret.usage.completionTokens;
            ret.usage.cacheHits = cacheReadTokens;
            ret.usage.cacheMisses = inputTokens + cacheCreationTokens;
        }

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
                        ToolCall call;
                        call.id = ("id" in block) ? block["id"].str : "";
                        call.name = ("name" in block) ? block["name"].str : "";
                        if ("input" in block)
                            call.arguments = block["input"];
                        choice.toolCalls ~= call;
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
     *  EndpointException because Claude has no embeddings endpoint.
     */
    override JSONValue parseEmbeddingsResponse(JSONValue json)
    {
        throw new EndpointException("POST", "embeddings", 0, "not supported", "Claude does not support embeddings.");
    }

private:
    /// Reads an integral value from a JSON object under the primary or fallback key.
    static size_t readUint(
        JSONValue obj,
        string primary,
        string fallback = null,
    )
    {
        foreach (key; [primary, fallback])
        {
            if (key is null || key !in obj)
                continue;

            JSONValue value = obj[key];
            if (value.type == JSONType.integer)
                return cast(size_t)value.integer;
            if (value.type == JSONType.uinteger)
                return cast(size_t)value.uinteger;
        }
        return 0;
    }

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

    /// Throws if the JSON response contains an exception field.
    static void checkException(JSONValue json)
    {
        if ("error" !in json)
            return;

        if (json["error"].type == JSONType.string)
            throw new Exception(json["error"].str);
        else if (json["error"].type == JSONType.object && "message" in json["error"])
            throw new EndpointException("POST", "messages", 0, "error", json["error"]["message"].str);
        else
            throw new EndpointException("POST", "messages", 0, "error", json.toPrettyString());
    }
}
