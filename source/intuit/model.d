/// Base model configuration class for LLM providers.
module intuit.model;

import intuit.error : EndpointError, FormatError;
import intuit.response;
import intuit.tool;

import std.algorithm.searching : canFind;
import std.json : JSONValue, JSONType, parseJSON;
import std.math : isNaN;
import std.string : toLower;

/// Configuration and request/response logic for an LLM model.
class ModelConfig
{
    /// Model identifier, e.g. "gpt-4o".
    string name;
    /// Sampling temperature.
    double temperature = double.nan;
    /// Nucleus sampling probability.
    double topP = double.nan;
    /// Maximum tokens to generate.
    long maxTokens = -1;
    /// Stop sequences.
    string[] stop;
    /// Random seed for deterministic outputs.
    long seed = 0;
    /// Response format schema.
    /// Recommended to set this using setResponseSchema().
    JSONValue responseSchema;
    /// Tool choice and liability.
    /// Recommended to set this using setRequiredTool() or setToolLiability().
    JSONValue toolConfig;
    /// Additional JSON parameters merged into the request payload.
    JSONValue params;

    /**
     * Sets the tool configuration to force a specific tool call.
     *
     * If tool is null or has no name, the tool configuration is reset.
     *
     * Params:
     *  tool = The tool forced to be called by the model.
     */
    void setRequiredTool(Tool tool)
    {
        if (tool is null || tool.name == null)
        {
            toolConfig = JSONValue(null);
            return;
        }
        
        toolConfig = JSONValue.emptyObject;
        toolConfig["type"] = JSONValue("function");

        JSONValue func = JSONValue.emptyObject;
        func["name"] = JSONValue(tool.name);
        toolConfig["function"] = func;
    }

    /**
     * Sets the tool configuration to control tool calling behavior.
     *
     * If liability is "required", the model must call 1 or more tools.
     * If liability is "auto", the model can call 0 or more tools.
     * If liability is "none", the model cannot call tools.
     *
     * Params:
     *  liability = The tool liability (required, auto, none).
     * 
     * Throws:
     *  FormatError if liability is not one of the allowed values.
     */
    void setToolLiability(string liability)
    {
        if (liability != "required" && liability != "auto" && liability != "none")
        {
            throw new FormatError(
                "Tool liability must be 'required', 'auto', or 'none', not '"~liability~"'"
            );
        }

        toolConfig = JSONValue(liability);
    }

    /**
     * Sets the response format JSON schema.
     *
     * Does not support multiple schemas.
     *
     * Params:
     *  name = The schema name.
     *  schema = The JSON schema object.
     *  strict = Whether to enforce strict schema adherence.
     */
    void setResponseSchema(string name, JSONValue schema, bool strict = true)
    {
        JSONValue json = JSONValue.emptyObject;
        json["type"] = JSONValue("json_schema");

        JSONValue spec = JSONValue.emptyObject;
        spec["name"] = JSONValue(name);
        spec["schema"] = schema;
        spec["strict"] = JSONValue(strict);
        json["json_schema"] = spec;
        responseSchema = json;
    }

    /**
     * Constructs a BaseModel.
     *
     * Params:
     *  name = The model name.
     */
    this(string name)
    {
        this.name = name;
    }

    /**
     * Builds the completions request payload.
     *
     * Params:
     *  input = The input messages or raw content.
     *  tools = Registered tools to include.
     *
     * Returns:
     *  The JSON payload for the completions endpoint.
     */
    JSONValue buildPayload(JSONValue input, ToolRegistry tools = ToolRegistry.init)
    {
        JSONValue ret = JSONValue.emptyObject;
        ret["model"] = JSONValue(name);

        if (maxTokens >= 0)
            ret["max_tokens"] = JSONValue(maxTokens);
        if (!isNaN(temperature))
            ret["temperature"] = JSONValue(temperature);
        if (!isNaN(topP))
            ret["top_p"] = JSONValue(topP);
        if (stop.length > 0)
        {
            JSONValue stops = JSONValue.emptyArray;
            foreach (seq; stop)
                stops.array ~= JSONValue(seq);
            ret["stop"] = stops;
        }
        if (seed > 0)
            ret["seed"] = JSONValue(seed);
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
            JSONValue message = JSONValue.emptyObject;
            message["role"] = JSONValue("user");
            message["content"] = input;
            ret["messages"].array ~= message;
        }

        if (params.type == JSONType.object)
        {
            foreach (key, value; params.object)
                ret[key] = value;
        }

        return ret;
    }

    /**
     * Builds the embeddings request payload.
     *
     * Params:
     *  input = The input data to embed.
     *
     * Returns:
     *  The JSON payload for the embeddings endpoint.
     */
    JSONValue buildEmbeddingsPayload(JSONValue input)
    {
        JSONValue ret = JSONValue.emptyObject;
        ret["model"] = JSONValue(name);
        ret["input"] = input;

        if (params.type == JSONType.object)
        {
            foreach (key, value; params.object)
                ret[key] = value;
        }

        return ret;
    }

    /**
     * Parses a raw completions response.
     *
     * Params:
     *  json = The raw JSON response.
     *
     * Returns:
     *  The parsed Completion.
     */
    Completion parseResponse(JSONValue json)
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
     * Parses a raw embeddings response.
     *
     * Params:
     *  json = The raw JSON response.
     *
     * Returns:
     *  A JSON array containing the embedding vectors.
     */
    JSONValue parseEmbeddingsResponse(JSONValue json)
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

    /// Recursively extracts text or reasoning content from a JSON value.
    static void parseContent(
        JSONValue value,
        ref string text,
        ref string reasoning,
        bool reasoningMode
    )
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
        case "tool_calls":
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
