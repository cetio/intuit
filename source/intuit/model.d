/// Base model configuration class for LLM providers.
module intuit.model;

import intuit.exception : EndpointException, FormatException, MalformedResponseException;
import intuit.json : fromJSON;
import intuit.response;
import intuit.tool;

import std.algorithm.searching : canFind;
import std.json : JSONValue, JSONType, parseJSON;
import std.math : isFinite, isNaN;
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
     *  FormatException if liability is not one of the allowed values.
     */
    void setToolLiability(string liability)
    {
        if (liability != "required" && liability != "auto" && liability != "none")
        {
            throw new FormatException(
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
        spec["schema"] = normalizeSchema(schema);
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
            ret.usage.promptTokens = readUint(usage, "prompt_tokens", "input_tokens");
            ret.usage.completionTokens = readUint(usage, "completion_tokens", "output_tokens");
            ret.usage.totalTokens = readUint(usage, "total_tokens", null);
            if (ret.usage.totalTokens == 0)
                ret.usage.totalTokens = ret.usage.promptTokens + ret.usage.completionTokens;

            if ("prompt_tokens_details" in usage
                && usage["prompt_tokens_details"].type == JSONType.object)
            {
                JSONValue details = usage["prompt_tokens_details"];
                ret.usage.cacheHits = readUint(details, "cached_tokens", null);
            }
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
        checkException(json);

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

    JSONValue buildDecisionsPayload(JSONValue input, DecisionQuestion[] questions)
    {
        if (input.type != JSONType.string && input.type != JSONType.array)
            throw new FormatException("OpenAI decisions require text or user messages.");

        JSONValue ret = JSONValue.emptyObject;
        if (params.type == JSONType.object)
        {
            foreach (key, value; params.object)
                ret[key] = value;
        }

        ret["model"] = JSONValue(name);
        ret["input"] = input;
        ret["questions"] = decisionQuestions(questions);
        return ret;
    }

    Decision parseDecisionsResponse(JSONValue json, DecisionQuestion[] questions = null)
    {
        if (json.type != JSONType.object)
            throw malformedResponse(json, "Expected a decision response object.");
        if ("error" in json)
            throw new EndpointException("POST", "decisions", 0, "error", json.toString());
        if ("answers" !in json || json["answers"].type != JSONType.array)
            throw malformedResponse(json, "Expected a decision answers array.");
        if (json["answers"].array.length == 0)
            throw malformedResponse(json, "Decision response contains no answers.");
        if (questions.length > 0 && questions.length != json["answers"].array.length)
            throw malformedResponse(json, "Decision answer count does not match the questions.");

        Decision ret;
        ret.raw = json;
        foreach (i, entry; json["answers"].array)
        {
            if (entry.type != JSONType.object || "type" !in entry)
                throw malformedResponse(json, "Decision answers require a type.");

            DecisionAnswer answer;
            try
                answer = fromJSON!DecisionAnswer(entry);
            catch (Exception error)
            {
                throw new MalformedResponseException(
                    "Invalid decision answer: "~error.msg,
                    json.toString(),
                    entry.toString(),
                );
            }
            answer.raw = entry;
            final switch (answer.type)
            {
                case DecisionType.Predicate:
                    validateDecisionProbability(answer.probability, json);
                    break;
                case DecisionType.Choice:
                    if (answer.choice.type != JSONType.string && answer.choice.type != JSONType.true_
                        && answer.choice.type != JSONType.false_)
                        throw malformedResponse(json, "Decision choices must be strings or booleans.");
                    break;
                case DecisionType.Score:
                    if (!answer.score.isFinite || answer.score < 0)
                        throw malformedResponse(json, "Invalid decision score.");
                    break;
                case DecisionType.Refusal:
                    break;
            }

            if ("confidence" in entry)
                validateDecisionProbability(answer.confidence, json);

            foreach (ref probability; answer.probabilities)
                validateDecisionProbability(probability.probability, json);

            if (questions.length > 0)
            {
                if (answer.name != questions[i].name)
                    throw malformedResponse(json, "Decision answer name does not match its question.");
                if (answer.type != DecisionType.Refusal && answer.type != questions[i].type)
                    throw malformedResponse(json, "Decision answer type does not match its question.");

                if (answer.type == DecisionType.Choice)
                {
                    bool found;
                    foreach (ref option; questions[i].choices)
                        found = found || option.value == answer.choice;

                    if (!found)
                        throw malformedResponse(json, "Decision selected an unknown choice.");
                }
                else if (answer.type == DecisionType.Score
                    && (answer.score < 0 || answer.score > questions[i].levels.length - 1))
                    throw malformedResponse(json, "Decision score is outside its levels.");
            }

            ret.answers ~= answer;
        }

        ret.usage.modelName = "model" in json && json["model"].type == JSONType.string
            ? json["model"].str : name;
        if ("latency" in json)
        {
            try
                ret.usage.latency = fromJSON!float(json["latency"]);
            catch (Exception)
                throw malformedResponse(json, "Invalid decision latency.");
        }

        if ("usage" in json && json["usage"].type == JSONType.object)
        {
            JSONValue usage = json["usage"];
            ret.usage.promptTokens = readUint(usage, "input_tokens", null);
            ret.usage.completionTokens = readUint(usage, "output_tokens", null);
            ret.usage.totalTokens = readUint(usage, "total_tokens", null);
            if ("total_tokens" !in usage)
                ret.usage.totalTokens = ret.usage.promptTokens + ret.usage.completionTokens;

            if ("input_tokens_details" in usage && usage["input_tokens_details"].type == JSONType.object)
                ret.usage.cacheHits = readUint(usage["input_tokens_details"], "cached_tokens", null);

            if (ret.usage.cacheHits > ret.usage.promptTokens)
                throw malformedResponse(json, "Cached decision tokens exceed input tokens.");

            ret.usage.cacheMisses = ret.usage.promptTokens - ret.usage.cacheHits;
        }
        return ret;
    }

protected:
    static MalformedResponseException malformedResponse(JSONValue response, string message)
    {
        return new MalformedResponseException(message, response.toString());
    }

    static JSONValue decisionQuestions(DecisionQuestion[] questions)
    {
        if (questions.length == 0)
            throw new FormatException("Decisions require at least one question.");

        JSONValue ret = JSONValue.emptyArray;
        bool[string] names;
        foreach (ref question; questions)
        {
            if (question.instructions.length == 0 || question.type == DecisionType.Refusal)
                throw new FormatException("Decision questions require instructions and a non-refusal type.");

            JSONValue entry = JSONValue.emptyObject;
            entry["type"] = JSONValue(cast(string)question.type);
            entry["instructions"] = JSONValue(question.instructions);
            if (question.name.length > 0)
            {
                if (question.name in names)
                    throw new FormatException("Decision question names must be unique.");

                names[question.name] = true;
                entry["name"] = JSONValue(question.name);
            }

            if (question.type == DecisionType.Choice)
            {
                if (question.choices.length < 2)
                    throw new FormatException("Choice questions require at least two choices.");

                entry["choices"] = JSONValue.emptyArray;
                bool[string] values;
                foreach (ref option; question.choices)
                {
                    if (option.value.type != JSONType.string && option.value.type != JSONType.true_
                        && option.value.type != JSONType.false_)
                        throw new FormatException("Decision choice values must be strings or booleans.");
                    if (option.value.toString() in values)
                        throw new FormatException("Decision choice values must be unique.");

                    values[option.value.toString()] = true;
                    JSONValue choice = JSONValue(["value": option.value]);
                    if (option.description.length > 0)
                        choice["description"] = JSONValue(option.description);

                    entry["choices"].array ~= choice;
                }
            }
            else if (question.type == DecisionType.Score)
            {
                if (question.levels.length < 2)
                    throw new FormatException("Score questions require at least two levels.");

                entry["levels"] = JSONValue.emptyArray;
                foreach (ref level; question.levels)
                {
                    if (level.label.length == 0)
                        throw new FormatException("Decision score levels require labels.");

                    JSONValue score = JSONValue(["label": JSONValue(level.label)]);
                    if (level.description.length > 0)
                        score["description"] = JSONValue(level.description);

                    entry["levels"].array ~= score;
                }
            }

            ret.array ~= entry;
        }
        return ret;
    }

private:
    static void validateDecisionProbability(double probability, JSONValue response)
    {
        if (!probability.isFinite || probability < 0 || probability > 1)
            throw malformedResponse(response, "Decision probabilities must be between zero and one.");
    }

    static JSONValue normalizeSchema(JSONValue schema)
    {
        if (schema.type == JSONType.string && isSchemaType(schema.str))
        {
            JSONValue ret = JSONValue.emptyObject;
            ret["type"] = schema;
            return ret;
        }

        if (schema.type != JSONType.object)
            return schema;

        JSONValue ret = schema;
        foreach (keyword; [
            "properties",
            "patternProperties",
            "dependentSchemas",
            "$defs",
            "definitions",
        ])
        {
            if (keyword in ret && ret[keyword].type == JSONType.object)
                ret[keyword] = normalizeSchemaMap(ret[keyword]);
        }

        foreach (keyword; [
            "items",
            "contains",
            "additionalProperties",
            "propertyNames",
            "unevaluatedItems",
            "unevaluatedProperties",
            "not",
            "if",
            "then",
            "else",
        ])
        {
            if (keyword in ret)
                ret[keyword] = normalizeSchema(ret[keyword]);
        }

        foreach (keyword; [
            "allOf",
            "anyOf",
            "oneOf",
            "prefixItems",
        ])
        {
            if (keyword in ret && ret[keyword].type == JSONType.array)
                ret[keyword] = normalizeSchemaArray(ret[keyword]);
        }

        return ret;
    }

    static JSONValue normalizeSchemaMap(JSONValue schemas)
    {
        JSONValue ret = schemas;
        foreach (string key, JSONValue schema; schemas.object)
            ret[key] = normalizeSchema(schema);
        return ret;
    }

    static JSONValue normalizeSchemaArray(JSONValue schemas)
    {
        JSONValue ret = JSONValue.emptyArray;
        foreach (schema; schemas.array)
            ret.array ~= normalizeSchema(schema);
        return ret;
    }

    static bool isSchemaType(string value)
    {
        switch (value)
        {
        case "null":
        case "boolean":
        case "object":
        case "array":
        case "number":
        case "integer":
        case "string":
            return true;
        default:
            return false;
        }
    }

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
        if ("reasoning_content" in message && !message["reasoning_content"].isNull)
            parseContent(message["reasoning_content"], choice.text, choice.reasoning, true);

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
            else if (value.type == JSONType.uinteger)
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
        case "missing":
            return FinishReason.Missing;
        case "length":
            return FinishReason.Length;
        case "max_tokens":
            return FinishReason.MaxTokens;
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

    /// Throws if the JSON response contains an exception field.
    static void checkException(JSONValue json)
    {
        if ("error" !in json)
            return;

        if (json["error"].type == JSONType.string)
            throw new EndpointException("POST", "chat/completions", 0, "error", json["error"].str);
        else if (json["error"].type == JSONType.object && "message" in json["error"])
            throw new EndpointException("POST", "chat/completions", 0, "error", json["error"]["message"].str);
        else
            throw new EndpointException("POST", "chat/completions", 0, "error", json.toPrettyString());
    }
}
