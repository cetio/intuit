/// Base model configuration class for model providers.
module intuit.model;

import intuit.exception : FormatException;
import intuit.response;
import intuit.tool;

import std.algorithm.searching : canFind;
import std.json : JSONValue, JSONType;
import std.math : isNaN;

/// Configuration and request/response logic for a model.
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
    JSONValue buildCompletionPayload(JSONValue input, ToolRegistry tools = ToolRegistry.init)
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
    JSONValue buildEmbeddingPayload(JSONValue input)
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
    Completion parseCompletionResponse(JSONValue json)
        => intuit.response.completion.parseCompletionResponse(json, name);

    /**
     * Parses a raw embeddings response.
     *
     * Params:
     *  json = The raw JSON response.
     *
     * Returns:
     *  A JSON array containing the embedding vectors.
     */
    JSONValue parseEmbeddingResponse(JSONValue json)
        => intuit.response.embedding.parseEmbeddingResponse(json);

    /**
     * Builds the decisions request payload.
     *
     * Params:
     *  input = The input data.
     *  questions = The decision questions to answer.
     *
     * Returns:
     *  The JSON payload for the decisions endpoint.
     */
    JSONValue buildDecisionPayload(JSONValue input, DecisionQuestion[] questions)
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
        ret["questions"] = intuit.response.decision.decisionQuestions(questions);
        return ret;
    }

    /**
     * Parses a raw decisions response.
     *
     * Params:
     *  json = The raw JSON response.
     *  questions = The decision questions used for validation.
     *
     * Returns:
     *  The parsed Decision.
     */
    Decision parseDecisionResponse(JSONValue json, DecisionQuestion[] questions = null)
        => intuit.response.decision.parseDecisionResponse(json, questions, name);

    deprecated("Use buildCompletionPayload instead.")
    alias buildPayload = buildCompletionPayload;

    deprecated("Use buildEmbeddingPayload instead.")
    alias buildEmbeddingsPayload = buildEmbeddingPayload;

    deprecated("Use buildDecisionPayload instead.")
    alias buildDecisionsPayload = buildDecisionPayload;

    deprecated("Use parseCompletionResponse instead.")
    alias parseResponse = parseCompletionResponse;

    deprecated("Use parseEmbeddingResponse instead.")
    alias parseEmbeddingsResponse = parseEmbeddingResponse;

    deprecated("Use parseDecisionResponse instead.")
    alias parseDecisionsResponse = parseDecisionResponse;

private:
    static JSONValue normalizeSchema(JSONValue schema)
    {
        if (schema.type == JSONType.string
            && canFind(["null", "boolean", "object", "array", "number", "integer", "string"], schema.str))
        {
            JSONValue ret = JSONValue.emptyObject;
            ret["type"] = schema;
            return ret;
        }

        if (schema.type != JSONType.object)
            return schema;

        foreach (keyword; ["properties", "patternProperties", "dependentSchemas", "$defs", "definitions"])
            if (keyword in schema && schema[keyword].type == JSONType.object)
                foreach (key; schema[keyword].object.keys)
                    schema[keyword][key] = normalizeSchema(schema[keyword][key]);

        foreach (keyword; [
            "items", "contains", "additionalProperties", "propertyNames", "unevaluatedItems",
            "unevaluatedProperties", "not", "if", "then", "else",
        ])
            if (keyword in schema)
                schema[keyword] = normalizeSchema(schema[keyword]);

        foreach (keyword; ["allOf", "anyOf", "oneOf", "prefixItems"])
            if (keyword in schema && schema[keyword].type == JSONType.array)
                foreach (i; 0..schema[keyword].array.length)
                    schema[keyword][i] = normalizeSchema(schema[keyword][i]);
        return schema;
    }
}
