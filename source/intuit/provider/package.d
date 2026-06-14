/// High-level endpoint interface and request functions for completions and embeddings.
module intuit.provider;

public import intuit.provider.openai;
public import intuit.provider.claude;
public import intuit.provider.qwen;

import intuit.context;
import intuit.model;
import intuit.response;
import intuit.tool;
import conductor.serialize : toJSON;

import std.json : JSONValue, JSONType;
import std.traits : isArray, isIntegral;

/// Interface for LLM endpoint implementations.
interface IEndpoint
{
    /// Gets the endpoint name.
    ref string name();
    /// Gets the base URL.
    ref string url();
    /// Gets the API key.
    ref string key();
    /// Gets the tool registry.
    ref ToolRegistry tools();

    /// Fetches available models from the endpoint.
    ModelConfig[] available();

    /**
     * Gets a model config by name or creating a new config if unset.
     *
     * Params:
     *  modelName = The model name.
     *
     * Returns:
     *  The requested ModelConfig.
     */
    ModelConfig config(string modelName);

    /// Gets all stored model configs.
    ModelConfig[] configs();

    /// Sends a raw completions request. Use `completions` instead.
    JSONValue _completions(ModelConfig cfg, JSONValue payload);
    /// Sends a raw embeddings request. Use `embeddings` instead.
    JSONValue _embeddings(ModelConfig cfg, JSONValue payload);
    /**
     * Sends a raw streaming completions request via SSE.
     *
     * Returns a CompletionStream that is fed by the endpoint's
     * background worker as events arrive.
     */
    CompletionStream _stream(ModelConfig cfg, JSONValue payload);
}

/**
 * Send a completion request to the endpoint using a specific model name.
 *
 * Enables for autoexec tool use if providing a Context instance.
 * All autoexec tools are always run if requested by the model.
 *
 * Params:
 *   ep = The endpoint to send the request to.
 *   modelName = The name of the model to use.
 *   data = The input data. If this is a Context, it will be mutated
 *          in-place with the assistant response.
 *
 * Returns: The completion response.
 */
Completion completions(E, D)(E ep, string modelName, auto ref D data)
    if (is(E : IEndpoint))
{
    ModelConfig cfg = ep.config(modelName);
    JSONValue input = data.toJSON();

    JSONValue payload = cfg.buildPayload(input, ep.tools);
    JSONValue resp = ep._completions(cfg, payload);
    Completion ret = cfg.parseResponse(resp);

    static if (is(D == Context))
    {
        data.assistant(ret);

        Choice first = ret.choice(0);
        bool cycle = first.toolCalls.length > 0;
        foreach (call; first.toolCalls)
        {
            Tool tool = ep.tools.get(call.name);
            cycle &= tool.autoexec;
            if (!tool.autoexec)
                continue;

            JSONValue result;
            try
                result = tool.impl(call.arguments);
            catch (Exception ex)
                result = JSONValue("Error: "~ex.msg);
            string serialized = result.type == JSONType.string ? result.str : result.toString();
            data.tool(call.id, serialized);
        }

        if (cycle)
            return completions(ep, modelName, data);
    }

    return ret;
}

/**
 * Send a streaming completion request to the endpoint using a specific model name.
 *
 * Params:
 *   ep = The endpoint to send the request to.
 *   modelName = The name of the model to use.
 *   data = The input data. Not mutated for streaming.
 *
 * Returns: A CompletionStream for token-by-token consumption.
 */
CompletionStream streamCompletions(E, D)(E ep, string modelName, auto ref D data)
    if (is(E : IEndpoint))
{
    ModelConfig cfg = ep.config(modelName);
    JSONValue input = data.toJSON();

    JSONValue payload = cfg.buildPayload(input, ep.tools);
    if ("stream" !in payload || payload["stream"].type != JSONType.true_)
        payload["stream"] = JSONValue(true);

    return ep._stream(cfg, payload);
}

/**
 * Request a single embedding vector from the endpoint.
 *
 * Params:
 *   ep = The endpoint to send the request to.
 *   modelName = The name of the model to use.
 *   data = The input data to embed.
 *
 * Returns: A single embedding vector.
 */
Embedding!T embeddings(T = float, E, D)(E ep, string modelName, D data)
    if (is(E : IEndpoint)
        && (is(D == string) || !isArray!D))
{
    ModelConfig cfg = ep.config(modelName);
    JSONValue payload = cfg.buildEmbeddingsPayload(data.toJSON());
    JSONValue resp = ep._embeddings(cfg, payload);
    JSONValue arr = cfg.parseEmbeddingsResponse(resp);

    Embedding!T ret;
    if (arr.type == JSONType.array && arr.array.length > 0)
        ret.value = toVector!T(arr.array[0]);
    return ret;
}

/**
 * Request embedding vectors for an array of inputs.
 *
 * Params:
 *   ep = The endpoint to send the request to.
 *   modelName = The name of the model to use.
 *   data = An array of input data to embed.
 *
 * Returns: An array of embedding vectors.
 */
Embedding!T[] embeddings(T = float, E, D)(E ep, string modelName, D data)
    if (is(E : IEndpoint)
        && isArray!D && !is(D == string))
{
    ModelConfig cfg = ep.config(modelName);
    JSONValue payload = cfg.buildEmbeddingsPayload(data.toJSON());
    JSONValue resp = ep._embeddings(cfg, payload);
    JSONValue arr = cfg.parseEmbeddingsResponse(resp);

    Embedding!T[] ret;
    if (arr.type == JSONType.array)
    {
        ret.length = arr.array.length;
        foreach (i, v; arr.array)
            ret[i].value = toVector!T(v);
    }
    return ret;
}

/// Converts a JSON array into a typed vector.
private T[] toVector(T)(JSONValue arr)
{
    if (arr.type != JSONType.array)
        return null;

    T[] ret = new T[](arr.array.length);
    foreach (i, v; arr.array)
    {
        if (v.type == JSONType.null_)
            continue;

        static if (isIntegral!T)
        {
            if (v.type == JSONType.integer)
                ret[i] = cast(T)v.integer;
            else if (v.type == JSONType.uinteger)
                ret[i] = cast(T)v.uinteger;
            else if (v.type == JSONType.float_)
                ret[i] = cast(T)v.floating;
        }
        else
        {
            if (v.type == JSONType.float_)
                ret[i] = cast(T)v.floating;
            else if (v.type == JSONType.integer)
                ret[i] = cast(T)v.integer;
            else if (v.type == JSONType.uinteger)
                ret[i] = cast(T)v.uinteger;
        }
    }
    return ret;
}
