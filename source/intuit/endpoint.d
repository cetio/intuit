/// High-level endpoint interface and request functions for completions and embeddings.
module intuit.endpoint;

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
    IModel[] available();

    /**
     * Gets a model by name, creating a placeholder if unknown.
     *
     * Params:
     *  name = The model name.
     *
     * Returns:
     *  The requested IModel.
     */
    IModel model(string name);

    /// Sends a raw completions request. Use `completions` instead.
    JSONValue _completions(IModel model, JSONValue payload);
    /// Sends a raw embeddings request. Use `embeddings` instead.
    JSONValue _embeddings(IModel model, JSONValue payload);
    /**
     * Sends a raw streaming completions request via SSE.
     *
     * Returns a CompletionStream that is fed by the endpoint's
     * background worker as events arrive.
     */
    CompletionStream _stream(IModel model, JSONValue payload);
}

/**
 * Send a completion request to the endpoint using a specific model.
 *
 * Enables for autoexec tool use if providing a Context instance.
 * All autoexec tools are always run if requested by the model.
 *
 * Params:
 *   ep = The endpoint to send the request to.
 *   model = The model to use for the completion.
 *   data = The input data. If this is a Context, it will be mutated
 *          in-place with the assistant response.
 *
 * Returns: The completion response.
 */
Completion completions(E, M, D)(E ep, M model, auto ref D data)
    if (is(E : IEndpoint) && is(M : IModel))
{
    JSONValue input = data.toJSON();

    JSONValue payload = model.completionsJSON(input, ep.tools);
    JSONValue resp = ep._completions(model, payload);
    Completion ret = model.parseCompletions(resp);

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
            return completions(ep, model, data);
    }

    return ret;
}

/**
 * Convenience overload that resolves the model by name before calling
 * completions.
 *
 * Params:
 *   ep = The endpoint to send the request to.
 *   model = The name of the model to use.
 *   data = The input data. If this is a Context, it will be mutated
 *          in-place with the assistant response.
 *
 * Returns: The completion response.
 */
Completion completions(E, D)(E ep, string model, auto ref D data)
    if (is(E : IEndpoint))
{
    return completions(ep, ep.model(model), data);
}

/**
 * Send a streaming completion request to the endpoint using a specific model.
 *
 * Params:
 *   ep = The endpoint to send the request to.
 *   model = The model to use for the completion.
 *   data = The input data. Not mutated for streaming.
 *
 * Returns: A CompletionStream for token-by-token consumption.
 */
CompletionStream streamCompletions(E, M, D)(E ep, M model, auto ref D data)
    if (is(E : IEndpoint) && is(M : IModel))
{
    JSONValue input = data.toJSON();

    JSONValue payload = model.completionsJSON(input, ep.tools);
    if ("stream" !in payload || payload["stream"].type != JSONType.true_)
        payload["stream"] = JSONValue(true);

    return ep._stream(model, payload);
}

/**
 * Convenience overload that resolves the model by name before calling
 * streamCompletions.
 *
 * Params:
 *   ep = The endpoint to send the request to.
 *   model = The name of the model to use.
 *   data = The input data.
 *
 * Returns: A CompletionStream for token-by-token consumption.
 */
CompletionStream streamCompletions(E, D)(E ep, string model, auto ref D data)
    if (is(E : IEndpoint))
{
    return streamCompletions(ep, ep.model(model), data);
}

/**
 * Request a single embedding vector from the endpoint.
 *
 * Params:
 *   ep = The endpoint to send the request to.
 *   model = The model to use for the embedding.
 *   data = The input data to embed.
 *
 * Returns: A single embedding vector.
 */
Embedding!T embeddings(T = float, E, M, D)(E ep, M model, D data)
    if (is(E : IEndpoint) && is(M : IModel)
        && (is(D == string) || !isArray!D))
{
    JSONValue payload = model.embeddingsJSON(data.toJSON());
    JSONValue resp = ep._embeddings(model, payload);
    JSONValue arr = model.parseEmbeddings(resp);

    Embedding!T ret;
    if (arr.type == JSONType.array && arr.array.length > 0)
        ret.value = toVector!T(arr.array[0]);
    return ret;
}

/**
 * Convenience overload that resolves the model by name before calling
 * embeddings.
 *
 * Params:
 *   ep = The endpoint to send the request to.
 *   model = The name of the model to use.
 *   data = The input data to embed.
 *
 * Returns: A single embedding vector.
 */
Embedding!T embeddings(T = float, E, D)(E ep, string model, D data)
    if (is(E : IEndpoint)
        && (is(D == string) || !isArray!D))
{
    return embeddings!T(ep, ep.model(model), data);
}

/**
 * Request embedding vectors for an array of inputs.
 *
 * Params:
 *   ep = The endpoint to send the request to.
 *   model = The model to use for the embeddings.
 *   data = An array of input data to embed.
 *
 * Returns: An array of embedding vectors.
 */
Embedding!T[] embeddings(T = float, E, M, D)(E ep, M model, D data)
    if (is(E : IEndpoint) && is(M : IModel)
        && isArray!D && !is(D == string))
{
    JSONValue payload = model.embeddingsJSON(data.toJSON());
    JSONValue resp = ep._embeddings(model, payload);
    JSONValue arr = model.parseEmbeddings(resp);

    Embedding!T[] ret;
    if (arr.type == JSONType.array)
    {
        ret.length = arr.array.length;
        foreach (i, v; arr.array)
            ret[i].value = toVector!T(v);
    }
    return ret;
}

/**
 * Convenience overload that resolves the model by name before calling
 * embeddings with an array of inputs.
 *
 * Params:
 *   ep = The endpoint to send the request to.
 *   model = The name of the model to use.
 *   data = An array of input data to embed.
 *
 * Returns: An array of embedding vectors.
 */
Embedding!T[] embeddings(T = float, E, D)(E ep, string model, D data)
    if (is(E : IEndpoint)
        && isArray!D && !is(D == string))
{
    return embeddings!T(ep, ep.model(model), data);
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
