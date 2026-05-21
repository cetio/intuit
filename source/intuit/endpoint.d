module intuit.endpoint;

import intuit.context;
import intuit.model;
import intuit.response;
import intuit.tool;
import conductor.http : toJSON;
import std.json : JSONValue, JSONType;
import std.traits : isArray, isIntegral;

interface IEndpoint
{
    ref string name();
    ref string url();
    ref string key();
    ref ToolRegistry tools();

    IModel[] available();
    IModel model(string name);

    JSONValue _completions(IModel model, JSONValue payload);
    JSONValue _embeddings(IModel model, JSONValue payload);
}

/**
 * Send a completion request to the endpoint using a specific model.
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
    static if (is(D == Context))
        JSONValue input = data.messages;
    else
        JSONValue input = data.toJSON();

    JSONValue payload = model.completionsJSON(input, ep.tools);
    JSONValue resp = ep._completions(model, payload);
    Completion ret = model.parseCompletions(resp);

    static if (is(D == Context))
    {
        Choice first = ret.choice(0);
        data.assistant(first.text, first.toolCalls);
        if (ret.choices.length > 1)
            ret.choices = ret.choices[0..1];

        bool hasNonAutoexec;
        foreach (call; first.toolCalls)
        {
            Tool tool = ep.tools.get(call.name);
            if (!tool.autoexec)
            {
                hasNonAutoexec = true;
                break;
            }
        }

        if (hasNonAutoexec)
            return ret;

        foreach (call; first.toolCalls)
        {
            Tool tool = ep.tools.get(call.name);
            JSONValue result = tool.impl(call.arguments);
            string serialized = result.type == JSONType.string ? result.str : result.toString();
            data.tool(call.id, serialized);
        }

        if (first.toolCalls.length > 0)
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

private T[] toVector(T)(JSONValue arr)
{
    if (arr.type != JSONType.array)
        return null;

    T[] ret = new T[](arr.array.length);
    foreach (i, v; arr.array)
    {
        static if (isIntegral!T)
        {
            if (v.type == JSONType.integer)
                ret[i] = cast(T)v.integer;
            else if (v.type == JSONType.uinteger)
                ret[i] = cast(T)v.uinteger;
            else
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
