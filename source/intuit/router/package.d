/// Router interface and request functions operating on a maintained context.
module intuit.router;

public import intuit.router.local;
public import intuit.router.openrouter;

import intuit.context;
import intuit.model;
import intuit.response;
import intuit.tool;
import conductor.serialize : toJSON;

import std.json : JSONValue, JSONType;
import std.traits : isArray, isIntegral;

/// Interface for routers that select among endpoints behind a single active model.
interface IRouter
{
    /// Gets the router name.
    ref string name();
    /// Gets the tool registry.
    ref ToolRegistry tools();
    /// Gets the maintained conversation context.
    ref Context context();

    /// Gets the active model name, or null when none is set.
    string active();
    /**
     * Sets the active model.
     *
     * Updates the context compactor token limit to the model's context window
     * and preserves existing messages.
     *
     * Params:
     *  name = The model name to activate.
     */
    void active(string name);

    /// Resolves the active model to an IModel, throwing when none is set.
    IModel model();

    /// Sends a raw completions request. Use `completions` instead.
    JSONValue _completions(JSONValue payload);
    /// Sends a raw embeddings request. Use `embeddings` instead.
    JSONValue _embeddings(JSONValue payload);
    /// Sends a raw streaming completions request. Use `streamCompletions` instead.
    CompletionStream _stream(JSONValue payload);
}

/**
 * Send a completion request using the router's maintained context.
 *
 * All autoexec tools are always run if requested by the model.
 *
 * Params:
 *   router = The router to send the request through.
 *
 * Returns: The completion response.
 */
Completion completions(R)(R router)
    if (is(R : IRouter))
{
    if (router.active is null)
        throw new Exception("Router has no active model set.");
    return runCompletion(router);
}

/**
 * Send a completion request after appending data as a user turn.
 *
 * Params:
 *   router = The router to send the request through.
 *   data = The input data appended to the maintained context.
 *
 * Returns: The completion response.
 */
Completion completions(R, D)(R router, auto ref D data)
    if (is(R : IRouter))
{
    if (router.active is null)
        throw new Exception("Router has no active model set.");
    router.context.user(data);
    return runCompletion(router);
}

/// Runs a completion against the router's active model and maintained context.
private Completion runCompletion(R)(R router)
    if (is(R : IRouter))
{
    IModel model = router.model();

    JSONValue payload = model.completionsJSON(router.context.toJSON(), router.tools);
    JSONValue resp = router._completions(payload);
    Completion ret = model.parseCompletions(resp);

    router.context.assistant(ret);

    Choice first = ret.choice(0);
    bool cycle = first.toolCalls.length > 0;
    foreach (call; first.toolCalls)
    {
        Tool tool = router.tools.get(call.name);
        cycle &= tool.autoexec;
        if (!tool.autoexec)
            continue;

        JSONValue result;
        try
            result = tool.impl(call.arguments);
        catch (Exception ex)
            result = JSONValue("Error: "~ex.msg);
        string serialized = result.type == JSONType.string ? result.str : result.toString();
        router.context.tool(call.id, serialized);
    }

    if (cycle)
        return runCompletion(router);

    return ret;
}

/**
 * Send a streaming completion request using the router's maintained context.
 *
 * Params:
 *   router = The router to send the request through.
 *
 * Returns: A CompletionStream for token-by-token consumption.
 */
CompletionStream streamCompletions(R)(R router)
    if (is(R : IRouter))
{
    if (router.active is null)
        throw new Exception("Router has no active model set.");
    return runStream(router);
}

/**
 * Send a streaming completion request after appending data as a user turn.
 *
 * Params:
 *   router = The router to send the request through.
 *   data = The input data appended to the maintained context.
 *
 * Returns: A CompletionStream for token-by-token consumption.
 */
CompletionStream streamCompletions(R, D)(R router, auto ref D data)
    if (is(R : IRouter))
{
    if (router.active is null)
        throw new Exception("Router has no active model set.");
    router.context.user(data);
    return runStream(router);
}

/// Builds and dispatches a streaming request against the active model.
private CompletionStream runStream(R)(R router)
    if (is(R : IRouter))
{
    IModel model = router.model();

    JSONValue payload = model.completionsJSON(router.context.toJSON(), router.tools);
    if ("stream" !in payload || payload["stream"].type != JSONType.true_)
        payload["stream"] = JSONValue(true);

    return router._stream(payload);
}

/**
 * Request a single embedding vector using the router's active model.
 *
 * Params:
 *   router = The router to send the request through.
 *   data = The input data to embed.
 *
 * Returns: A single embedding vector.
 */
Embedding!T embeddings(T = float, R, D)(R router, D data)
    if (is(R : IRouter) && (is(D == string) || !isArray!D))
{
    if (router.active is null)
        throw new Exception("Router has no active model set.");

    IModel model = router.model();
    JSONValue payload = model.embeddingsJSON(data.toJSON());
    JSONValue resp = router._embeddings(payload);
    JSONValue arr = model.parseEmbeddings(resp);

    Embedding!T ret;
    if (arr.type == JSONType.array && arr.array.length > 0)
        ret.value = toVector!T(arr.array[0]);
    return ret;
}

/**
 * Request embedding vectors for an array of inputs using the active model.
 *
 * Params:
 *   router = The router to send the request through.
 *   data = An array of input data to embed.
 *
 * Returns: An array of embedding vectors.
 */
Embedding!T[] embeddings(T = float, R, D)(R router, D data)
    if (is(R : IRouter) && isArray!D && !is(D == string))
{
    if (router.active is null)
        throw new Exception("Router has no active model set.");

    IModel model = router.model();
    JSONValue payload = model.embeddingsJSON(data.toJSON());
    JSONValue resp = router._embeddings(payload);
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
