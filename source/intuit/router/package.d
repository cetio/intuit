/// Router interface and request functions operating on a maintained context.
module intuit.router;

public import intuit.router.details;
public import intuit.router.litellm;
public import intuit.router.openrouter;

import intuit.context;
import intuit.json : toJSON;
import intuit.model;
import intuit.response;
import intuit.tool : ToolRegistry;

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
     *  modelName = The model name to activate.
     */
    void active(string modelName);

    /// Resolves the active model to a ModelConfig, throwing when none is set.
    ModelConfig config();
    /// Gets or creates a ModelConfig for any model by name.
    ModelConfig config(string modelName);
    /// Gets all stored model configs.
    ModelConfig[] configs();
    /// Gets the capability catalog for all known models.
    ModelDetails[string] catalog();

    /// Sends a raw completions request. Use `completions` instead.
    JSONValue _completions(JSONValue payload);
    /// Sends a raw embeddings request. Use `embeddings` instead.
    JSONValue _embeddings(JSONValue payload);

    /// Re-fetches the model catalog. Implementation defined, and may have additional behavior.
    void refresh();
}

/**
 * Send a completion request using the router's maintained context.
 *
 * Params:
 *   router = The router to send the request through.
 *   maxToolRounds = Maximum automatically handled tool rounds. Defaults to 8.
 *
 * Returns: The completion response.
 */
Completion completions(R)(R router, int maxToolRounds = 8)
    if (is(R : IRouter))
{
    return completeWithPolicy(router, maxToolRounds);
}

private Completion completeWithPolicy(R)(R router, int maxToolRounds)
    if (is(R : IRouter))
{
    if (maxToolRounds < 0)
        throw new Exception("maxToolRounds cannot be negative.");

    Completion ret = requestCompletion(router);
    if (maxToolRounds == 0)
    {
        foreach (i; 0..(ret.choices.length == 0 ? 0 : ret.choice.toolCalls.length))
        {
            ret.choice.toolCalls[i].policyResult.status = ToolPolicyStatus.Pending;
            ret.choice.toolCalls[i].policyResult.message =
                "Automatic tool policy execution disabled by maxToolRounds.";
        }
        return ret;
    }

    foreach (round; 0..maxToolRounds)
    {
        if (ret.choices.length == 0 || ret.choice.toolCalls.length == 0)
            return ret;

        router.context.executePolicy(router.tools, ret.choice.toolCalls);
        bool requiresManualHandling;
        foreach (call; ret.choice.toolCalls)
        {
            if (call.policyResult.status == ToolPolicyStatus.Failed
                || call.policyResult.status == ToolPolicyStatus.Pending)
            {
                requiresManualHandling = true;
                break;
            }
        }
        if (requiresManualHandling || round + 1 == maxToolRounds)
            return ret;
        ret = requestCompletion(router);
    }
    return ret;
}

private Completion requestCompletion(R)(R router)
    if (is(R : IRouter))
{
    if (router.active is null)
        throw new Exception("Router has no active model set.");

    ModelConfig cfg = router.config();
    JSONValue payload = cfg.buildPayload(router.context.toJSON(), router.tools);
    JSONValue resp = router._completions(payload);
    Completion ret = cfg.parseResponse(resp);
    router.context.assistant(ret);
    return ret;
}

Completion completions(R, D)(R router, auto ref D data, int maxToolRounds = 8)
    if (is(R : IRouter))
{
    if (maxToolRounds < 0)
        throw new Exception("maxToolRounds cannot be negative.");
    if (router.active is null)
        throw new Exception("Router has no active model set.");
    router.context.user(data);
    return completeWithPolicy(router, maxToolRounds);
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

    ModelConfig cfg = router.config();
    JSONValue payload = cfg.buildEmbeddingsPayload(data.toJSON());
    JSONValue resp = router._embeddings(payload);
    JSONValue arr = cfg.parseEmbeddingsResponse(resp);

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

    ModelConfig cfg = router.config();
    JSONValue payload = cfg.buildEmbeddingsPayload(data.toJSON());
    JSONValue resp = router._embeddings(payload);
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
