/// High-level endpoint interface and request functions for completions and embeddings.
module intuit.provider;

public import intuit.provider.openai;
public import intuit.provider.claude;
public import intuit.provider.qwen;
public import intuit.provider.typesafe;

import intuit.context;
import intuit.exception :
    AuthException,
    EndpointException,
    MalformedResponseException,
    RateLimitException,
    RequestTimeoutException,
    TransportException;
import intuit.json : toJSON;
import intuit.model;
import intuit.response;
import intuit.tool;

import std.conv : to;
import std.json : JSONValue, JSONType, parseJSON;
import std.net.curl : CurlException, CurlTimeoutException, HTTP;
import std.string : assumeUTF;
import std.traits : isArray, isIntegral;
import core.time : Duration, MonoTime;

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

    JSONValue _decisions(ModelConfig cfg, JSONValue payload);

    void operationTimeout(Duration timeout);

    void connectTimeout(Duration timeout);
}

/**
 * Sends a JSON request using an HTTP client with the given headers.
 *
 * When `headers` is non-null, all pre-set headers on the HTTP client are
 * cleared and replaced with the provided map. When null, pre-set headers
 * are preserved.
 *
 * Params:
 *  http = The HTTP client.
 *  method = The HTTP method.
 *  url = The full request URL.
 *  headers = Request headers. Clears pre-set headers when non-null.
 *  payload = Optional JSON payload for requests with a body.
 *
 * Returns:
 *  The parsed JSON response.
 *
 * Throws:
 *  EndpointException subclasses on HTTP failures.
 *  TransportException subclasses on transport failures.
 *  MalformedResponseException if the response body is invalid JSON.
 */
package(intuit) JSONValue request(
    ref HTTP http,
    HTTP.Method method,
    string url,
    string[string] headers = null,
    JSONValue payload = JSONValue.init,
)
{
    MonoTime start = MonoTime.currTime;

    if (headers !is null)
    {
        http.clearRequestHeaders();
        foreach (string key, string value; headers)
            http.addRequestHeader(key, value);
    }

    http.url = url;
    http.method = method;

    const(ubyte)[] body;
    if (payload.type != JSONType.null_)
        body = cast(const(ubyte)[])payload.toString();

    if (method == HTTP.Method.post || method == HTTP.Method.put || method == HTTP.Method.patch)
    {
        size_t offset;
        http.contentLength = body.length;
        http.onSend = delegate size_t(void[] buffer) {
            if (offset >= body.length)
                return 0;

            size_t count = body.length - offset;
            if (count > buffer.length)
                count = buffer.length;

            buffer[0..count] = cast(void[])body[offset..offset + count];
            offset += count;
            return count;
        };
    }
    else if (method == HTTP.Method.del)
    {
        http.onSend = delegate size_t(void[] buffer) {
            return 0;
        };
    }
    else
        http.onSend = null;

    ushort status;
    string reason;
    ubyte[] responseBody;
    http.onReceiveStatusLine = delegate void(HTTP.StatusLine line) {
        status = line.code;
        reason = line.reason.idup;
    };
    http.onReceive = delegate size_t(ubyte[] chunk) {
        if (chunk !is null)
            responseBody ~= chunk;
        return chunk.length;
    };

    try
        http.perform();
    catch (CurlTimeoutException error)
        throw new RequestTimeoutException(method.to!string, url, error.msg);
    catch (CurlException error)
        throw new TransportException(method.to!string, url, error.msg);

    string content = responseBody is null ? null : responseBody.assumeUTF().idup;

    if (status == 401 || status == 403)
        throw new AuthException(method.to!string, url, status, reason, content);

    if (status == 429)
        throw new RateLimitException(method.to!string, url, status, reason, content);

    if (status < 200 || status >= 300)
        throw new EndpointException(method.to!string, url, status, reason, content);

    JSONValue ret;
    try
        ret = content.parseJSON();
    catch (Exception)
        throw new MalformedResponseException("Endpoint returned invalid JSON.", content);

    long usecs = (MonoTime.currTime - start).total!"usecs";
    ret["latency"] = JSONValue(usecs / 1000.0f);
    return ret;
}

/**
 * Send a completion request to the endpoint using a specific model name.
 *
 * Params:
 *   ep = The endpoint to send the request to.
 *   modelName = The name of the model to use.
 *   data = The input data. If this is a Context, it will be mutated
 *          in-place with the assistant response.
 *   maxToolRounds = Maximum automatically handled tool rounds. Defaults to 8.
 *
 * Returns: The completion response.
 */
Completion completions(E, D)(E ep, string modelName, auto ref D data, int maxToolRounds = 8)
    if (is(E : IEndpoint))
{
    if (maxToolRounds < 0)
        throw new Exception("maxToolRounds cannot be negative.");

    ModelConfig cfg = ep.config(modelName);
    static if (is(D == Context))
    {
        Completion ret = requestCompletion(ep, cfg, data);
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

            data.executePolicy(ep.tools, ret.choice.toolCalls);
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
            ret = requestCompletion(ep, cfg, data);
        }
        return ret;
    }
    else
    {
        JSONValue payload = cfg.buildPayload(data.toJSON(), ep.tools);
        JSONValue resp = ep._completions(cfg, payload);
        return cfg.parseResponse(resp);
    }
}

deprecated("The legacy completions API is deprecated. Use chat completions instead.")
Completion legacyCompletions(E)(E ep, JSONValue payload)
    if (is(E : OpenAI))
{
    return ep.legacyCompletions(payload);
}

Decision decisions(E, D)(
    E ep,
    string modelName,
    auto ref D data,
    DecisionQuestion[] questions,
)
    if (is(E : IEndpoint))
{
    ModelConfig cfg = ep.config(modelName);
    JSONValue payload = cfg.buildDecisionsPayload(data.toJSON(), questions);
    return cfg.parseDecisionsResponse(ep._decisions(cfg, payload), questions);
}

private Completion requestCompletion(E)(E ep, ModelConfig cfg, ref Context context)
    if (is(E : IEndpoint))
{
    JSONValue payload = cfg.buildPayload(context.toJSON(), ep.tools);
    JSONValue resp = ep._completions(cfg, payload);
    Completion ret = cfg.parseResponse(resp);
    context.assistant(ret);
    return ret;
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
