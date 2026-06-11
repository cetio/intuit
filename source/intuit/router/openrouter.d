/// OpenRouter router with dynamic model discovery over the OpenAI-compatible API.
module intuit.router.openrouter;

import intuit.context;
import intuit.error : EndpointError;
import intuit.model;
import intuit.provider.openai.model : OpenAIModel;
import intuit.response;
import intuit.router;
import intuit.stream.sse : SSEParser;
import intuit.tool;
import conductor.http : Response, send;

import core.thread : Thread;
import std.conv : to;
import std.json : JSONType, JSONValue, parseJSON;
import std.net.curl : HTTP;
import std.string : assumeUTF, join;

/// Dynamic metadata for a single OpenRouter model, populated from `/models`.
struct ModelDetails
{
    /// The model slug, e.g. "openai/gpt-4o".
    string id;
    /// Human-readable display name.
    string name;
    /// Model description text.
    string description;
    /// Total context window in tokens; drives the compactor token limit.
    size_t contextLength;
    /// Maximum tokens the top provider can generate in a single response.
    size_t maxCompletionTokens;
    /// Supported input modalities, e.g. ["text", "image"].
    string[] inputModalities;
    /// Supported output modalities, e.g. ["text"].
    string[] outputModalities;
    /// OpenAI-compatible parameters the model accepts, e.g. ["tools", "temperature"].
    string[] supportedParameters;
    /// Cost in USD per input token.
    double promptCost;
    /// Cost in USD per output token.
    double completionCost;
}

/**
 * Router backed by OpenRouter's OpenAI-compatible API.
 *
 * Unlike LocalRouter, the model catalog is discovered dynamically from the
 * `/models` endpoint rather than statically defined. Models are served through
 * OpenAIModel since OpenRouter normalizes every provider to the OpenAI schema.
 */
class OpenRouter : IRouter
{
private:
    string _name;
    string _url;
    string _key;
    ToolRegistry _tools;
    Context _context;
    string _active;
    IModel _activeModel;
    HTTP _http;
    ModelDetails[string] _catalog;

    // OpenRouter request decorations and identifying headers.
    string _referer;
    string _title;
    string[] _categories;
    string[] _fallbackModels;
    string _route;
    string[] _transforms;
    JSONValue _provider;
    bool _hasProvider;
    JSONValue _plugins;
    bool _hasPlugins;
    bool _includeReasoning;

public:
    /**
     * Constructs an OpenRouter router.
     *
     * Params:
     *  key = The OpenRouter API key.
     *  url = The base URL, defaulting to the public OpenRouter host.
     *  name = Display name for the router.
     */
    this(string key, string url = "https://openrouter.ai", string name = "OpenRouter")
    {
        this._name = name;
        this._url = normalizeBaseUrl(url);
        this._key = key;
        this._http = HTTP();
        this._context.compactor = new Compactor();
    }

    override ref string name()
        => _name;

    override ref ToolRegistry tools()
        => _tools;

    override ref Context context()
        => _context;

    override string active()
        => _active;

    override void active(string name)
    {
        // The catalog is huge and dynamic, so only fetch it once on demand. Aliases
        // like "openrouter/auto" or "~openai/gpt-latest" may be absent; we still allow
        // them and simply leave the compactor limit untouched when metadata is missing.
        // NOTE: the Auto Router resolves the real model per request in the response
        // "model" field. We could pin the compactor limit to that resolved model once
        // the response types expose it.
        if (name !in _catalog && _catalog.length == 0)
            refresh();

        _active = name;
        _activeModel = new OpenAIModel(name, null);
        if (auto details = name in _catalog)
            _context.compactor.maxTokens = details.contextLength;
    }

    override IModel model()
    {
        if (_active is null)
            throw new Exception("Router has no active model set.");
        return _activeModel;
    }

    override JSONValue _completions(JSONValue payload)
        => request(HTTP.Method.post, "chat/completions", decorate(payload));

    override JSONValue _embeddings(JSONValue payload)
    {
        if (_hasProvider)
            payload["provider"] = _provider;
        return request(HTTP.Method.post, "embeddings", payload);
    }

    override CompletionStream _stream(JSONValue payload)
    {
        // NOTE: this SSE pump is copied almost verbatim from OpenAI._stream and
        // Claude._stream. The transport (header setup, onSend/onReceive wiring,
        // status capture, worker thread) should be extracted into a shared HTTP
        // streaming helper that takes a per-provider event handler delegate.
        IModel streamModel = model();
        string target = resolve("chat/completions");

        string[string] headers = requestHeaders();
        headers["Accept"] = "text/event-stream";

        CompletionStream stream = new CompletionStream(streamModel.name, null);

        HTTP http = HTTP();
        http.clearRequestHeaders();
        http.url = target;
        http.method = HTTP.Method.post;
        foreach (key, value; headers)
            http.addRequestHeader(key, value);
        http.addRequestHeader("Content-Type", "application/json");

        string bodyStr = decorate(payload).toString();
        http.contentLength = bodyStr.length;
        size_t offset;
        http.onSend = delegate size_t(void[] buffer) {
            if (offset >= bodyStr.length)
                return 0;
            size_t count = bodyStr.length - offset;
            if (count > buffer.length)
                count = buffer.length;
            buffer[0..count] = cast(void[])bodyStr[offset..offset + count];
            offset += count;
            return count;
        };

        ushort status;
        string reason;
        http.onReceiveStatusLine = (HTTP.StatusLine line) {
            status = line.code;
            reason = line.reason.idup;
        };

        SSEParser parser = new SSEParser();
        http.onReceive = (ubyte[] chunk) {
            try
            {
                auto events = parser.feed(chunk);
                foreach (event; events)
                {
                    if (event.data == "[DONE]")
                    {
                        stream.complete = true;
                        continue;
                    }

                    try
                    {
                        JSONValue json = event.data.parseJSON();
                        Completion completion = streamModel.parseCompletions(json);
                        stream.update(completion);
                        if (stream.callback !is null)
                            stream.callback(completion);
                    }
                    catch (Exception ex)
                    {
                        stream.error = ex;
                        stream.complete = true;
                    }
                }
            }
            catch (Exception ex)
            {
                stream.error = ex;
                stream.complete = true;
            }
            return chunk.length;
        };

        void doStream()
        {
            try
            {
                http.perform();

                if (status < 200 || status >= 300)
                {
                    stream.error = new EndpointError(
                        "POST", target, status, reason, null, "Streaming request failed."
                    );
                    stream.complete = true;
                    return;
                }

                auto finalEvents = parser.flush();
                foreach (event; finalEvents)
                {
                    if (event.data == "[DONE]")
                    {
                        stream.complete = true;
                        continue;
                    }

                    try
                    {
                        JSONValue json = event.data.parseJSON();
                        Completion completion = streamModel.parseCompletions(json);
                        stream.update(completion);
                        if (stream.callback !is null)
                            stream.callback(completion);
                    }
                    catch (Exception ex)
                    {
                        stream.error = ex;
                    }
                }
                stream.complete = true;
            }
            catch (Exception ex)
            {
                stream.error = ex;
                stream.complete = true;
            }
        }

        stream.commence((CompletionStream) { new StreamThread(&doStream).start(); });
        return stream;
    }

    /// Re-fetches the model catalog from the `/models` endpoint.
    void refresh()
    {
        JSONValue json = request(HTTP.Method.get, "models");
        _catalog = null;
        if ("data" in json && json["data"].type == JSONType.array)
        {
            foreach (item; json["data"].array)
            {
                ModelDetails details = parseDetails(item);
                if (details.id.length > 0)
                    _catalog[details.id] = details;
            }
        }
    }

    /// Lists every discovered model, fetching the catalog if it is empty.
    ModelDetails[] available()
    {
        if (_catalog.length == 0)
            refresh();

        ModelDetails[] ret;
        foreach (details; _catalog.byValue)
            ret ~= details;
        return ret;
    }

    /// Gets the discovered details for a model, fetching the catalog if needed.
    ModelDetails details(string name)
    {
        if (name !in _catalog && _catalog.length == 0)
            refresh();
        if (auto found = name in _catalog)
            return *found;
        throw new Exception("Model not found in OpenRouter catalog: "~name);
    }

    /// Gets or sets the HTTP-Referer used to identify the app to OpenRouter.
    ref string referer()
        => _referer;

    /// ditto
    OpenRouter referer(string val)
    {
        _referer = val;
        return this;
    }

    /// Gets or sets the X-Title used to identify the app to OpenRouter.
    ref string title()
        => _title;

    /// ditto
    OpenRouter title(string val)
    {
        _title = val;
        return this;
    }

    /// Gets or sets the marketplace categories sent via X-OpenRouter-Categories.
    ref string[] categories()
        => _categories;

    /// ditto
    OpenRouter categories(string[] val)
    {
        _categories = val;
        return this;
    }

    /// Gets or sets the fallback model slugs sent as the "models" field.
    ref string[] fallbackModels()
        => _fallbackModels;

    /// ditto
    OpenRouter fallbackModels(string[] val)
    {
        _fallbackModels = val;
        return this;
    }

    /// Gets or sets the routing strategy, e.g. "fallback".
    ref string route()
        => _route;

    /// ditto
    OpenRouter route(string val)
    {
        _route = val;
        return this;
    }

    /// Gets or sets the message transforms, e.g. ["middle-out"].
    ref string[] transforms()
        => _transforms;

    /// ditto
    OpenRouter transforms(string[] val)
    {
        _transforms = val;
        return this;
    }

    /// Gets or sets whether to request reasoning tokens in responses.
    ref bool includeReasoning()
        => _includeReasoning;

    /// ditto
    OpenRouter includeReasoning(bool val)
    {
        _includeReasoning = val;
        return this;
    }

    /// Sets the provider preferences object merged into every request.
    OpenRouter provider(JSONValue val)
    {
        _provider = val;
        _hasProvider = true;
        return this;
    }

    /// Sets the plugins array merged into chat requests.
    OpenRouter plugins(JSONValue val)
    {
        _plugins = val;
        _hasPlugins = true;
        return this;
    }

private:
    /// Injects OpenRouter-only chat fields into a request payload.
    JSONValue decorate(JSONValue payload)
    {
        if (_fallbackModels.length > 0)
        {
            JSONValue arr = JSONValue.emptyArray;
            foreach (slug; _fallbackModels)
                arr.array ~= JSONValue(slug);
            payload["models"] = arr;
        }
        if (_route.length > 0)
            payload["route"] = JSONValue(_route);
        if (_transforms.length > 0)
        {
            JSONValue arr = JSONValue.emptyArray;
            foreach (transform; _transforms)
                arr.array ~= JSONValue(transform);
            payload["transforms"] = arr;
        }
        if (_hasPlugins)
            payload["plugins"] = _plugins;
        if (_hasProvider)
            payload["provider"] = _provider;
        if (_includeReasoning)
            payload["include_reasoning"] = JSONValue(true);
        return payload;
    }

    /// Sends an HTTP request to the endpoint and parses the JSON response.
    JSONValue request(HTTP.Method method, string tail, JSONValue payload = JSONValue.init)
    {
        string target = resolve(tail);
        Response response;

        if (payload.type == JSONType.null_)
            response = send(_http, method, target, null, null, requestHeaders());
        else
        {
            response = send(
                _http,
                method,
                target,
                cast(const(ubyte)[])payload.toString(),
                "application/json",
                requestHeaders(),
            );
        }

        string content = response.content is null ? null : response.content.assumeUTF().idup;
        if (response.status < 200 || response.status >= 300)
            throw new EndpointError(method.to!string, target, response.status, response.reason, content);

        try
            return content.parseJSON();
        catch (Exception)
            throw new EndpointError(
                method.to!string,
                target,
                response.status,
                response.reason,
                content,
                "Endpoint returned invalid JSON.",
            );
    }

    /// Builds request headers including authorization and optional app identity.
    string[string] requestHeaders()
    {
        string[string] headers;
        if (_key !is null && _key.length > 0)
            headers["Authorization"] = "Bearer "~_key;
        if (_referer.length > 0)
            headers["HTTP-Referer"] = _referer;
        if (_title.length > 0)
            headers["X-Title"] = _title;
        if (_categories.length > 0)
            headers["X-OpenRouter-Categories"] = _categories.join(",");
        return headers;
    }

    /// Constructs a full route from a path tail under the OpenRouter API prefix.
    string resolve(string tail)
    {
        while (tail.length > 0 && tail[0] == '/')
            tail = tail[1..$];
        return _url~"/api/v1/"~tail;
    }

    /// Normalizes a base URL by stripping trailing slashes and API suffixes.
    static string normalizeBaseUrl(string url)
    {
        string ret = url;
        while (ret.length > 0 && ret[$-1] == '/')
            ret = ret[0..$-1];
        if (ret.length >= 7 && ret[$-7..$] == "/api/v1")
            ret = ret[0..$-7];
        else if (ret.length >= 3 && ret[$-3..$] == "/v1")
            ret = ret[0..$-3];
        return ret;
    }

    /// Parses a single `/models` entry into ModelDetails.
    static ModelDetails parseDetails(JSONValue item)
    {
        ModelDetails ret;
        if (item.type != JSONType.object)
            return ret;

        ret.id = "id" in item ? item["id"].str : null;
        ret.name = "name" in item ? item["name"].str : null;
        ret.description = "description" in item ? item["description"].str : null;
        ret.contextLength = jsonSize(item, "context_length");

        if ("top_provider" in item && item["top_provider"].type == JSONType.object)
        {
            JSONValue provider = item["top_provider"];
            if (ret.contextLength == 0)
                ret.contextLength = jsonSize(provider, "context_length");
            ret.maxCompletionTokens = jsonSize(provider, "max_completion_tokens");
        }

        if ("architecture" in item && item["architecture"].type == JSONType.object)
        {
            JSONValue architecture = item["architecture"];
            ret.inputModalities = jsonStrings(architecture, "input_modalities");
            ret.outputModalities = jsonStrings(architecture, "output_modalities");
        }

        ret.supportedParameters = jsonStrings(item, "supported_parameters");

        if ("pricing" in item && item["pricing"].type == JSONType.object)
        {
            JSONValue pricing = item["pricing"];
            ret.promptCost = jsonDouble(pricing, "prompt");
            ret.completionCost = jsonDouble(pricing, "completion");
        }

        return ret;
    }

    /// Reads an integral field from a JSON object, accepting numbers and strings.
    static size_t jsonSize(JSONValue obj, string key)
    {
        if (key !in obj)
            return 0;

        JSONValue value = obj[key];
        switch (value.type)
        {
        case JSONType.integer:
            return cast(size_t)value.integer;
        case JSONType.uinteger:
            return cast(size_t)value.uinteger;
        case JSONType.float_:
            return cast(size_t)value.floating;
        case JSONType.string:
            try
                return value.str.to!size_t;
            catch (Exception)
                return 0;
        default:
            return 0;
        }
    }

    /// Reads a floating field from a JSON object, accepting numbers and strings.
    static double jsonDouble(JSONValue obj, string key)
    {
        if (key !in obj)
            return 0;

        JSONValue value = obj[key];
        switch (value.type)
        {
        case JSONType.float_:
            return value.floating;
        case JSONType.integer:
            return cast(double)value.integer;
        case JSONType.uinteger:
            return cast(double)value.uinteger;
        case JSONType.string:
            try
                return value.str.to!double;
            catch (Exception)
                return 0;
        default:
            return 0;
        }
    }

    /// Reads an array of strings from a JSON object.
    static string[] jsonStrings(JSONValue obj, string key)
    {
        if (key !in obj || obj[key].type != JSONType.array)
            return null;

        string[] ret;
        foreach (entry; obj[key].array)
        {
            if (entry.type == JSONType.string)
                ret ~= entry.str;
        }
        return ret;
    }
}

private final class StreamThread : Thread
{
    this(void delegate() runDg)
    {
        super(runDg);
    }
}
