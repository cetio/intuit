/// OpenRouter router with dynamic model discovery over the OpenAI-compatible API.
module intuit.router.openrouter;

import intuit.context;
import intuit.exception : EndpointException;
import intuit.model;
import intuit.router.details;
import intuit.provider.openai;
import intuit.response;
import intuit.router;
import intuit.response.stream.sse : SSEParser;
import intuit.tool;
import conductor.http : Response, send;

import core.thread : Thread;
import std.conv : to;
import std.json : JSONType, JSONValue, parseJSON;
import std.net.curl : HTTP;
import std.string : assumeUTF, join;

/// OpenRouter router implementation.
class OpenRouter : IRouter
{
private:
    string _name;
    string _url;
    string _key;
    ToolRegistry _tools;
    Context _context;
    string _active;
    ModelConfig[string] _configs;
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

    override void active(string modelName)
    {
        // The catalog is huge and dynamic, so only fetch it once on demand. Aliases
        // like "openrouter/auto" or "~openai/gpt-latest" may be absent; we still allow
        // them and simply leave the compactor limit untouched when metadata is missing.
        // NOTE: the Auto Router resolves the real model per request in the response
        // "model" field. We could pin the compactor limit to that resolved model once
        // the response types expose it.
        if (modelName !in _catalog && _catalog.length == 0)
            refresh();

        _active = modelName;
        config(modelName);
        if (auto details = modelName in _catalog)
            _context.compactor.maxTokens = details.contextLength;
    }

    override ModelConfig config()
    {
        if (_active is null)
            throw new Exception("Router has no active model set.");
        return _configs[_active];
    }

    override ModelConfig config(string modelName)
    {
        if (auto found = modelName in _configs)
            return *found;
        ModelConfig ret = new ModelConfig(modelName);
        _configs[modelName] = ret;
        return ret;
    }

    override ModelConfig[] configs()
    {
        ModelConfig[] ret;
        foreach (cfg; _configs.byValue)
            ret ~= cfg;
        return ret;
    }

    override ModelDetails[string] catalog()
    {
        if (_catalog.length == 0)
            refresh();
        return _catalog;
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
        ModelConfig streamCfg = config();
        string target = resolve("chat/completions");

        string[string] headers = requestHeaders();
        headers["Accept"] = "text/event-stream";

        CompletionStream stream = new CompletionStream(streamCfg.name, null);

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
                        Completion completion = streamCfg.parseResponse(json);
                        stream.update(completion);
                        if (stream.callback !is null)
                            stream.callback(completion);
                    }
                    catch (Exception ex)
                    {
                        stream.exception = ex;
                        stream.complete = true;
                    }
                }
            }
            catch (Exception ex)
            {
                stream.exception = ex;
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
                    stream.exception = new EndpointException(
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
                        Completion completion = streamCfg.parseResponse(json);
                        stream.update(completion);
                        if (stream.callback !is null)
                            stream.callback(completion);
                    }
                    catch (Exception ex)
                    {
                        stream.exception = ex;
                    }
                }
                stream.complete = true;
            }
            catch (Exception ex)
            {
                stream.exception = ex;
                stream.complete = true;
            }
        }

        stream.commence((CompletionStream) { new Thread(&doStream).start(); });
        return stream;
    }

    /// Re-fetches the model catalog.
    override void refresh()
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
            throw new EndpointException(method.to!string, target, response.status, response.reason, content);

        try
            return content.parseJSON();
        catch (Exception)
            throw new EndpointException(
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

        if ("context_length" in item && item["context_length"].type == JSONType.integer)
            ret.contextLength = cast(size_t)item["context_length"].integer;

        if ("top_provider" in item && item["top_provider"].type == JSONType.object)
        {
            JSONValue provider = item["top_provider"];
            if (ret.contextLength == 0 && "context_length" in provider
                && provider["context_length"].type == JSONType.integer)
                ret.contextLength = cast(size_t)provider["context_length"].integer;
            if ("max_completion_tokens" in provider
                && provider["max_completion_tokens"].type == JSONType.integer)
                ret.maxCompletionTokens = cast(size_t)provider["max_completion_tokens"].integer;
        }

        if ("architecture" in item && item["architecture"].type == JSONType.object)
        {
            JSONValue architecture = item["architecture"];
            if ("input_modalities" in architecture
                && architecture["input_modalities"].type == JSONType.array)
            {
                foreach (entry; architecture["input_modalities"].array)
                {
                    if (entry.type == JSONType.string)
                        ret.inputModalities ~= entry.str;
                }
            }

            if ("output_modalities" in architecture
                && architecture["output_modalities"].type == JSONType.array)
            {
                foreach (entry; architecture["output_modalities"].array)
                {
                    if (entry.type == JSONType.string)
                        ret.outputModalities ~= entry.str;
                }
            }
        }

        if ("supported_parameters" in item && item["supported_parameters"].type == JSONType.array)
        {
            foreach (entry; item["supported_parameters"].array)
            {
                if (entry.type == JSONType.string)
                    ret.supportedParameters ~= entry.str;
            }
        }

        if ("pricing" in item && item["pricing"].type == JSONType.object)
        {
            JSONValue pricing = item["pricing"];
            if ("prompt" in pricing && pricing["prompt"].type == JSONType.float_)
                ret.promptCost = pricing["prompt"].floating;
            if ("completion" in pricing && pricing["completion"].type == JSONType.float_)
                ret.completionCost = pricing["completion"].floating;
        }

        return ret;
    }
}
