/// OpenRouter router with dynamic model discovery over the OpenAI-compatible API.
module intuit.router.openrouter;

import intuit.context;
import intuit.exception : EndpointException;
import intuit.model;
import intuit.router.details;
import intuit.provider.openai;
import intuit.router;
import intuit.tool;

import std.conv : to;
import std.json : JSONType, JSONValue;
import std.net.curl : HTTP;
import std.string : join;

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
     * The provided URL is used as-is; the caller must supply the correct
     * base URL for the endpoint.
     *
     * Params:
     *  key = The OpenRouter API key.
     *  url = The base URL, defaulting to the public OpenRouter host.
     *  name = Display name for the router.
     */
    this(string key, string url = "https://openrouter.ai", string name = "OpenRouter")
    {
        this._name = name;
        this._url = url;
        this._key = key;
        this._http = HTTP();
        this._context.compactor = new Compactor();
        this._http.addRequestHeader("Content-Type", "application/json");
        if (this._key.length > 0)
            this._http.addRequestHeader("Authorization", "Bearer "~this._key);
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
        => request(_http, HTTP.Method.post, _url~"/api/v1/chat/completions", decorate(payload));

    override JSONValue _embeddings(JSONValue payload)
    {
        if (_hasProvider)
            payload["provider"] = _provider;
        return request(_http, HTTP.Method.post, _url~"/api/v1/embeddings", payload);
    }

    /// Re-fetches the model catalog.
    override void refresh()
    {
        JSONValue json = request(_http, HTTP.Method.get, _url~"/api/v1/models");
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
        if (val.length > 0)
            _http.addRequestHeader("HTTP-Referer", val);
        return this;
    }

    /// Gets or sets the X-Title used to identify the app to OpenRouter.
    ref string title()
        => _title;

    /// ditto
    OpenRouter title(string val)
    {
        _title = val;
        if (val.length > 0)
            _http.addRequestHeader("X-Title", val);
        return this;
    }

    /// Gets or sets the marketplace categories sent via X-OpenRouter-Categories.
    ref string[] categories()
        => _categories;

    /// ditto
    OpenRouter categories(string[] val)
    {
        _categories = val;
        if (val.length > 0)
            _http.addRequestHeader("X-OpenRouter-Categories", val.join(","));
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
