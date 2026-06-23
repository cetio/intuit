/// OpenRouter router with dynamic model discovery over the OpenAI-compatible API.
module intuit.router.openrouter;

import intuit.context;
import intuit.exception : EndpointException;
import intuit.model;
import intuit.provider.openai;
import intuit.router;
import intuit.tool;

import std.conv : to;
import std.json : JSONType, JSONValue;
import std.net.curl : HTTP;
import std.string : join;

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
 * ModelConfig since OpenRouter normalizes every provider to the OpenAI schema.
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
    ModelConfig _activeConfig;
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
        _activeConfig = new ModelConfig(modelName);
        if (auto details = modelName in _catalog)
            _context.compactor.maxTokens = details.contextLength;
    }

    override ModelConfig config()
    {
        if (_active is null)
            throw new Exception("Router has no active model set.");
        return _activeConfig;
    }

    override JSONValue _completions(JSONValue payload)
        => request(_http, HTTP.Method.post, _url~"/api/v1/chat/completions", decorate(payload));

    override JSONValue _embeddings(JSONValue payload)
    {
        if (_hasProvider)
            payload["provider"] = _provider;
        return request(_http, HTTP.Method.post, _url~"/api/v1/embeddings", payload);
    }

    /// Re-fetches the model catalog from the `/models` endpoint.
    void refresh()
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
    ModelDetails details(string modelName)
    {
        if (modelName !in _catalog && _catalog.length == 0)
            refresh();
        if (auto found = modelName in _catalog)
            return *found;
        throw new Exception("Model not found in OpenRouter catalog: "~modelName);
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
