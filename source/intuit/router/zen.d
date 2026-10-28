/// OpenCode Zen router for legacy Chat Completions and Anthropic Messages APIs.
module intuit.router.zen;

import intuit.context;
import intuit.exception : EndpointException, FormatException;
import intuit.model;
import intuit.provider : request;
import intuit.provider.claude.model : ClaudeModelConfig;
import intuit.router.details;
import intuit.router;
import intuit.tool;

import std.json : JSONType, JSONValue;
import std.net.curl : HTTP;
import std.string : startsWith;
import core.time : Duration;

/// OpenCode Zen router implementation.
class Zen : IRouter
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
    bool _catalogLoaded;

public:
    /**
     * Constructs an OpenCode Zen router.
     *
     * The provided URL is used as-is and the caller is responsible for
     * supplying the correct base URL for the endpoint.
     *
     * Params:
     *  key = The OpenCode Zen API key.
     *  url = The base URL, defaulting to the public OpenCode host.
     *  name = Display name for the router.
     */
    this(string key, string url = "https://opencode.ai", string name = "OpenCode Zen")
    {
        this._name = name;
        this._url = url;
        this._key = key;
        this._http = HTTP();
        this._context.compactor = new Compactor();
    }

    override void operationTimeout(Duration timeout)
    {
        _http.operationTimeout = timeout;
    }

    override void connectTimeout(Duration timeout)
    {
        _http.connectTimeout = timeout;
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
        if (!_catalogLoaded)
            refresh();

        _active = modelName;
        config(modelName);
        if (auto details = modelName in _catalog)
        {
            if (details.contextLength > 0)
                _context.compactor.maxTokens = details.contextLength;
        }
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

        if (usesMessages(modelName))
            return _configs[modelName] = new ClaudeModelConfig(modelName);
        return _configs[modelName] = new ModelConfig(modelName);
    }

    override ModelConfig[] configs()
        => _configs.values;

    override ModelDetails[string] catalog()
    {
        if (!_catalogLoaded)
            refresh();
        return _catalog;
    }

    override JSONValue _completions(JSONValue payload)
    {
        if (payload.type != JSONType.object || "model" !in payload
            || payload["model"].type != JSONType.string || payload["model"].str.length == 0)
            throw new FormatException("OpenCode Zen completion payload requires a non-empty string model.");

        string modelName = payload["model"].str;
        if (isUnsupported(modelName))
            throw new EndpointException(
                "POST",
                _url~"/zen/v1/chat/completions",
                0,
                "not supported",
                "",
                "OpenCode Zen does not support the API family for model "~modelName~".",
            );

        if (usesMessages(modelName))
            return _http.request(
                HTTP.Method.post,
                _url~"/zen/v1/messages",
                buildHeaders(true),
                payload,
            );

        return _http.request(
            HTTP.Method.post,
            _url~"/zen/v1/chat/completions",
            buildHeaders(),
            payload,
        );
    }

    override JSONValue _decisions(JSONValue payload)
    {
        throw new EndpointException(
            "POST",
            _url~"/zen/v1/decisions",
            0,
            "not supported",
            "",
            "OpenCode Zen does not support decision requests.",
        );
    }

    override JSONValue _embeddings(JSONValue payload)
    {
        throw new EndpointException(
            "POST",
            _url~"/zen/v1/embeddings",
            0,
            "not supported",
            "",
            "OpenCode Zen does not support embeddings.",
        );
    }

    /// Re-fetches the model catalog.
    override void refresh()
    {
        JSONValue json = _http.request(
            HTTP.Method.get,
            _url~"/zen/v1/models",
            buildHeaders(),
        );
        ModelDetails[string] catalog;
        if (json.type == JSONType.object && "data" in json && json["data"].type == JSONType.array)
        {
            foreach (item; json["data"].array)
            {
                ModelDetails details = parseDetails(item);
                if (details.id.length > 0)
                    catalog[details.id] = details;
            }
        }
        _catalog = catalog;
        _catalogLoaded = true;
    }

private:
    string[string] buildHeaders(bool messages = false)
    {
        string[string] ret = ["Content-Type": "application/json"];
        if (_key.length > 0)
            ret["Authorization"] = "Bearer "~_key;
        if (messages)
            ret["anthropic-version"] = "2023-06-01";
        return ret;
    }

    static bool usesMessages(string modelName)
    {
        return modelName.startsWith("claude-")
            || (modelName.startsWith("qwen") && !modelName.startsWith("qwen3-coder"));
    }

    static bool isUnsupported(string modelName)
    {
        return modelName.startsWith("gpt-")
            || modelName.startsWith("gemini-")
            || modelName.startsWith("jev-");
    }

    static ModelDetails parseDetails(JSONValue item)
    {
        ModelDetails ret;
        if (item.type != JSONType.object || "id" !in item
            || item["id"].type != JSONType.string || item["id"].str.length == 0)
            return ret;

        ret.id = item["id"].str;
        ret.name = ret.id;
        if ("name" in item && item["name"].type == JSONType.string)
            ret.name = item["name"].str;
        if ("description" in item && item["description"].type == JSONType.string)
            ret.description = item["description"].str;
        if ("context_length" in item && item["context_length"].type == JSONType.integer
            && item["context_length"].integer > 0)
            ret.contextLength = cast(size_t)item["context_length"].integer;
        return ret;
    }
}
