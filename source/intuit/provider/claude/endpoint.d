/// Anthropic Claude endpoint implementation.
module intuit.provider.claude.endpoint;

public import intuit.provider;
import intuit.provider.claude.model;
import intuit.model;
import intuit.exception : EndpointException;
import intuit.response;
import intuit.tool;
import conductor.http : Response, send;

import std.net.curl : HTTP;
import std.json : JSONType, JSONValue, parseJSON;
import std.string : assumeUTF;
import std.conv : to;

/// Anthropic Claude LLM endpoint.
class Claude : IEndpoint
{
private:
    string _name;
    string _url;
    string _key;
    ToolRegistry _tools;
    HTTP _http;

protected:
    ModelConfig[] _configs;

public:
    /**
     * Constructs a Claude endpoint.
     *
     * Params:
     *  url = The base URL of the endpoint.
     *  key = Optional API key.
     *  name = Display name for the endpoint.
     */
    this(string url, string key = null, string name = "Claude")
    {
        _name = name;
        _url = normalizeBaseUrl(url);
        _key = key;
        _http = HTTP();
    }

    override ref string name()
        => _name;

    override ref string url()
        => _url;

    override ref string key()
        => _key;

    override ref ToolRegistry tools()
        => _tools;

    override ModelConfig[] available()
    {
        JSONValue json = request(HTTP.Method.get, "models");
        if ("data" in json && json["data"].type == JSONType.array)
        {
            foreach (item; json["data"].array)
            {
                string name = "id" in item ? item["id"].str : null;
                if (name is null)
                    continue;

                bool found = false;
                foreach (cfg; _configs)
                {
                    if (cfg.name == name)
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                    _configs ~= new ClaudeModelConfig(name);
            }
        }
        return _configs;
    }

    override ModelConfig config(string modelName)
    {
        foreach (cfg; _configs)
        {
            if (cfg.name == modelName)
                return cfg;
        }
        
        ModelConfig cfg = new ClaudeModelConfig(modelName);
        _configs ~= cfg;
        return cfg;
    }

    override ModelConfig[] configs()
        => _configs;

    override JSONValue _completions(ModelConfig cfg, JSONValue payload)
        => request(HTTP.Method.post, "messages", payload);

    override JSONValue _embeddings(ModelConfig cfg, JSONValue payload)
    {
        throw new EndpointException("POST", "embeddings", 0, "not supported", "Claude does not support embeddings.");
    }

    /**
     * Sends an HTTP request to the endpoint.
     *
     * Params:
     *  method = The HTTP method.
     *  tail = The API path tail.
     *  payload = Optional JSON payload for POST requests.
     *
     * Returns:
     *  The parsed JSON response.
     *
     * Throws:
     *  EndpointException on HTTP or JSON parse failures.
     */
    JSONValue request(HTTP.Method method, string tail, JSONValue payload = JSONValue.init)
    {
        string target = route(tail);
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

    /// Builds request headers including auth and version.
    string[string] requestHeaders()
    {
        string[string] headers;
        headers["anthropic-version"] = "2023-06-01";
        if (_key !is null && _key.length > 0)
            headers["x-api-key"] = _key;
        return headers;
    }

    /// Constructs a full route from a path tail.
    string route(string tail)
    {
        while (tail.length > 0 && tail[0] == '/')
            tail = tail[1..$];
        return _url~"/v1/"~tail;
    }

    /// Normalizes a base URL by stripping trailing slashes and /v1 suffixes.
    static string normalizeBaseUrl(string url)
    {
        string ret = url;
        while (ret.length > 0 && ret[$-1] == '/')
            ret = ret[0..$-1];
        if (ret.length >= 3 && ret[$-3..$] == "/v1")
            ret = ret[0..$-3];
        return ret;
    }
}
