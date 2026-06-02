/// OpenAI-compatible endpoint implementation.
module intuit.openai.endpoint;

public import intuit.endpoint;
import intuit.error : EndpointError;
import intuit.model;
import intuit.openai.model;
import intuit.tool;
import conductor.http : Response, send;
import std.net.curl : HTTP;
import std.json : JSONType, JSONValue, parseJSON;
import std.string : assumeUTF;
import std.conv : to;

/// OpenAI-compatible LLM endpoint.
class OpenAI : IEndpoint
{
private:
    string _name;
    string _url;
    string _key;
    ToolRegistry _tools;
    HTTP _http;

    IModel[string] _models;

public:
    /**
     * Constructs an OpenAI endpoint.
     *
     * Params:
     *  url = The base URL of the endpoint.
     *  key = Optional API key.
     *  name = Display name for the endpoint.
     */
    this(string url, string key = null, string name = "OpenAI")
    {
        this._name = name;
        this._url = normalizeBaseUrl(url);
        this._key = key;
        this._http = HTTP();
    }

    override ref string name()
        => _name;

    override ref string url()
        => _url;

    override ref string key()
        => _key;

    override ref ToolRegistry tools()
        => _tools;

    override IModel[] available()
    {
        JSONValue json = request(HTTP.Method.get, "models");
        if ("data" in json && json["data"].type == JSONType.array)
        {
            foreach (item; json["data"].array)
            {
                string name = "id" in item ? item["id"].str : null;
                if (name !in _models)
                {
                    string owner = "owned_by" in item ? item["owned_by"].str : null;
                    _models[name] = cast(IModel)(new OpenAIModel(name, owner));
                }
            }
        }
        return _models.values;
    }

    override IModel model(string name)
        => name in _models ? _models[name] : new OpenAIModel(name, null);

    override JSONValue _completions(IModel model, JSONValue payload)
        => request(HTTP.Method.post, "chat/completions", payload);

    override JSONValue _embeddings(IModel model, JSONValue payload)
        => request(HTTP.Method.post, "embeddings", payload);

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
     *  EndpointError on HTTP or JSON parse failures.
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

    /// Builds request headers including authorization if a key is set.
    string[string] requestHeaders()
    {
        string[string] headers;
        if (_key !is null && _key.length > 0)
            headers["Authorization"] = "Bearer "~_key;
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
