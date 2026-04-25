module intuit.openai.endpoint;

import intuit.error : EndpointError;
import intuit.endpoint;
import intuit.model;
import intuit.openai.model;
import conductor.http : JSONValue, Response, parseJSON, send;
import std.net.curl : HTTP;
import std.json : JSONType;
import std.string : assumeUTF;

class OpenAI : IEndpoint
{
    private string _name;
    private string _url;
    private string _key;
    HTTP http;

    this(string url, string key = null, string name = "OpenAI")
    {
        this._name = name;
        this._url = normalizeBaseUrl(url);
        this._key = key;
        this.http = HTTP();
    }

    override string name() => _name;
    override void name(string value) { _name = value; }
    override string url() => _url;
    override void url(string value) { _url = value; }
    override void key(string value)
    {
        _key = value;
    }

    override IModel[] available()
    {
        JSONValue json = request(HTTP.Method.get, "models");
        IModel[] models;
        if ("data" in json && json["data"].type == JSONType.array)
        {
            foreach (m; json["data"].array)
            {
                string name = "id" in m ? m["id"].str : null;
                string owner = "owned_by" in m ? m["owned_by"].str : null;
                models ~= new OpenAIModel(name, owner);
            }
        }
        return models;
    }

    override IModel model(string name)
    {
        return new OpenAIModel(name);
    }

    override JSONValue _completions(IModel model, JSONValue payload)
    {
        return request(HTTP.Method.post, "chat/completions", payload);
    }

    override JSONValue _embeddings(IModel model, JSONValue payload)
    {
        return request(HTTP.Method.post, "embeddings", payload);
    }

private:
    JSONValue request(HTTP.Method method, string tail, JSONValue payload = JSONValue.init)
    {
        string target = route(tail);
        Response response;

        if (payload.type == JSONType.null_)
        {
            response = send(http, method, target, null, null, requestHeaders());
        }
        else
        {
            response = send(
                http,
                method,
                target,
                cast(const(ubyte)[])payload.toString(),
                "application/json",
                requestHeaders(),
            );
        }

        string body = response.content is null ? null : response.content.assumeUTF().idup;
        if (response.status < 200 || response.status >= 300)
            throw new EndpointError(methodName(method), target, response.status, response.reason, body);

        try
            return body.parseJSON();
        catch (Exception)
            throw new EndpointError(
                methodName(method),
                target,
                response.status,
                response.reason,
                body,
                "Endpoint returned invalid JSON.",
            );
    }

    string[string] requestHeaders()
    {
        string[string] headers;
        if (_key !is null && _key.length > 0)
            headers["Authorization"] = "Bearer "~_key;
        return headers;
    }

    string route(string tail)
    {
        while (tail.length > 0 && tail[0] == '/')
            tail = tail[1..$];
        return _url~"/v1/"~tail;
    }

    static string normalizeBaseUrl(string url)
    {
        string ret = url;
        while (ret.length > 0 && ret[$-1] == '/')
            ret = ret[0..$-1];
        if (ret.length >= 3 && ret[$-3..$] == "/v1")
            ret = ret[0..$-3];
        return ret;
    }

    static string methodName(HTTP.Method method)
    {
        switch (method)
        {
        case HTTP.Method.undefined:
            return "UNDEFINED";
        case HTTP.Method.get:
            return "GET";
        case HTTP.Method.post:
            return "POST";
        case HTTP.Method.put:
            return "PUT";
        case HTTP.Method.patch:
            return "PATCH";
        case HTTP.Method.del:
            return "DELETE";
        case HTTP.Method.head:
            return "HEAD";
        case HTTP.Method.options:
            return "OPTIONS";
        case HTTP.Method.trace:
            return "TRACE";
        case HTTP.Method.connect:
            return "CONNECT";
        default:
            return "UNKNOWN";
        }
    }
}
