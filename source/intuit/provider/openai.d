/// OpenAI-compatible endpoint implementation.
module intuit.provider.openai;

public import intuit.provider;
import intuit.exception : EndpointException;
import intuit.model;
import intuit.response;
import intuit.tool;
import intuit.response.stream.sse : SSEParser, SSEEvent;
import conductor.http : Response, send;

import core.thread : Thread;
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

protected:
    ModelConfig[] _configs;

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
                    _configs ~= new ModelConfig(name);
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
        
        ModelConfig cfg = new ModelConfig(modelName);
        _configs ~= cfg;
        return cfg;
    }

    override ModelConfig[] configs()
        => _configs;

    override JSONValue _completions(ModelConfig cfg, JSONValue payload)
        => request(HTTP.Method.post, "chat/completions", payload);

    override JSONValue _embeddings(ModelConfig cfg, JSONValue payload)
        => request(HTTP.Method.post, "embeddings", payload);

    override CompletionStream _stream(ModelConfig cfg, JSONValue payload)
    {
        string target = route("chat/completions");

        string[string] headers = requestHeaders();
        headers["Accept"] = "text/event-stream";

        CompletionStream stream = new CompletionStream(cfg.name, null);

        HTTP http = HTTP();
        http.clearRequestHeaders();
        http.url = target;
        http.method = HTTP.Method.post;
        foreach (key, value; headers)
            http.addRequestHeader(key, value);
        http.addRequestHeader("Content-Type", "application/json");

        string bodyStr = payload.toString();
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
                SSEEvent[] events = parser.feed(chunk);
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
                        Completion completion = cfg.parseResponse(json);
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

                SSEEvent[] finalEvents = parser.flush();
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
                        Completion completion = cfg.parseResponse(json);
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
        {
            throw new EndpointException(
                method.to!string,
                target,
                response.status,
                response.reason,
                content,
                "Endpoint returned invalid JSON.",
            );
        }
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
