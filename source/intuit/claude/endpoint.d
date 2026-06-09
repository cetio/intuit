/// Anthropic Claude endpoint implementation.
module intuit.claude.endpoint;

public import intuit.endpoint;
import intuit.error : EndpointError;
import intuit.model;
import intuit.claude.model;
import intuit.response;
import intuit.stream.sse : SseParser;
import intuit.tool;
import conductor.http : Response, send;
import conductor.serialize.json : toJSON;
import core.thread : Thread;
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
    IModel[string] _models;

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
                    string owner = "anthropic";
                    _models[name] = new ClaudeModel(name, owner);
                }
            }
        }
        return _models.values;
    }

    override IModel model(string name)
        => name in _models ? _models[name] : new ClaudeModel(name, null);

    override JSONValue _completions(IModel model, JSONValue payload)
        => request(HTTP.Method.post, "messages", payload);

    override JSONValue _embeddings(IModel model, JSONValue payload)
    {
        throw new EndpointError("POST", "embeddings", 0, "not supported", "Claude does not support embeddings.");
    }

    override CompletionStream _stream(IModel model, JSONValue payload)
    {
        string target = route("messages");

        string[string] headers = requestHeaders();
        headers["Accept"] = "text/event-stream";

        CompletionStream stream = new CompletionStream(model.name, null);

        HTTP http = HTTP();
        http.clearRequestHeaders();
        http.url = target;
        http.method = HTTP.Method.post;
        foreach (k, v; headers)
            http.addRequestHeader(k, v);
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

        string toolInputBuffer;
        string currentToolId;
        string currentToolName;

        SseParser parser = new SseParser();
        http.onReceive = (ubyte[] chunk) {
            try
            {
                auto events = parser.feed(chunk);
                foreach (event; events)
                {
                    switch (event.event)
                    {
                    case "ping":
                        continue;
                    case "message_stop":
                        stream.complete = true;
                        continue;
                    default:
                        break;
                    }

                    if (event.data.length == 0)
                        continue;

                    try
                    {
                        JSONValue json = event.data.parseJSON();

                        if (event.event == "content_block_delta" && "delta" in json)
                        {
                            JSONValue delta = json["delta"];
                            if ("type" in delta && delta["type"].str == "text_delta"
                                && "text" in delta)
                            {
                                Completion completion;
                                completion.choices = [Choice.init];
                                completion.choices[0].text = delta["text"].str;
                                stream.update(completion);
                                if (stream.callback !is null)
                                    stream.callback(completion);
                            }
                            else if ("type" in delta && delta["type"].str == "input_json_delta"
                                && "partial_json" in delta)
                                toolInputBuffer ~= delta["partial_json"].str;
                        }
                        else if (event.event == "message_delta" && "delta" in json)
                        {
                            JSONValue delta = json["delta"];
                            Completion completion;
                            completion.choices = [Choice.init];
                            if ("stop_reason" in delta)
                                completion.choices[0].finishReason = parseClaudeStopReason(delta["stop_reason"]);
                            stream.update(completion);
                            if (stream.callback !is null)
                                stream.callback(completion);
                        }
                        else if (event.event == "content_block_start" && "content_block" in json)
                        {
                            JSONValue block = json["content_block"];
                            if ("type" in block && block["type"].str == "tool_use")
                            {
                                currentToolId = ("id" in block) ? block["id"].str : "";
                                currentToolName = ("name" in block) ? block["name"].str : "";
                                toolInputBuffer = null;
                            }
                        }
                        else if (event.event == "content_block_stop")
                        {
                            if (currentToolId.length > 0)
                            {
                                Completion completion;
                                completion.choices = [Choice.init];
                                ToolCall tc;
                                tc.id = currentToolId;
                                tc.name = currentToolName;
                                tc.arguments = toolInputBuffer.toJSON();

                                completion.choices[0].toolCalls ~= tc;
                                stream.update(completion);
                                if (stream.callback !is null)
                                    stream.callback(completion);
                                currentToolId = null;
                                currentToolName = null;
                                toolInputBuffer = null;
                            }
                        }
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
                    if (event.event == "message_stop")
                    {
                        stream.complete = true;
                        continue;
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

private final class StreamThread : Thread
{
    this(void delegate() runDg)
    {
        super(runDg);
    }
}

/// Maps a Claude stop_reason string to the FinishReason enum.
private FinishReason parseClaudeStopReason(JSONValue value)
{
    if (value.type != JSONType.string)
        return FinishReason.Unknown;

    switch (value.str)
    {
    case "end_turn":
        return FinishReason.EndTurn;
    case "max_tokens":
        return FinishReason.Max_Tokens;
    case "stop_sequence":
        return FinishReason.StopSequence;
    case "tool_use":
        return FinishReason.ToolUse;
    case "content_filter":
        return FinishReason.ContentFilter;
    default:
        return FinishReason.Unknown;
    }
}
