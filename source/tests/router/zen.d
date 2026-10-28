module tests.router.zen;

import intuit;
import intuit.exception : EndpointException, FormatException;
import intuit.provider.claude.model : ClaudeModelConfig;
import unit_threaded : Name;

import std.algorithm.searching : canFind;
import std.conv : to;
import std.json : JSONValue, parseJSON;
import std.socket : InternetAddress, Socket, SocketOption, SocketOptionLevel, TcpSocket;
import std.string : indexOf, split, strip;
import std.uni : icmp;
import core.thread : Thread;
import core.time : dur;

@Name("Zen is publicly constructible and caches model configs without network access")
unittest
{
    string[] requests;
    withServer([], delegate void(string url) {
        Zen router = new Zen("key", url);

        assert(router.name == "OpenCode Zen");
        assert(router.active is null);
        assert(router.context.compactor !is null);

        bool unset;
        try
            router.config();
        catch (Exception)
            unset = true;
        assert(unset);

        ModelConfig claude = router.config("claude-sonnet-4");
        assert(cast(ClaudeModelConfig)claude !is null);
        assert(router.config("qwen2.5") is router.config("qwen2.5"));
        assert(cast(ClaudeModelConfig)router.config("qwen2.5") !is null);
        assert(cast(ClaudeModelConfig)router.config("qwen3-coder") is null);
        assert(router.config("gpt-5").name == "gpt-5");
        assert(router.active is null);
        assert(router.config("claude-sonnet-4") is claude);
        assert(requests.length == 0);
    }, &requests);
}

@Name("Zen catalog validates minimal model data and caches empty responses")
unittest
{
    string[] requests;
    withServer([
        `{"data":[
            {"id":"kimi-k3"},
            {"id":"custom","name":"Custom","description":"Compatible","context_length":8192},
            {"id":"bad-fields","name":5,"description":false,"context_length":-1},
            {"id":"gpt-5"},
            {"id":"gemini-2.5-pro"},
            {"id":"jev-1"},
            {"id":7},
            {"id":""},
            null
        ]}`,
    ], delegate void(string url) {
        Zen router = new Zen("catalog-key", url);
        ModelDetails[string] catalog = router.catalog;

        assert(catalog.length == 6);
        assert(catalog["kimi-k3"].name == "kimi-k3");
        assert(catalog["kimi-k3"].contextLength == 0);
        assert(catalog["kimi-k3"].promptCost == 0);
        assert(catalog["custom"].name == "Custom");
        assert(catalog["custom"].description == "Compatible");
        assert(catalog["custom"].contextLength == 8192);
        assert(catalog["bad-fields"].name == "bad-fields");
        assert(catalog["bad-fields"].description.length == 0);
        assert(catalog["bad-fields"].contextLength == 0);
        assert("gpt-5" in catalog);
        assert("gemini-2.5-pro" in catalog);
        assert("jev-1" in catalog);

        router.catalog;
        assert(requests.length == 1);
        assert(requests[0].split("\r\n")[0] == "GET /zen/v1/models HTTP/1.1");
        assert(hasHeader(requests[0], "Authorization", "Bearer catalog-key"));
        assert(hasHeader(requests[0], "Content-Type", "application/json"));
    }, &requests);

    requests = null;
    withServer([`{"data":[]}`], delegate void(string url) {
        Zen router = new Zen("", url);
        assert(router.catalog.length == 0);
        assert(router.catalog.length == 0);
        assert(requests.length == 1);
    }, &requests);
}

@Name("Zen activation accepts aliases and updates compaction only for positive context sizes")
unittest
{
    string[] requests;
    withServer([
        `{"data":[
            {"id":"small","context_length":0},
            {"id":"large","context_length":4096}
        ]}`,
    ], delegate void(string url) {
        Zen router = new Zen("", url);
        router.context.system("keep this message");
        router.context.compactor.maxTokens = 3456;

        router.active("missing-alias");
        assert(router.active == "missing-alias");
        assert(router.context.compactor.maxTokens == 3456);
        assert(router.context.length == 1);

        router.active("small");
        assert(router.context.compactor.maxTokens == 3456);
        router.active("large");
        assert(router.context.compactor.maxTokens == 4096);
        assert(router.context.length == 1);
        assert(requests.length == 1);
    }, &requests);
}

@Name("Zen refresh replaces catalog while preserving active config and context")
unittest
{
    string[] requests;
    withServer([
        `{"data":[{"id":"old"}]}`,
        `{"data":[{"id":"new"}]}`,
    ], delegate void(string url) {
        Zen router = new Zen("", url);
        assert("old" in router.catalog);
        ModelConfig config = router.config("alias");
        router.context.system("preserved");
        router.active("alias");

        router.refresh();

        assert("old" !in router.catalog);
        assert("new" in router.catalog);
        assert(router.active == "alias");
        assert(router.config() is config);
        assert(router.context.length == 1);
        assert(requests.length == 2);
    }, &requests);
}

@Name("Zen Chat Completions route parses responses and maintains context")
unittest
{
    string[] requests;
    withServer([
        `{"data":[{"id":"kimi-k3"}]}`,
        `{"model":"kimi-k3","choices":[{"message":{"role":"assistant","content":"Answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}`,
    ], delegate void(string url) {
        Zen router = new Zen("chat-key", url);
        router.active("kimi-k3");
        router.context.system("Be concise.");
        router.config().temperature = 0.25;

        Completion result = completions(router, "Question");

        assert(result.text == "Answer");
        assert(result.usage.promptTokens == 4);
        assert(result.usage.completionTokens == 2);
        assert(router.context.length == 3);
        assert(requests.length == 2);
        assert(requests[1].split("\r\n")[0] == "POST /zen/v1/chat/completions HTTP/1.1");
        assert(hasHeader(requests[1], "Authorization", "Bearer chat-key"));
        assert(hasHeader(requests[1], "Content-Type", "application/json"));
        JSONValue payload = requestBody(requests[1]);
        assert(payload["model"].str == "kimi-k3");
        assert(payload["temperature"].floating == 0.25);
        assert(payload["messages"].array.length == 2);
        assert(payload["messages"][0]["content"].str == "Be concise.");
        assert(payload["messages"][1]["content"].str == "Question");
    }, &requests);
}

@Name("Zen uses Anthropic Messages for Claude and Qwen but Chat for Qwen3 Coder")
unittest
{
    string[] requests;
    withServer([
        `{"data":[]}`,
        `{"model":"claude-sonnet-4","content":[{"type":"text","text":"Claude answer"}],"stop_reason":"end_turn","usage":{"input_tokens":7,"output_tokens":3}}`,
        `{"model":"qwen3-coder","choices":[{"message":{"role":"assistant","content":"Qwen answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}}`,
    ], delegate void(string url) {
        Zen router = new Zen("family-key", url);
        router.active("claude-sonnet-4");
        ClaudeModelConfig claude = cast(ClaudeModelConfig)router.config();
        assert(claude !is null);
        claude.maxTokens = 512;

        Completion claudeResult = completions(router, "Claude question");
        assert(claudeResult.text == "Claude answer");
        assert(claudeResult.usage.promptTokens == 7);
        assert(requests.length == 2);
        assert(requests[1].split("\r\n")[0] == "POST /zen/v1/messages HTTP/1.1");
        assert(hasHeader(requests[1], "Authorization", "Bearer family-key"));
        assert(hasHeader(requests[1], "Content-Type", "application/json"));
        assert(hasHeader(requests[1], "anthropic-version", "2023-06-01"));
        JSONValue claudePayload = requestBody(requests[1]);
        assert(claudePayload["model"].str == "claude-sonnet-4");
        assert(claudePayload["max_tokens"].integer == 512);

        router.active("qwen3-coder");
        assert(cast(ClaudeModelConfig)router.config() is null);
        Completion qwenResult = completions(router, "Qwen question");
        assert(qwenResult.text == "Qwen answer");
        assert(requests.length == 3);
        assert(requests[2].split("\r\n")[0] == "POST /zen/v1/chat/completions HTTP/1.1");
        assert(requestBody(requests[2])["model"].str == "qwen3-coder");
    }, &requests);
}

@Name("Zen rejects unsupported API families and APIs before network access")
unittest
{
    Zen router = new Zen("", "http://127.0.0.1:1");
    foreach (model; ["gpt-5", "gemini-2.5-pro", "jev-1"])
    {
        JSONValue payload = JSONValue.emptyObject;
        payload["model"] = JSONValue(model);

        bool rejected;
        try
            router._completions(payload);
        catch (EndpointException error)
        {
            rejected = error.msg.canFind("does not support the API family");
        }
        assert(rejected);
    }

    bool embeddingsRejected;
    try
        router._embeddings(JSONValue.emptyObject);
    catch (EndpointException error)
        embeddingsRejected = error.msg.canFind("does not support embeddings");
    assert(embeddingsRejected);

    bool decisionsRejected;
    try
        router._decisions(JSONValue.emptyObject);
    catch (EndpointException error)
        decisionsRejected = error.msg.canFind("does not support decision requests");
    assert(decisionsRejected);
}

@Name("Zen validates raw completion model before network access")
unittest
{
    Zen router = new Zen("", "http://127.0.0.1:1");
    JSONValue[] invalid = [JSONValue.emptyObject, JSONValue("missing"), parseJSON(`{"model":5}`)];

    foreach (payload; invalid)
    {
        bool rejected;
        try
            router._completions(payload);
        catch (FormatException error)
            rejected = error.msg.canFind("non-empty string model");
        assert(rejected);
    }
}

private:

void withServer(string[] responses, scope void delegate(string) action, string[]* requests)
{
    TcpSocket listener = new TcpSocket();
    listener.bind(new InternetAddress("127.0.0.1", 0));
    listener.listen(cast(int)(responses.length > 0 ? responses.length : 1));
    listener.setOption(SocketOptionLevel.SOCKET, SocketOption.RCVTIMEO, dur!"seconds"(3));

    Throwable serverFailure;
    Thread server = new Thread(delegate void() {
        try
        {
            foreach (response; responses)
            {
                Socket connection = listener.accept();
                scope(exit)
                    connection.close();
                connection.setOption(SocketOptionLevel.SOCKET, SocketOption.RCVTIMEO, dur!"seconds"(3));

                string captured = readRequest(connection);
                *requests ~= captured;

                string reply = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                    ~response.length.to!string~"\r\nConnection: close\r\n\r\n"~response;
                size_t bytesSent;
                while (bytesSent < reply.length)
                {
                    ptrdiff_t count = connection.send(reply[bytesSent..$]);
                    if (count <= 0)
                        break;
                    bytesSent += cast(size_t)count;
                }
                assert(bytesSent == reply.length);
            }
        }
        catch (Throwable error)
            serverFailure = error;
    });
    server.start();

    string url = "http://127.0.0.1:"~(cast(InternetAddress)listener.localAddress).port.to!string;
    try
        action(url);
    catch (Throwable error)
    {
        listener.close();
        server.join();
        throw error;
    }

    listener.close();
    server.join();
    if (serverFailure !is null)
        throw serverFailure;
}

string readRequest(Socket connection)
{
    ubyte[8192] buffer;
    size_t bytesReceived;
    size_t requestLength = size_t.max;

    while (bytesReceived < buffer.length)
    {
        ptrdiff_t count = connection.receive(buffer[bytesReceived..$]);
        if (count <= 0)
            break;
        bytesReceived += cast(size_t)count;

        if (requestLength == size_t.max)
        {
            size_t headersEnd = size_t.max;
            for (size_t position; position + 3 < bytesReceived; position++)
            {
                if (buffer[position] == '\r' && buffer[position + 1] == '\n'
                    && buffer[position + 2] == '\r' && buffer[position + 3] == '\n')
                {
                    headersEnd = position;
                    break;
                }
            }

            if (headersEnd != size_t.max)
            {
                size_t bodyLength;
                string headers = cast(string)buffer[0..headersEnd];
                foreach (line; headers.split("\r\n"))
                {
                    ptrdiff_t separator = line.indexOf(':');
                    if (separator >= 0
                        && line[0..cast(size_t)separator].icmp("Content-Length") == 0)
                    {
                        bodyLength = line[cast(size_t)separator + 1..$].strip.to!size_t;
                        break;
                    }
                }
                requestLength = headersEnd + 4 + bodyLength;
            }
        }

        if (requestLength != size_t.max && bytesReceived >= requestLength)
            break;
    }

    assert(requestLength != size_t.max && bytesReceived >= requestLength);
    return cast(string)buffer[0..requestLength].idup;
}

JSONValue requestBody(string request)
{
    ptrdiff_t separator = request.indexOf("\r\n\r\n");
    assert(separator >= 0);
    return parseJSON(request[cast(size_t)separator + 4..$]);
}

bool hasHeader(string request, string name, string value)
{
    foreach (line; request.split("\r\n"))
    {
        ptrdiff_t separator = line.indexOf(':');
        if (separator >= 0
            && line[0..cast(size_t)separator].icmp(name) == 0
            && line[cast(size_t)separator + 1..$].strip == value)
            return true;
    }
    return false;
}
