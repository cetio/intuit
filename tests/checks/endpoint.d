module checks.endpoint;

import conductor.http : toJSON;
import core.thread : Thread;
import intuit;
import intuit.openai : OpenAI;
import std.algorithm.searching : canFind, countUntil;
import std.conv : to;
import std.json : JSONType, JSONValue, parseJSON;
import std.socket : AddressFamily, InternetAddress, Socket, SocketType, TcpSocket;
import std.string : split, strip, toLower;

private struct RequestData
{
    string method;
    string path;
    string body;
    string[string] headers;
}

private class StubModel : IModel
{
    string _name;

    this(string name)
    {
        _name = name;
    }

    override string name()
    {
        return _name;
    }

    override JSONValue completionsJSON(JSONValue input)
    {
        JSONValue ret = JSONValue.emptyObject;
        ret["model"] = JSONValue(_name);
        ret["input"] = input;
        return ret;
    }

    override JSONValue embeddingsJSON(JSONValue input)
    {
        JSONValue ret = JSONValue.emptyObject;
        ret["model"] = JSONValue(_name);
        ret["input"] = input;
        return ret;
    }

    override Completion parseCompletions(JSONValue response)
    {
        Completion ret;
        Choice choice;
        choice.text = response["text"].str;
        ret.choices = [choice];
        return ret;
    }

    override JSONValue parseEmbeddings(JSONValue response)
    {
        return response["data"];
    }
}

private class StubEndpoint : IEndpoint
{
    string _name = "stub";
    string _url = "stub://";
    string _key;
    string lastModel;

    override string name() => _name;
    override void name(string value) { _name = value; }
    override string url() => _url;
    override void url(string value) { _url = value; }
    override void key(string value) { _key = value; }

    override IModel[] available()
    {
        return null;
    }

    override IModel model(string name)
    {
        lastModel = name;
        return new StubModel(name);
    }

    override JSONValue _completions(IModel model, JSONValue payload)
    {
        JSONValue ret = JSONValue.emptyObject;
        ret["text"] = JSONValue(payload["model"].str);
        return ret;
    }

    override JSONValue _embeddings(IModel model, JSONValue payload)
    {
        JSONValue ret = JSONValue.emptyObject;
        ret["data"] = JSONValue.emptyArray;
        ret["data"].array ~= JSONValue.emptyArray;
        ret["data"][0].array ~= JSONValue(1.0f);
        ret["data"][0].array ~= JSONValue(2.0f);
        return ret;
    }
}

private RequestData withServer(
    string responseBody,
    void delegate(string) run,
    ushort status = 200,
    string reason = "OK",
)
{
    auto listener = new TcpSocket(AddressFamily.INET);
    listener.blocking = true;
    listener.bind(new InternetAddress("127.0.0.1", 0));
    listener.listen(1);

    auto bound = cast(InternetAddress)listener.localAddress();
    string baseUrl = "http://127.0.0.1:"~bound.port.to!string;

    RequestData request;
    Thread server = new Thread({
        auto client = listener.accept();
        scope(exit)
        {
            client.close();
            listener.close();
        }

        string data;
        ubyte[1024] buffer;
        while (!data.canFind("\r\n\r\n"))
        {
            ptrdiff_t got = client.receive(buffer[]);
            if (got <= 0)
                break;
            data ~= cast(string)buffer[0..got];
        }

        ptrdiff_t headerEnd = data.countUntil("\r\n\r\n");
        string headerText = headerEnd < 0 ? data : data[0..headerEnd];
        string body = headerEnd < 0 ? null : data[headerEnd + 4..$];
        string[] lines = headerText.split("\r\n");
        if (lines.length > 0)
        {
            string[] head = lines[0].split(" ");
            if (head.length >= 2)
            {
                request.method = head[0];
                request.path = head[1];
            }
        }

        size_t contentLength;
        foreach (string line; lines[1..$])
        {
            ptrdiff_t colon = line.countUntil(":");
            if (colon < 0)
                continue;

            string key = line[0..colon].toLower;
            string value = line[colon + 1..$].strip;
            request.headers[key] = value;
            if (key == "content-length")
                contentLength = value.to!size_t;
        }

        while (body.length < contentLength)
        {
            ptrdiff_t got = client.receive(buffer[]);
            if (got <= 0)
                break;
            body ~= cast(string)buffer[0..got];
        }
        request.body = body;

        string response =
            "HTTP/1.1 "~status.to!string~" "~reason~"\r\n"
            ~"Content-Type: application/json\r\n"
            ~"Content-Length: "~responseBody.length.to!string~"\r\n"
            ~"Connection: close\r\n\r\n"
            ~responseBody;
        client.send(cast(const(ubyte)[])response);
    });

    server.start();
    run(baseUrl);
    server.join();
    return request;
}

unittest
{
    auto endpoint = new StubEndpoint;
    Completion completion = completions(endpoint, "gpt-test", "hello");
    assert(endpoint.lastModel == "gpt-test");
    assert(completion.text() == "gpt-test");
}

unittest
{
    auto endpoint = new StubEndpoint;
    auto embedding = embeddings(endpoint, "embed-test", "hello");
    assert(endpoint.lastModel == "embed-test");
    assert(embedding.value.length == 2);
    assert(embedding.value[0] == 1.0f);
}

unittest
{
    RequestData request = withServer(
        `{"data":[{"id":"gpt-test","owned_by":"openai"}]}`,
        (string baseUrl) {
            auto endpoint = new OpenAI(baseUrl~"/v1/", "secret");
            IModel[] models = endpoint.available();
            assert(models.length == 1);
            assert(models[0].name() == "gpt-test");
        },
    );

    assert(request.method == "GET");
    assert(request.path == "/v1/models");
    assert(request.headers["authorization"] == "Bearer secret");
}

unittest
{
    RequestData request = withServer(
        `{"choices":[{"message":{"content":"Hello"},"finish_reason":"stop"}]}`,
        (string baseUrl) {
            auto endpoint = new OpenAI(baseUrl, "secret");
            Completion completion = completions(endpoint, "gpt-test", "hello");
            assert(completion.text() == "Hello");
        },
    );

    assert(request.method == "POST");
    assert(request.path == "/v1/chat/completions");
    JSONValue body = parseJSON(request.body);
    assert(body["model"].str == "gpt-test");
}

unittest
{
    withServer(
        `{"error":{"message":"bad key"}}`,
        (string baseUrl) {
            auto endpoint = new OpenAI(baseUrl, "secret");
            try
            {
                auto _ = endpoint.available();
                assert(false);
            }
            catch (EndpointError e)
            {
                assert(e.status == 401);
                assert(e.body == `{"error":{"message":"bad key"}}`);
                assert(e.route.canFind("/v1/models"));
            }
        },
        401,
        "Unauthorized",
    );
}

unittest
{
    withServer(
        `not-json`,
        (string baseUrl) {
            auto endpoint = new OpenAI(baseUrl);
            try
            {
                auto _ = endpoint.available();
                assert(false);
            }
            catch (EndpointError e)
            {
                assert(e.status == 200);
                assert(e.body == "not-json");
            }
        },
    );
}
