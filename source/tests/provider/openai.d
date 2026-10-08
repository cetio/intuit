module tests.provider.openai;

import intuit.exception :
    AuthException,
    MalformedResponseException,
    RateLimitException,
    RequestTimeoutException,
    TransportException;
import intuit.model;
import intuit.provider : IEndpoint, legacyCompletions;
import intuit.provider.claude : Claude;
import intuit.provider.openai : OpenAI;
import intuit.provider.qwen : Qwen;
import intuit.provider.typesafe : TypeSafe;
import intuit.router : IRouter, LiteLLM, ModelsDev, OpenRouter;
import intuit.response;
import unit_threaded;

import std.conv : to;
import std.json : JSONValue, JSONType, parseJSON;
import std.socket : InternetAddress, Socket, TcpSocket;
import core.thread : Thread;
import core.time : Duration, dur;

@Name("Legacy completions return parsed text choices and usage")
unittest
{
    withServer(`{"model":"resolved","choices":[
        {"text":"A","finish_reason":"stop","logprobs":{"tokens":["A"]}},
        {"text":"B","finish_reason":"length"}],
        "usage":{"prompt_tokens":12,"completion_tokens":2,"total_tokens":14}}`, delegate void(string url) {
        OpenAI endpoint = new OpenAI(url);
        Completion ret = legacyCompletions(endpoint, parseJSON(`{"model":"decider","prompt":"Answer: ("}`));
        ret.text.should == "A";
        ret.choice.content.str.should == "A";
        ret.text(1).should == "B";
        ret.choice.finishReason.should == FinishReason.Stop;
        ret.choice(1).finishReason.should == FinishReason.Length;
        ret.choice.logProbs["tokens"][0].str.should == "A";
        ret.usage.modelName.should == "resolved";
        ret.usage.promptTokens.should == 12;
        ret.usage.completionTokens.should == 2;
        ret.usage.totalTokens.should == 14;
        ret.raw["choices"][0]["text"].str.should == "A";
    });
}

@Name("Endpoint operation timeouts apply to HTTP requests")
unittest
{
    withServer(
        null,
        delegate void(string url) {
            IEndpoint[] endpoints = [
                cast(IEndpoint)new OpenAI(url),
                new Claude(url),
                new Qwen(url),
                new TypeSafe(url),
            ];
            foreach (endpoint; endpoints)
            {
                endpoint.connectTimeout(dur!"seconds"(1));
                endpoint.operationTimeout(dur!"msecs"(30));
                endpoint.available().shouldThrow!RequestTimeoutException;
            }
        },
        dur!"msecs"(200),
        4,
    );
}

@Name("Router operation timeouts apply to catalog requests")
unittest
{
    withServer(
        null,
        delegate void(string url) {
            IRouter[] routers = [cast(IRouter)new OpenRouter(null, url), new LiteLLM(url), new ModelsDev(url)];
            foreach (router; routers)
            {
                router.connectTimeout(dur!"seconds"(1));
                router.operationTimeout(dur!"msecs"(30));
                router.refresh().shouldThrow!RequestTimeoutException;
            }
        },
        dur!"msecs"(200),
        3,
    );
}

@Name("ModelConfig parseResponse captures OpenAI usage and latency")
unittest
{
    ModelConfig cfg = new ModelConfig("gpt-4");

    JSONValue json = JSONValue.emptyObject;
    json["model"] = JSONValue("gpt-4-Resolved");
    json["latency"] = JSONValue(99.5f);

    JSONValue choices = JSONValue.emptyArray;
    JSONValue choice = JSONValue.emptyObject;
    JSONValue message = JSONValue.emptyObject;
    message["role"] = JSONValue("assistant");
    message["content"] = JSONValue("Hi!");
    choice["message"] = message;
    choice["finish_reason"] = JSONValue("stop");
    choices.array ~= choice;
    json["choices"] = choices;

    json["usage"] = JSONValue.emptyObject;
    json["usage"]["prompt_tokens"] = JSONValue(20);
    json["usage"]["completion_tokens"] = JSONValue(10);
    json["usage"]["total_tokens"] = JSONValue(30);
    json["usage"]["prompt_tokens_details"] = JSONValue.emptyObject;
    json["usage"]["prompt_tokens_details"]["cached_tokens"] = JSONValue(8);

    Completion completion = cfg.parseResponse(json);

    completion.usage.modelName.should == "gpt-4-Resolved";
    completion.usage.latency.should == 99.5f;
    completion.usage.promptTokens.should == 20;
    completion.usage.completionTokens.should == 10;
    completion.usage.totalTokens.should == 30;
    completion.usage.cacheHits.should == 8;
    completion.usage.cacheMisses.should == 12;
}

@Name("Exception subclasses preserve endpoint and transport details")
unittest
{
    AuthException auth = new AuthException("GET", "/v1/models", 401, "Unauthorized", "");
    auth.status.should == 401;

    RateLimitException rateLimit = new RateLimitException(
        "POST",
        "/v1/chat/completions",
        429,
        "Too Many Requests",
        "",
    );
    rateLimit.status.should == 429;

    TransportException transport = new TransportException("GET", "/v1/models", "connection refused");
    transport.method.should == "GET";
    transport.detail.should == "connection refused";
    transport.status.should == 0;

    RequestTimeoutException timeout = new RequestTimeoutException("GET", "/v1/models", "deadline exceeded");
    assert(cast(TransportException)timeout !is null);
}

@Name("OpenAI endpoint normalizes curl transport failures")
unittest
{
    OpenAI endpoint = new OpenAI("unsupported://transport-test");
    endpoint.available().shouldThrow!TransportException;
}

@Name("Generic legacy completions delegate to OpenAI endpoints")
unittest
{
    OpenAI endpoint = new OpenAI("unsupported://transport-test");
    endpoint.legacyCompletions(JSONValue.emptyObject).shouldThrow!TransportException;
}

@Name("Completion JSON parsing classifies malformed model text")
unittest
{
    Completion completion;
    completion.raw = JSONValue.emptyObject;
    Choice choice;
    choice.text = "{";
    completion.choices ~= choice;
    completion.json().shouldThrow!MalformedResponseException;
}

@Name("ModelConfig normalizes shorthand response schema nodes")
unittest
{
    ModelConfig cfg = new ModelConfig("gpt-4");
    JSONValue schema = JSONValue.emptyObject;
    schema["type"] = JSONValue("object");
    schema["properties"] = JSONValue.emptyObject;
    schema["properties"]["answer"] = JSONValue("string");
    schema["properties"]["tags"] = JSONValue.emptyObject;
    schema["properties"]["tags"]["type"] = JSONValue("array");
    schema["properties"]["tags"]["items"] = JSONValue("string");
    schema["properties"]["status"] = JSONValue.emptyObject;
    schema["properties"]["status"]["type"] = JSONValue("string");
    schema["properties"]["status"]["enum"] = JSONValue([JSONValue("draft"), JSONValue("published")]);

    cfg.setResponseSchema("answer_schema", schema);
    JSONValue normalized = cfg.responseSchema["json_schema"]["schema"];

    normalized["properties"]["answer"]["type"].str.should == "string";
    normalized["properties"]["tags"]["items"]["type"].str.should == "string";
    normalized["properties"]["status"]["enum"][0].str.should == "draft";
}

@Name("ModelConfig parses Gemma reasoning_content")
unittest
{
    ModelConfig cfg = new ModelConfig("google/gemma-4-e4b");

    JSONValue json = JSONValue.emptyObject;
    JSONValue choices = JSONValue.emptyArray;
    JSONValue choice = JSONValue.emptyObject;
    JSONValue message = JSONValue.emptyObject;
    message["content"] = JSONValue("The answer.");
    message["reasoning_content"] = JSONValue("First inspect the clues. Then answer.");
    choice["message"] = message;
    choice["finish_reason"] = JSONValue("stop");
    choices.array ~= choice;
    json["choices"] = choices;

    Completion completion = cfg.parseResponse(json);

    completion.text.should == "The answer.";
    completion.reasoning.should == "First inspect the clues. Then answer.";
}

@Name("ModelConfig exposes truncated Gemma reasoning outside raw JSON")
unittest
{
    ModelConfig cfg = new ModelConfig("google/gemma-4-e4b");

    JSONValue json = JSONValue.emptyObject;
    JSONValue choices = JSONValue.emptyArray;
    JSONValue choice = JSONValue.emptyObject;
    JSONValue message = JSONValue.emptyObject;
    message["content"] = JSONValue("");
    message["reasoning_content"] = JSONValue("The response needs more tokens to finish.");
    choice["message"] = message;
    choice["finish_reason"] = JSONValue("length");
    choices.array ~= choice;
    json["choices"] = choices;

    Completion completion = cfg.parseResponse(json);

    completion.text.should == "";
    completion.reasoning.should == "The response needs more tokens to finish.";
    completion.choice.finishReason.should == FinishReason.Length;
}

@Name("ModelConfig parseResponse derives total and cache miss")
unittest
{
    ModelConfig cfg = new ModelConfig("gpt-4");

    JSONValue json = JSONValue.emptyObject;
    JSONValue choices = JSONValue.emptyArray;
    JSONValue choice = JSONValue.emptyObject;
    JSONValue message = JSONValue.emptyObject;
    message["role"] = JSONValue("assistant");
    message["content"] = JSONValue("Hi!");
    choice["message"] = message;
    choice["finish_reason"] = JSONValue("stop");
    choices.array ~= choice;
    json["choices"] = choices;

    json["usage"] = JSONValue.emptyObject;
    json["usage"]["prompt_tokens"] = JSONValue(15);
    json["usage"]["completion_tokens"] = JSONValue(5);
    json["usage"]["prompt_tokens_details"] = JSONValue.emptyObject;
    json["usage"]["prompt_tokens_details"]["cached_tokens"] = JSONValue(3);

    Completion completion = cfg.parseResponse(json);

    completion.usage.modelName.should == "gpt-4";
    completion.usage.totalTokens.should == 20;
    completion.usage.cacheHits.should == 3;
    completion.usage.cacheMisses.should == 12;
}

private:

void withServer(
    string response,
    scope void delegate(string) request,
    Duration delay = Duration.zero,
    size_t count = 1,
)
{
    TcpSocket listener = new TcpSocket();
    listener.bind(new InternetAddress("127.0.0.1", 0));
    listener.listen(cast(int)count);
    scope(exit)
        listener.close();

    Thread server = new Thread(delegate void() {
        foreach (i; 0..count)
        {
            Socket connection = listener.accept();
            scope(exit)
                connection.close();

            if (delay > Duration.zero)
                Thread.sleep(delay);

            if (response !is null)
            {
                ubyte[8192] buffer;
                connection.receive(buffer);
                connection.send("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                    ~response.length.to!string~"\r\nConnection: close\r\n\r\n"~response);
            }
        }
    });
    server.start();
    scope(exit)
        server.join();

    request("http://127.0.0.1:"~(cast(InternetAddress)listener.localAddress).port.to!string);
}
