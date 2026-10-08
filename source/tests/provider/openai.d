module tests.provider.openai;

import intuit.exception :
    AuthException,
    MalformedResponseException,
    RateLimitException,
    RequestTimeoutException,
    TransportException;
import intuit.model;
import intuit.provider.openai : OpenAI;
import intuit.response;
import unit_threaded;

import std.json : JSONValue, JSONType;

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
