module tests.provider.openai;

import intuit.model;
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
