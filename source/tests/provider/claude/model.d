module tests.provider.claude.model;

import intuit.provider.claude.model;
import intuit.response;
import unit_threaded;
import std.json : JSONValue, JSONType;

@Name("completionsJSON builds correct payload")
unittest
{
    ClaudeModelConfig cfg = new ClaudeModelConfig("claude-opus-4-8");
    cfg.maxTokens = 1024;
    cfg.temperature = 0.5;
    cfg.system = "Be helpful.";

    JSONValue input = JSONValue.emptyArray;
    JSONValue msg = JSONValue.emptyObject;
    msg["role"] = JSONValue("user");
    msg["content"] = JSONValue("Hello");
    input.array ~= msg;

    JSONValue payload = cfg.buildPayload(input);

    payload["model"].str.should == "claude-opus-4-8";
    payload["max_tokens"].integer.should == 1024;
    payload["temperature"].floating.should == 0.5;
    payload["system"].str.should == "Be helpful.";
    payload["messages"].type.should == JSONType.array;
    payload["messages"].array.length.should == 1;
}

@Name("buildPayload extracts role system messages to top level")
unittest
{
    ClaudeModelConfig cfg = new ClaudeModelConfig("claude-opus-4-8");
    cfg.system = "Original system.";

    JSONValue input = JSONValue.emptyArray;
    
    JSONValue sysMsg = JSONValue.emptyObject;
    sysMsg["role"] = JSONValue("system");
    sysMsg["content"] = JSONValue("Extracted system.");
    input.array ~= sysMsg;

    JSONValue userMsg = JSONValue.emptyObject;
    userMsg["role"] = JSONValue("user");
    userMsg["content"] = JSONValue("Hello");
    input.array ~= userMsg;

    JSONValue payload = cfg.buildPayload(input);

    payload["system"].str.should == "Original system.\nExtracted system.";
    payload["messages"].array.length.should == 1;
    payload["messages"].array[0]["role"].str.should == "user";
}

@Name("buildPayload wraps raw string input")
unittest
{
    ClaudeModelConfig cfg = new ClaudeModelConfig("claude-opus-4-8");
    JSONValue payload = cfg.buildPayload(JSONValue("Hello"));

    payload["messages"].type.should == JSONType.array;
    payload["messages"].array.length.should == 1;
    payload["messages"].array[0]["role"].str.should == "user";
    payload["messages"].array[0]["content"].str.should == "Hello";
}

@Name("parseResponse extracts text and finish reason")
unittest
{
    ClaudeModelConfig cfg = new ClaudeModelConfig("claude-opus-4-8");

    JSONValue json = JSONValue.emptyObject;
    json["id"] = JSONValue("msg_01");
    json["type"] = JSONValue("message");
    json["role"] = JSONValue("assistant");

    JSONValue content = JSONValue.emptyArray;
    JSONValue block = JSONValue.emptyObject;
    block["type"] = JSONValue("text");
    block["text"] = JSONValue("Hello!");
    content.array ~= block;
    json["content"] = content;

    json["model"] = JSONValue("claude-opus-4-8");
    json["stop_reason"] = JSONValue("end_turn");
    json["stop_sequence"] = JSONValue(null);
    json["usage"] = JSONValue.emptyObject;
    json["usage"]["input_tokens"] = JSONValue(12);
    json["usage"]["output_tokens"] = JSONValue(6);

    Completion completion = cfg.parseResponse(json);

    completion.choices.length.should == 1;
    completion.choices[0].text.should == "Hello!";
    completion.choices[0].finishReason.should == FinishReason.EndTurn;
    completion.usage.promptTokens.should == 12;
    completion.usage.completionTokens.should == 6;
    completion.usage.totalTokens.should == 18;
}

@Name("parseResponse extracts tool use blocks")
unittest
{
    ClaudeModelConfig cfg = new ClaudeModelConfig("claude-opus-4-8");

    JSONValue json = JSONValue.emptyObject;
    json["id"] = JSONValue("msg_02");
    json["type"] = JSONValue("message");
    json["role"] = JSONValue("assistant");

    JSONValue content = JSONValue.emptyArray;
    JSONValue block = JSONValue.emptyObject;
    block["type"] = JSONValue("tool_use");
    block["id"] = JSONValue("toolu_01");
    block["name"] = JSONValue("get_weather");
    block["input"] = JSONValue.emptyObject;
    block["input"]["location"] = JSONValue("San Francisco");
    content.array ~= block;
    json["content"] = content;

    json["model"] = JSONValue("claude-opus-4-8");
    json["stop_reason"] = JSONValue("tool_use");

    Completion completion = cfg.parseResponse(json);

    completion.choices.length.should == 1;
    completion.choices[0].toolCalls.length.should == 1;
    completion.choices[0].toolCalls[0].id.should == "toolu_01";
    completion.choices[0].toolCalls[0].name.should == "get_weather";
    completion.choices[0].finishReason.should == FinishReason.ToolUse;
}

@Name("parseResponse captures usage metadata and latency")
unittest
{
    ClaudeModelConfig cfg = new ClaudeModelConfig("claude-opus-4-8");

    JSONValue json = JSONValue.emptyObject;
    json["id"] = JSONValue("msg_03");
    json["type"] = JSONValue("message");
    json["role"] = JSONValue("assistant");
    json["model"] = JSONValue("claude-opus-4-8-resolved");
    json["latency"] = JSONValue(123.456f);

    JSONValue content = JSONValue.emptyArray;
    JSONValue block = JSONValue.emptyObject;
    block["type"] = JSONValue("text");
    block["text"] = JSONValue("Hello!");
    content.array ~= block;
    json["content"] = content;

    json["stop_reason"] = JSONValue("end_turn");
    json["usage"] = JSONValue.emptyObject;
    json["usage"]["input_tokens"] = JSONValue(10);
    json["usage"]["output_tokens"] = JSONValue(5);
    json["usage"]["cache_read_input_tokens"] = JSONValue(3);
    json["usage"]["cache_creation_input_tokens"] = JSONValue(2);

    Completion completion = cfg.parseResponse(json);

    completion.usage.modelName.should == "claude-opus-4-8-resolved";
    completion.usage.latency.should == 123.456f;
    completion.usage.promptTokens.should == 15;
    completion.usage.completionTokens.should == 5;
    completion.usage.totalTokens.should == 20;
    completion.usage.cacheHits.should == 3;
    completion.usage.cacheMisses.should == 12;
}

@Name("parseResponse falls back to config name when model is missing")
unittest
{
    ClaudeModelConfig cfg = new ClaudeModelConfig("claude-opus-4-8");

    JSONValue json = JSONValue.emptyObject;
    json["id"] = JSONValue("msg_04");
    json["type"] = JSONValue("message");
    json["role"] = JSONValue("assistant");
    json["content"] = JSONValue.emptyArray;
    json["stop_reason"] = JSONValue("end_turn");
    json["usage"] = JSONValue.emptyObject;
    json["usage"]["input_tokens"] = JSONValue(1);
    json["usage"]["output_tokens"] = JSONValue(1);

    Completion completion = cfg.parseResponse(json);

    completion.usage.modelName.should == "claude-opus-4-8";
    completion.usage.promptTokens.should == 1;
    completion.usage.completionTokens.should == 1;
    completion.usage.totalTokens.should == 2;
}

@Name("embeddingsJSON throws for Claude")
unittest
{
    ClaudeModelConfig cfg = new ClaudeModelConfig("claude-opus-4-8");
    cfg.parseEmbeddingsResponse(JSONValue("test")).shouldThrow!Exception;
}
