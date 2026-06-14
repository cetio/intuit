module tests.claude.model;

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

    assert(payload["model"].str == "claude-opus-4-8");
    assert(payload["max_tokens"].integer == 1024);
    assert(payload["temperature"].floating == 0.5);
    assert(payload["system"].str == "Be helpful.");
    assert(payload["messages"].type == JSONType.array);
    assert(payload["messages"].array.length == 1);
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

    assert(payload["system"].str == "Original system.\nExtracted system.");
    assert(payload["messages"].array.length == 1);
    assert(payload["messages"].array[0]["role"].str == "user");
}

@Name("buildPayload wraps raw string input")
unittest
{
    ClaudeModelConfig cfg = new ClaudeModelConfig("claude-opus-4-8");
    JSONValue payload = cfg.buildPayload(JSONValue("Hello"));

    assert(payload["messages"].type == JSONType.array);
    assert(payload["messages"].array.length == 1);
    assert(payload["messages"].array[0]["role"].str == "user");
    assert(payload["messages"].array[0]["content"].str == "Hello");
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

    assert(completion.choices.length == 1);
    assert(completion.choices[0].text == "Hello!");
    assert(completion.choices[0].finishReason == FinishReason.EndTurn);
    assert(completion.usage.promptTokens == 12);
    assert(completion.usage.completionTokens == 6);
    assert(completion.usage.totalTokens == 18);
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

    assert(completion.choices.length == 1);
    assert(completion.choices[0].toolCalls.length == 1);
    assert(completion.choices[0].toolCalls[0].id == "toolu_01");
    assert(completion.choices[0].toolCalls[0].name == "get_weather");
    assert(completion.choices[0].finishReason == FinishReason.ToolUse);
}

@Name("embeddingsJSON throws for Claude")
unittest
{
    ClaudeModelConfig cfg = new ClaudeModelConfig("claude-opus-4-8");
    bool threw = false;
    try
        cfg.parseEmbeddingsResponse(JSONValue("test"));
    catch (Exception)
        threw = true;
    assert(threw);
}
