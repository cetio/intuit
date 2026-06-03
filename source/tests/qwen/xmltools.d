module tests.qwen.xmltools;

import std.algorithm.searching : canFind;
import intuit.qwen.model;
import intuit.response;
import std.json : JSONValue, JSONType;

unittest
{
    // Standard OpenAI-style tool_calls array still works.
    QwenModel model = new QwenModel("qwen-test");
    JSONValue response = JSONValue.emptyObject;
    JSONValue choices = JSONValue.emptyArray;
    JSONValue choice = JSONValue.emptyObject;
    JSONValue message = JSONValue.emptyObject;
    JSONValue toolCalls = JSONValue.emptyArray;
    JSONValue toolCall = JSONValue.emptyObject;
    JSONValue func = JSONValue.emptyObject;

    func["name"] = JSONValue("get_weather");
    func["arguments"] = JSONValue(`{"location":"NYC"}`);
    toolCall["id"] = JSONValue("call_123");
    toolCall["type"] = JSONValue("function");
    toolCall["function"] = func;
    toolCalls.array ~= toolCall;
    message["tool_calls"] = toolCalls;
    message["content"] = JSONValue(null);
    choice["message"] = message;
    choice["finish_reason"] = JSONValue("tool_calls");
    choices.array ~= choice;
    response["choices"] = choices;

    Completion completion = model.parseCompletions(response);
    assert(completion.choices.length == 1);
    assert(completion.choices[0].toolCalls.length == 1);
    assert(completion.choices[0].toolCalls[0].name == "get_weather");
    assert(completion.choices[0].toolCalls[0].id == "call_123");
    assert(completion.choices[0].text.length == 0);
}

unittest
{
    // Qwen3-Coder custom XML format.
    QwenModel model = new QwenModel("qwen3-coder-test");
    JSONValue response = JSONValue.emptyObject;
    JSONValue choices = JSONValue.emptyArray;
    JSONValue choice = JSONValue.emptyObject;
    JSONValue message = JSONValue.emptyObject;

    string xml = `<tool_call>
<function=search_products>
<parameter=query>waterproof running shoes</parameter>
<parameter=sort_by>price_low_to_high</parameter>
</function>
</tool_call>`;

    message["content"] = JSONValue(xml);
    choice["message"] = message;
    choice["finish_reason"] = JSONValue("stop");
    choices.array ~= choice;
    response["choices"] = choices;

    Completion completion = model.parseCompletions(response);
    assert(completion.choices.length == 1);
    assert(completion.choices[0].toolCalls.length == 1);
    assert(completion.choices[0].toolCalls[0].name == "search_products");
    assert(completion.choices[0].toolCalls[0].arguments["query"].str == "waterproof running shoes");
    assert(completion.choices[0].toolCalls[0].arguments["sort_by"].str == "price_low_to_high");
    assert(completion.choices[0].text.length == 0);
}

unittest
{
    // Qwen2.5/Qwen3 Hermes JSON-in-XML format.
    QwenModel model = new QwenModel("qwen2.5-test");
    JSONValue response = JSONValue.emptyObject;
    JSONValue choices = JSONValue.emptyArray;
    JSONValue choice = JSONValue.emptyObject;
    JSONValue message = JSONValue.emptyObject;

    string xml = `<tool_call>{"name": "get_weather", "arguments": {"location": "San Francisco"}}</tool_call>`;

    message["content"] = JSONValue(xml);
    choice["message"] = message;
    choice["finish_reason"] = JSONValue("stop");
    choices.array ~= choice;
    response["choices"] = choices;

    Completion completion = model.parseCompletions(response);
    assert(completion.choices.length == 1);
    assert(completion.choices[0].toolCalls.length == 1);
    assert(completion.choices[0].toolCalls[0].name == "get_weather");
    assert(completion.choices[0].toolCalls[0].arguments["location"].str == "San Francisco");
    assert(completion.choices[0].text.length == 0);
}

unittest
{
    // Mixed text and XML tool calls: text should be preserved.
    QwenModel model = new QwenModel("qwen-test");
    JSONValue response = JSONValue.emptyObject;
    JSONValue choices = JSONValue.emptyArray;
    JSONValue choice = JSONValue.emptyObject;
    JSONValue message = JSONValue.emptyObject;

    string xml = `I will search for shoes.
<tool_call>{"name": "search", "arguments": {"q": "shoes"}}</tool_call>
Here are the results.`;

    message["content"] = JSONValue(xml);
    choice["message"] = message;
    choice["finish_reason"] = JSONValue("stop");
    choices.array ~= choice;
    response["choices"] = choices;

    Completion completion = model.parseCompletions(response);
    assert(completion.choices.length == 1);
    assert(completion.choices[0].toolCalls.length == 1);
    assert(completion.choices[0].toolCalls[0].name == "search");
    assert(completion.choices[0].text.canFind("I will search for shoes."));
    assert(completion.choices[0].text.canFind("Here are the results."));
    assert(!completion.choices[0].text.canFind("<tool_call>"));
}

unittest
{
    // Multiple JSON-in-XML tool calls (parallel).
    QwenModel model = new QwenModel("qwen-test");
    JSONValue response = JSONValue.emptyObject;
    JSONValue choices = JSONValue.emptyArray;
    JSONValue choice = JSONValue.emptyObject;
    JSONValue message = JSONValue.emptyObject;

    string xml = `<tool_call>{"name": "a", "arguments": {}}</tool_call>`
        ~`<tool_call>{"name": "b", "arguments": {}}</tool_call>`;

    message["content"] = JSONValue(xml);
    choice["message"] = message;
    choice["finish_reason"] = JSONValue("stop");
    choices.array ~= choice;
    response["choices"] = choices;

    Completion completion = model.parseCompletions(response);
    assert(completion.choices[0].toolCalls.length == 2);
    assert(completion.choices[0].toolCalls[0].name == "a");
    assert(completion.choices[0].toolCalls[1].name == "b");
}

unittest
{
    // No tool calls in content: text should be untouched.
    QwenModel model = new QwenModel("qwen-test");
    JSONValue response = JSONValue.emptyObject;
    JSONValue choices = JSONValue.emptyArray;
    JSONValue choice = JSONValue.emptyObject;
    JSONValue message = JSONValue.emptyObject;

    message["content"] = JSONValue("Just a regular message.");
    choice["message"] = message;
    choice["finish_reason"] = JSONValue("stop");
    choices.array ~= choice;
    response["choices"] = choices;

    Completion completion = model.parseCompletions(response);
    assert(completion.choices[0].toolCalls.length == 0);
    assert(completion.choices[0].text == "Just a regular message.");
}
