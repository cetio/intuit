module tests.provider.qwen.model;

import intuit.provider.qwen.model;
import intuit.response;
import unit_threaded;

import std.algorithm.searching : canFind;
import std.json : JSONValue;

@Name("standard OpenAI-style tool_calls array")
unittest
{
    QwenModelConfig cfg = new QwenModelConfig("qwen-test");
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

    Completion completion = cfg.parseResponse(response);
    completion.choices.length.should == 1;
    completion.choices[0].toolCalls.length.should == 1;
    completion.choices[0].toolCalls[0].name.should == "get_weather";
    completion.choices[0].toolCalls[0].id.should == "call_123";
    completion.choices[0].text.length.should == 0;
}

@Name("Qwen3-Coder custom XML format")
unittest
{
    QwenModelConfig cfg = new QwenModelConfig("qwen3-coder-test");
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

    Completion completion = cfg.parseResponse(response);
    completion.choices.length.should == 1;
    completion.choices[0].toolCalls.length.should == 1;
    completion.choices[0].toolCalls[0].name.should == "search_products";
    completion.choices[0].toolCalls[0].arguments["query"].str.should == "waterproof running shoes";
    completion.choices[0].toolCalls[0].arguments["sort_by"].str.should == "price_low_to_high";
    completion.choices[0].text.length.should == 0;
}

@Name("Qwen2.5/Qwen3 Hermes JSON-in-XML format")
unittest
{
    QwenModelConfig cfg = new QwenModelConfig("qwen2.5-test");
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

    Completion completion = cfg.parseResponse(response);
    completion.choices.length.should == 1;
    completion.choices[0].toolCalls.length.should == 1;
    completion.choices[0].toolCalls[0].name.should == "get_weather";
    completion.choices[0].toolCalls[0].arguments["location"].str.should == "San Francisco";
    completion.choices[0].text.length.should == 0;
}

@Name("mixed text and XML tool calls preserves surrounding prose")
unittest
{
    QwenModelConfig cfg = new QwenModelConfig("qwen-test");
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

    Completion completion = cfg.parseResponse(response);
    completion.choices.length.should == 1;
    completion.choices[0].toolCalls.length.should == 1;
    completion.choices[0].toolCalls[0].name.should == "search";
    completion.choices[0].text.canFind("I will search for shoes.").should == true;
    completion.choices[0].text.canFind("Here are the results.").should == true;
    completion.choices[0].text.canFind("<tool_call>").should == false;
}

@Name("multiple JSON-in-XML tool calls in parallel")
unittest
{
    QwenModelConfig cfg = new QwenModelConfig("qwen-test");
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

    Completion completion = cfg.parseResponse(response);
    completion.choices[0].toolCalls.length.should == 2;
    completion.choices[0].toolCalls[0].name.should == "a";
    completion.choices[0].toolCalls[1].name.should == "b";
}

@Name("no tool calls in content leaves text untouched")
unittest
{
    QwenModelConfig model = new QwenModelConfig("qwen-test");
    JSONValue response = JSONValue.emptyObject;
    JSONValue choices = JSONValue.emptyArray;
    JSONValue choice = JSONValue.emptyObject;
    JSONValue message = JSONValue.emptyObject;

    message["content"] = JSONValue("Just a regular message.");
    choice["message"] = message;
    choice["finish_reason"] = JSONValue("stop");
    choices.array ~= choice;
    response["choices"] = choices;

    Completion completion = model.parseResponse(response);
    completion.choices[0].toolCalls.length.should == 0;
    completion.choices[0].text.should == "Just a regular message.";
}
