module tests.context.message;

import intuit.context.message;
import intuit.response;
import unit_threaded;
import std.json : JSONValue, JSONType;

@Name("SystemMessage toJSON produces correct object")
unittest
{
    JSONValue content = JSONValue("Be helpful.");
    SystemMessage message = new SystemMessage(content);

    JSONValue json = message.toJSON();
    assert(json["role"].str == "system");
    assert(json["content"].str == "Be helpful.");
}

@Name("UserMessage toJSON produces correct object")
unittest
{
    JSONValue content = JSONValue("Hello!");
    UserMessage message = new UserMessage(content);

    JSONValue json = message.toJSON();
    assert(json["role"].str == "user");
    assert(json["content"].str == "Hello!");
}

@Name("AssistantMessage toJSON with text only")
unittest
{
    AssistantMessage message = new AssistantMessage("Hello, world!");

    JSONValue json = message.toJSON();
    assert(json["role"].str == "assistant");
    assert(json["content"].str == "Hello, world!");
    assert("tool_calls" !in json);
}

@Name("AssistantMessage toJSON with tool calls and no text omits content")
unittest
{
    ToolCall call;
    call.id = "call_01";
    call.name = "get_weather";
    call.arguments = JSONValue.emptyObject;
    call.arguments["location"] = JSONValue("Paris");

    AssistantMessage message = new AssistantMessage("", [call]);

    JSONValue json = message.toJSON();
    assert(json["role"].str == "assistant");
    assert("content" !in json);
    assert(json["tool_calls"].type == JSONType.array);
    assert(json["tool_calls"].array.length == 1);
    assert(json["tool_calls"].array[0]["id"].str == "call_01");
    assert(json["tool_calls"].array[0]["function"]["name"].str == "get_weather");
}

@Name("AssistantMessage toJSON with both text and tool calls includes content")
unittest
{
    ToolCall call;
    call.id = "call_02";
    call.name = "search";

    AssistantMessage message = new AssistantMessage("Sure!", [call]);

    JSONValue json = message.toJSON();
    assert(json["content"].str == "Sure!");
    assert(json["tool_calls"].array.length == 1);
}

@Name("AssistantMessage wraps Completion and exposes usage")
unittest
{
    Choice choice;
    choice.text = "Result";

    Completion completion;
    completion.choices = [choice];
    completion.usage.promptTokens = 10;
    completion.usage.completionTokens = 5;
    completion.usage.totalTokens = 15;

    AssistantMessage message = new AssistantMessage(completion, 0);
    assert(message.text == "Result");
    assert(message.usage.promptTokens == 10);
    assert(message.usage.completionTokens == 5);
    assert(message.usage.totalTokens == 15);
}

@Name("ToolMessage toJSON with tool call id")
unittest
{
    JSONValue content = JSONValue("Sunny");
    ToolMessage message = new ToolMessage("call_01", content);

    JSONValue json = message.toJSON();
    assert(json["role"].str == "tool");
    assert(json["tool_call_id"].str == "call_01");
    assert(json["content"].str == "Sunny");
}

@Name("ToolMessage toJSON without tool call id omits field")
unittest
{
    JSONValue content = JSONValue("Result");
    ToolMessage message = new ToolMessage(content);

    JSONValue json = message.toJSON();
    assert(json["role"].str == "tool");
    assert("tool_call_id" !in json);
    assert(json["content"].str == "Result");
}
