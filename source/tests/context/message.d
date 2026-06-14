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
    json["role"].str.should == "system";
    json["content"].str.should == "Be helpful.";
}

@Name("UserMessage toJSON produces correct object")
unittest
{
    JSONValue content = JSONValue("Hello!");
    UserMessage message = new UserMessage(content);

    JSONValue json = message.toJSON();
    json["role"].str.should == "user";
    json["content"].str.should == "Hello!";
}

@Name("AssistantMessage toJSON with text only")
unittest
{
    AssistantMessage message = new AssistantMessage("Hello, world!");

    JSONValue json = message.toJSON();
    json["role"].str.should == "assistant";
    json["content"].str.should == "Hello, world!";
    ("tool_calls" in json).shouldBeNull;
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
    json["role"].str.should == "assistant";
    ("content" in json).shouldBeNull;
    json["tool_calls"].type.should == JSONType.array;
    json["tool_calls"].array.length.should == 1;
    json["tool_calls"].array[0]["id"].str.should == "call_01";
    json["tool_calls"].array[0]["function"]["name"].str.should == "get_weather";
}

@Name("AssistantMessage toJSON with both text and tool calls includes content")
unittest
{
    ToolCall call;
    call.id = "call_02";
    call.name = "search";

    AssistantMessage message = new AssistantMessage("Sure!", [call]);

    JSONValue json = message.toJSON();
    json["content"].str.should == "Sure!";
    json["tool_calls"].array.length.should == 1;
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
    message.text.should == "Result";
    message.usage.promptTokens.should == 10;
    message.usage.completionTokens.should == 5;
    message.usage.totalTokens.should == 15;
}

@Name("ToolMessage toJSON with tool call id")
unittest
{
    JSONValue content = JSONValue("Sunny");
    ToolMessage message = new ToolMessage("call_01", content);

    JSONValue json = message.toJSON();
    json["role"].str.should == "tool";
    json["tool_call_id"].str.should == "call_01";
    json["content"].str.should == "Sunny";
}

@Name("ToolMessage toJSON without tool call id omits field")
unittest
{
    JSONValue content = JSONValue("Result");
    ToolMessage message = new ToolMessage(content);

    JSONValue json = message.toJSON();
    json["role"].str.should == "tool";
    ("tool_call_id" in json).shouldBeNull;
    json["content"].str.should == "Result";
}
