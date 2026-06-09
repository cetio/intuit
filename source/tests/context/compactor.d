module tests.context.compactor;

import intuit.context.compactor;
import intuit.context.message;
import intuit.response;
import unit_threaded;
import std.json : JSONValue;

@Name("Trim removes oldest non-system messages to respect maxMessages")
unittest
{
    Compactor compactor = new Compactor();
    compactor.maxMessages = 3;
    compactor.maxTokens = 0;
    compactor.strategy = CompactorStrategy.Trim;

    IMessage[] messages = [
        cast(IMessage)new UserMessage(JSONValue("first")),
        cast(IMessage)new UserMessage(JSONValue("second")),
        cast(IMessage)new UserMessage(JSONValue("third")),
        cast(IMessage)new UserMessage(JSONValue("fourth")),
    ];

    IMessage[] ret = compactor.compact(messages);
    assert(ret.length == 3);
    assert((cast(UserMessage)ret[0]).content.str == "second");
}

@Name("Trim preserves system messages")
unittest
{
    Compactor compactor = new Compactor();
    compactor.maxMessages = 2;
    compactor.maxTokens = 0;

    IMessage[] messages = [
        cast(IMessage)new SystemMessage(JSONValue("sys")),
        cast(IMessage)new UserMessage(JSONValue("user")),
        cast(IMessage)new UserMessage(JSONValue("extra")),
    ];

    IMessage[] ret = compactor.compact(messages);
    assert(ret.length == 2);
    assert(ret[0].role == Role.System);
    assert((cast(UserMessage)ret[1]).content.str == "extra");
}

@Name("Trim respects maxTokens using completion usage")
unittest
{
    Compactor compactor = new Compactor();
    compactor.maxMessages = 0;
    compactor.maxTokens = 25;

    Choice choice;
    choice.text = "a";

    Completion completion1;
    completion1.choices = [choice];
    completion1.usage.promptTokens = 10;
    completion1.usage.completionTokens = 5;

    Completion completion2;
    completion2.choices = [choice];
    completion2.usage.promptTokens = 20;
    completion2.usage.completionTokens = 10;

    IMessage[] messages = [
        cast(IMessage)new UserMessage(JSONValue("u1")),
        cast(IMessage)new AssistantMessage(completion1),
        cast(IMessage)new UserMessage(JSONValue("u2")),
        cast(IMessage)new AssistantMessage(completion2),
    ];

    IMessage[] ret = compactor.compact(messages);
    assert(ret.length == 0, "Expected empty after trimming all messages to get under token limit");
}

@Name("lowest limit wins when both maxMessages and maxTokens are set")
unittest
{
    Compactor compactor = new Compactor();
    compactor.maxMessages = 5;
    compactor.maxTokens = 10;

    Choice choice;
    choice.text = "a";

    Completion completion;
    completion.choices = [choice];
    completion.usage.promptTokens = 8;
    completion.usage.completionTokens = 5;

    IMessage[] messages = [
        cast(IMessage)new UserMessage(JSONValue("u1")),
        cast(IMessage)new AssistantMessage(completion),
    ];

    // total = 5 + 8 = 13 > 10 (token limit is hit)
    IMessage[] ret = compactor.compact(messages);
    assert(ret.length < 2, "Token limit should trigger before message limit");
}

@Name("Callback delegates receives and replaces messages")
unittest
{
    Compactor compactor = new Compactor();
    compactor.maxMessages = 1;
    compactor.maxTokens = 0;
    compactor.strategy = CompactorStrategy.Callback;
    compactor.callback = (IMessage[] msgs) {
        return cast(IMessage[])[new SystemMessage(JSONValue("Summary"))];
    };

    IMessage[] messages = [
        cast(IMessage)new UserMessage(JSONValue("hello")),
        cast(IMessage)new UserMessage(JSONValue("world")),
    ];

    IMessage[] ret = compactor.compact(messages);
    assert(ret.length == 1);
    assert(ret[0].role == Role.System);
    assert((cast(SystemMessage)ret[0]).content.str == "Summary");
}

@Name("Callback with null delegate returns input unchanged")
unittest
{
    Compactor compactor = new Compactor();
    compactor.strategy = CompactorStrategy.Callback;

    IMessage[] messages = [
        cast(IMessage)new UserMessage(JSONValue("hello")),
    ];

    IMessage[] ret = compactor.compact(messages);
    assert(ret.length == 1);
    assert(ret[0].role == Role.User);
}

@Name("tokens computes sum of completion plus latest prompt")
unittest
{
    Choice choice;
    choice.text = "a";

    Completion c1;
    c1.choices = [choice];
    c1.usage.promptTokens = 10;
    c1.usage.completionTokens = 5;

    Completion c2;
    c2.choices = [choice];
    c2.usage.promptTokens = 20;
    c2.usage.completionTokens = 8;

    IMessage[] messages = [
        cast(IMessage)new AssistantMessage(c1),
        cast(IMessage)new AssistantMessage(c2),
    ];

    // 5 + 8 + 20 = 33
    assert(Compactor.tokens(messages) == 33);
}

@Name("no-op when under all limits")
unittest
{
    Compactor compactor = new Compactor();
    compactor.maxMessages = 10;
    compactor.maxTokens = 1000;

    IMessage[] messages = [
        cast(IMessage)new UserMessage(JSONValue("hello")),
    ];

    IMessage[] ret = compactor.compact(messages);
    assert(ret.length == 1);
    assert(ret[0].role == Role.User);
}
