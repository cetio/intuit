module checks.completion;

import intuit;
import std.exception : assertThrown;

unittest
{
    Completion completion;
    Choice choice;
    choice.text = `{"name":"Quiz"}`;
    choice.reasoning = "Because";
    completion.choices = [choice];

    assert(completion.text() == `{"name":"Quiz"}`);
    assert(completion.reasoning() == "Because");
    assert(completion.parsedJSON()["name"].str == "Quiz");
}

unittest
{
    Completion completion;
    Choice choice;
    choice.text = "```json\n{\"name\":\"Quiz\"}\n```";
    completion.choices = [choice];

    assert(completion.parsedJSON()["name"].str == "Quiz");
}

unittest
{
    Completion completion;
    Choice choice;
    choice.text = "Here you go:\n{\"note\":\"use {braces}\"}";
    completion.choices = [choice];

    assert(completion.parsedJSON()["note"].str == "use {braces}");
}

unittest
{
    Completion completion;
    Choice choice;
    choice.text = "not json";
    completion.choices = [choice];

    try
    {
        auto _ = completion.parsedJSON();
        assert(false);
    }
    catch (CompletionParseError e)
    {
        assert(e.rawText == "not json");
        assert(e.candidateText is null);
    }
}

unittest
{
    Completion completion;
    assertThrown!Exception(completion.choice());
}
