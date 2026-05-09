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

    assert(completion.text == `{"name":"Quiz"}`);
    assert(completion.reasoning == "Because");
    assert(completion.json["name"].str == "Quiz");
}

unittest
{
    Completion completion;
    Choice choice;
    choice.text = "not json";
    completion.choices = [choice];

    assertThrown!Exception(completion.json);
}

unittest
{
    Completion completion;
    assertThrown!Exception(completion.choice());
}

unittest
{
    auto stream = new CompletionStream("model", null);
    Completion completion;
    Choice choice;
    choice.text = "hello";
    completion.choices = [choice];

    stream.commence((CompletionStream current) {
        current.update(completion);
        current.complete = true;
    });
    assert(stream.next().text == "hello");
}
