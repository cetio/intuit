module tests.stream.delta;

import intuit.stream.delta;
import intuit.response;
import unit_threaded;
import std.json : JSONValue;

@Name("accumulates text across chunks")
unittest
{
    DeltaAccumulator acc = new DeltaAccumulator();

    Completion chunk1;
    chunk1.choices = [Choice.init];
    chunk1.choices[0].text = "Hello";
    acc.accumulate(chunk1);

    Completion chunk2;
    chunk2.choices = [Choice.init];
    chunk2.choices[0].text = " world";
    acc.accumulate(chunk2);

    assert(acc.current.text == "Hello world");
}

@Name("accumulates reasoning across chunks")
unittest
{
    DeltaAccumulator acc = new DeltaAccumulator();

    Completion chunk1;
    chunk1.choices = [Choice.init];
    chunk1.choices[0].reasoning = "Thinking:";
    acc.accumulate(chunk1);

    Completion chunk2;
    chunk2.choices = [Choice.init];
    chunk2.choices[0].reasoning = " done";
    acc.accumulate(chunk2);

    assert(acc.current.reasoning(0) == "Thinking: done");
}

@Name("appends tool calls across chunks")
unittest
{
    DeltaAccumulator acc = new DeltaAccumulator();

    Completion chunk1;
    chunk1.choices = [Choice.init];
    chunk1.choices[0].toolCalls = [ToolCall("1", "calc", JSONValue.emptyObject)];
    acc.accumulate(chunk1);

    Completion chunk2;
    chunk2.choices = [Choice.init];
    chunk2.choices[0].toolCalls = [ToolCall("2", "search", JSONValue.emptyObject)];
    acc.accumulate(chunk2);

    assert(acc.current.choice(0).toolCalls.length == 2);
    assert(acc.current.choice(0).toolCalls[0].name == "calc");
    assert(acc.current.choice(0).toolCalls[1].name == "search");
}

@Name("captures finish reason from first non-unknown chunk")
unittest
{
    DeltaAccumulator acc = new DeltaAccumulator();

    Completion chunk1;
    chunk1.choices = [Choice.init];
    chunk1.choices[0].text = "ok";
    acc.accumulate(chunk1);

    Completion chunk2;
    chunk2.choices = [Choice.init];
    chunk2.choices[0].finishReason = FinishReason.Stop;
    acc.accumulate(chunk2);

    assert(acc.current.choice(0).finishReason == FinishReason.Stop);
}

@Name("handles multiple choices")
unittest
{
    DeltaAccumulator acc = new DeltaAccumulator();

    Completion chunk;
    chunk.choices = [Choice.init, Choice.init];
    chunk.choices[0].text = "A";
    chunk.choices[1].text = "B";
    acc.accumulate(chunk);

    assert(acc.current.choices.length == 2);
    assert(acc.current.choices[0].text == "A");
    assert(acc.current.choices[1].text == "B");
}

@Name("ignores empty chunks")
unittest
{
    DeltaAccumulator acc = new DeltaAccumulator();

    Completion empty;
    acc.accumulate(empty);

    assert(acc.current.choices.length == 1);
    assert(acc.current.choices[0].text == "");
}
