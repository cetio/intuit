module checks.model;

import intuit;
import intuit.openai : OpenAIModel;
import std.json : JSONType, JSONValue, parseJSON;

unittest
{
    auto model = new OpenAIModel("gpt-test");
    JSONValue payload = model.completionsJSON(JSONValue("hello"));

    assert("max_tokens" !in payload);
    assert("temperature" !in payload);
    assert(payload["messages"].array.length == 1);
    assert(payload["messages"][0]["content"].str == "hello");
}

unittest
{
    auto model = new OpenAIModel("gpt-test")
        .maxTokens(64)
        .jsonMode();

    JSONValue payload = model.completionsJSON(JSONValue("hello"));
    assert(payload["max_tokens"].integer == 64);
    assert(payload["response_format"]["type"].str == "json_object");
}

unittest
{
    JSONValue schema = parseJSON(`{"type":"object","properties":{"name":{"type":"string"}}}`);
    auto model = new OpenAIModel("gpt-test").jsonSchema("rubric", schema);
    JSONValue payload = model.completionsJSON(JSONValue("hello"));

    assert(payload["response_format"]["type"].str == "json_schema");
    assert(payload["response_format"]["json_schema"]["name"].str == "rubric");
    assert(payload["response_format"]["json_schema"]["schema"]["type"].str == "object");
    assert(payload["response_format"]["json_schema"]["strict"].type == JSONType.true_);
}

unittest
{
    auto model = new OpenAIModel("gpt-test");
    Completion completion = model.parseCompletions(parseJSON(`{
        "choices": [{
            "message": {
                "content": [
                    {"type":"output_text","text":"Hello"},
                    {"type":"reasoning","text":"Plan"},
                    {"type":"reasoning","summary":[{"type":"summary_text","text":" more"}]}
                ]
            },
            "finish_reason":"stop",
            "logprobs":{"tokens":[]}
        }]
    }`));

    assert(completion.text() == "Hello");
    assert(completion.reasoning() == "Plan more");
    assert(completion.choice().content.type == JSONType.array);
    assert(completion.choice().logProbs.type == JSONType.object);
    assert(completion.choice().finishReason == FinishReason.Stop);
}

unittest
{
    auto model = new OpenAIModel("gpt-test");
    Completion completion = model.parseCompletions(parseJSON(`{
        "choices": [{
            "delta": {
                "content": "Hello"
            },
            "finish_reason":"mystery"
        }]
    }`));

    assert(completion.text() == "Hello");
    assert(completion.choice().finishReason == FinishReason.Unknown);
}

unittest
{
    auto model = new OpenAIModel("gpt-test");
    Completion completion = model.parseCompletions(parseJSON(`{
        "choices": [{
            "message": {
                "refusal": "No."
            }
        }]
    }`));

    assert(completion.choices.length == 1);
    assert(completion.text() is null);
    assert(completion.reasoning() is null);
}
