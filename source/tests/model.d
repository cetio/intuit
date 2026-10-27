module tests.model;

import intuit;
import intuit.provider.claude.model : ClaudeModelConfig;
import intuit.provider.openai : OpenAI;
import intuit.provider.qwen.model : QwenModelConfig;
import intuit.provider.systemone : SystemOneModelConfig;
import unit_threaded;

import std.json : JSONValue, JSONType, parseJSON;
import std.math : isNaN;

private class EmbeddingTestEndpoint : OpenAI
{
    JSONValue response;

    this(JSONValue response)
    {
        super("http://localhost");
        this.response = response;
    }

    override JSONValue _embeddings(ModelConfig cfg, JSONValue payload)
        => response;
}

private class DispatchModelConfig : ModelConfig
{
    this()
    {
        super("dispatch");
    }

    override JSONValue buildCompletionPayload(JSONValue input, ToolRegistry tools = ToolRegistry.init)
        => JSONValue("completion");

    override JSONValue buildEmbeddingPayload(JSONValue input)
        => JSONValue("embedding");

    override JSONValue buildDecisionPayload(JSONValue input, DecisionQuestion[] questions)
        => JSONValue("decision");

    override Completion parseCompletionResponse(JSONValue json)
    {
        Completion ret;
        ret.usage.modelName = "completion";
        return ret;
    }

    override JSONValue parseEmbeddingResponse(JSONValue json)
        => JSONValue("embedding");

    override Decision parseDecisionResponse(JSONValue json, DecisionQuestion[] questions = null)
    {
        Decision ret;
        ret.usage.modelName = "decision";
        return ret;
    }
}

@Name("deprecated ModelConfig aliases forward through the singular API")
unittest
{
    ModelConfig cfg = new ModelConfig("test");
    JSONValue completionInput = JSONValue("hello");
    cfg.buildPayload(completionInput).should == cfg.buildCompletionPayload(completionInput);
    cfg.buildEmbeddingsPayload(completionInput).should == cfg.buildEmbeddingPayload(completionInput);

    DecisionQuestion question;
    question.instructions = "Is this urgent?";
    cfg.buildDecisionsPayload(completionInput, [question]).should
        == cfg.buildDecisionPayload(completionInput, [question]);

    JSONValue completionResponse = parseJSON(
        `{"choices":[{"message":{"content":"hello"},"finish_reason":"stop"}]}`,
    );
    cfg.parseResponse(completionResponse).text.should
        == cfg.parseCompletionResponse(completionResponse).text;

    JSONValue embeddingResponse = parseJSON(`{"data":[{"embedding":[1,2]}]}`);
    cfg.parseEmbeddingsResponse(embeddingResponse).should
        == cfg.parseEmbeddingResponse(embeddingResponse);

    JSONValue decisionResponse = parseJSON(
        `{"answers":[{"type":"predicate","probability":0.5}]}`,
    );
    cfg.parseDecisionsResponse(decisionResponse).answer.probability.should
        == cfg.parseDecisionResponse(decisionResponse).answer.probability;
}

@Name("deprecated aliases dispatch to Claude Qwen and System One overrides through the base type")
unittest
{
    ClaudeModelConfig claudeConfig = new ClaudeModelConfig("claude-test");
    claudeConfig.system = "Claude system";
    ModelConfig claude = claudeConfig;
    JSONValue claudePayload = claude.buildPayload(JSONValue("hello"));
    claudePayload["system"].str.should == "Claude system";
    claude.parseResponse(parseJSON(
        `{"content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn"}`,
    )).text.should == "hello";
    claude.buildEmbeddingsPayload(JSONValue("hello")).shouldThrow!EndpointException;
    claude.parseEmbeddingsResponse(JSONValue("response")).shouldThrow!EndpointException;

    QwenModelConfig qwenConfig = new QwenModelConfig("qwen-test");
    qwenConfig.topK = 40;
    qwenConfig.encodingFormat = "base64";
    qwenConfig.dimensions = 64;
    ModelConfig qwen = qwenConfig;
    qwen.buildPayload(JSONValue("hello"))["top_k"].integer.should == 40;
    qwen.buildEmbeddingsPayload(JSONValue("hello"))["encoding_format"].str.should == "base64";
    JSONValue qwenResponse = parseJSON(
        `{"choices":[{"message":{"content":"<tool_call>{\"name\":\"search\",\"arguments\":{}}</tool_call>"}}]}`,
    );
    qwen.parseResponse(qwenResponse).choice.toolCalls[0].name.should == "search";
    qwen.parseEmbeddingsResponse(parseJSON(`{"data":[{"embedding":[1]}]}`))[0][0].integer.should == 1;

    SystemOneModelConfig systemOneConfig = new SystemOneModelConfig("jev-latest");
    ModelConfig systemOne = systemOneConfig;
    DecisionQuestion question;
    question.instructions = "Is this urgent?";
    systemOne.buildDecisionsPayload(JSONValue("state"), [question])["state"].str.should == "state";
    Decision decision = systemOne.parseDecisionsResponse(
        parseJSON(`{"answers":{"__intuit_question_0":{"type":"noul","noul":0.75}}}`),
        [question],
    );
    decision.answer.probability.should == 0.75;
}

@Name("singular ModelConfig methods remain virtual through base references")
unittest
{
    ModelConfig cfg = new DispatchModelConfig();
    cfg.buildCompletionPayload(JSONValue.init).str.should == "completion";
    cfg.buildEmbeddingPayload(JSONValue.init).str.should == "embedding";
    cfg.buildDecisionPayload(JSONValue.init, null).str.should == "decision";
    cfg.parseCompletionResponse(JSONValue.init).usage.modelName.should == "completion";
    cfg.parseEmbeddingResponse(JSONValue.init).str.should == "embedding";
    cfg.parseDecisionResponse(JSONValue.init).usage.modelName.should == "decision";
}

@Name("completion parsing keeps distinct and unknown wire finish reasons")
unittest
{
    ModelConfig cfg = new ModelConfig("test");
    JSONValue response = parseJSON(`{"choices":[
        {"finish_reason":"tool_call"},
        {"finish_reason":"tool_calls"},
        {"finish_reason":"future_reason"},
        {"finish_reason":null},
        {}
    ]}`);
    Completion completion = cfg.parseCompletionResponse(response);
    completion.choices[0].finishReason.should == FinishReason.ToolCall;
    completion.choices[1].finishReason.should == FinishReason.ToolCalls;
    assert(cast(string)completion.choices[2].finishReason == "future_reason");
    completion.choices[3].finishReason.should == FinishReason.Unknown;
    completion.choices[4].finishReason.should == FinishReason.Unknown;
}

@Name("completion parsing recursively extracts content reasoning tool calls and fallback usage")
unittest
{
    ModelConfig cfg = new ModelConfig("requested");
    JSONValue response = parseJSON(`{
        "usage": {
            "input_tokens": 20,
            "output_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 4}
        },
        "choices": [{
            "message": {
                "content": [
                    {"type": "text", "text": "Answer."},
                    {"type": "reasoning", "content": [{"text": "First reason."}]}
                ],
                "reasoning": {"summary": "Second reason."},
                "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": "lookup", "arguments": "{\"key\":\"value\"}"}
                }]
            },
            "finish_reason": "tool_call"
        }]
    }`);
    Completion completion = cfg.parseCompletionResponse(response);
    completion.usage.modelName.should == "requested";
    completion.usage.promptTokens.should == 20;
    completion.usage.completionTokens.should == 5;
    completion.usage.totalTokens.should == 25;
    completion.usage.cacheHits.should == 4;
    completion.usage.cacheMisses.should == 16;
    completion.text.should == "Answer.";
    completion.reasoning.should == "First reason.Second reason.";
    completion.choice.toolCalls[0].id.should == "call_1";
    completion.choice.toolCalls[0].name.should == "lookup";
    completion.choice.toolCalls[0].arguments["key"].str.should == "value";
}

@Name("embedding conversion preserves single and batch vector shapes and numeric casts")
unittest
{
    EmbeddingTestEndpoint endpoint = new EmbeddingTestEndpoint(parseJSON(
        `{"data":[{"embedding":[0.1,2,null,"ignored",true,4.75]},{"embedding":[3.5]},{"embedding":"invalid"}]}`,
    ));

    Embedding!float single = embeddings!float(endpoint, "embedder", "one");
    single.value.length.should == 6;
    single.value[0].should == 0.1f;
    single.value[1].should == 2f;
    single.value[2].isNaN.should == true;
    single.value[3].isNaN.should == true;
    single.value[4].isNaN.should == true;
    single.value[5].should == 4.75f;

    Embedding!double doubles = embeddings!double(endpoint, "embedder", "one");
    doubles.value[0].should == 0.1;

    Embedding!int integers = embeddings!int(endpoint, "embedder", "one");
    integers.value.should == [0, 2, 0, 0, 0, 4];

    Embedding!float[] batch = embeddings!float(endpoint, "embedder", ["one", "two", "three"]);
    batch.length.should == 3;
    batch[0].value.length.should == 6;
    batch[0].value[0].should == single.value[0];
    batch[0].value[1].should == single.value[1];
    batch[0].value[2].isNaN.should == true;
    batch[1].value.should == [3.5f];
    batch[2].value.length.should == 0;
}

@Name("embedding parsing handles non-array responses and endpoint errors")
unittest
{
    EmbeddingTestEndpoint endpoint = new EmbeddingTestEndpoint(
        parseJSON(`{"data":[{"embedding":"not an array"}]}`),
    );
    embeddings(endpoint, "embedder", "one").value.length.should == 0;

    endpoint.response = parseJSON(`{"error":{"message":"embedding failed"}}`);
    embeddings(endpoint, "embedder", "one").shouldThrow!EndpointException;
}
