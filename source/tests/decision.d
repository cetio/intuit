module tests.decision;

import intuit;
import intuit.provider.systemone : SystemOneModelConfig;
import unit_threaded;

import std.conv : to;
import std.json : JSONValue, JSONType, parseJSON;
import std.math : isNaN;

@Name("Decision requests use OpenAI questions without chat parameters")
unittest
{
    ModelConfig cfg = new ModelConfig("gpt-6-luna");
    cfg.temperature = 0.5;
    cfg.maxTokens = 50;
    DecisionQuestion question;
    question.name = "urgent";
    question.instructions = "Is this urgent?";

    JSONValue payload = cfg.buildDecisionsPayload(JSONValue("Help now"), [question]);
    payload["model"].str.should == "gpt-6-luna";
    payload["questions"][0]["type"].str.should == "predicate";
    payload["questions"][0]["name"].str.should == "urgent";
    assert("messages" !in payload);
    assert("temperature" !in payload);
    assert("max_tokens" !in payload);
    cfg.buildDecisionsPayload(JSONValue("text"), []).shouldThrow!FormatException;
    cfg.buildDecisionsPayload(JSONValue("text"), [question, question]).shouldThrow!FormatException;
}

@Name("Decision parses OpenAI typed values refusals distributions and usage")
unittest
{
    ModelConfig cfg = new ModelConfig("gpt-6-luna");
    JSONValue json = parseJSON(`{
        "model": "resolved", "latency": 12.5,
        "answers": [
            {"type": "predicate", "name": "urgent", "probability": 0},
            {"type": "choice", "choice": true, "confidence": 0.8,
             "probabilities": [{"value": true, "probability": 0.9}, {"value": "true", "probability": 0.1}]},
            {"type": "score", "score": 0.25, "confidence": 0.5,
             "probabilities": [{"value": 0, "label": "Low", "probability": 0.75},
                               {"value": 1, "label": "High", "probability": 0.25}]},
            {"type": "refusal", "name": "blocked"}
        ],
        "usage": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25,
                  "input_tokens_details": {"cached_tokens": 3}}
    }`);
    Decision ret = cfg.parseDecisionsResponse(json);
    ret.answers.length.should == 4;
    ret.answer("urgent").probability.should == 0;
    ret.answer(1).choice.type.should == JSONType.true_;
    ret.answer(1).probabilities[1].value.type.should == JSONType.string;
    ret.answer(2).score.should == 0.25;
    ret.answer("blocked").type.should == DecisionType.Refusal;
    assert(ret.answer("blocked").probability.isNaN);
    ret.usage.modelName.should == "resolved";
    ret.usage.promptTokens.should == 20;
    ret.usage.completionTokens.should == 5;
    ret.usage.cacheHits.should == 3;
    ret.usage.cacheMisses.should == 17;
    ret.usage.latency.should == 12.5f;
    ret.raw.should == json;
    ret.answer("missing").shouldThrow!Exception;
    ret.answer(4).shouldThrow!Exception;
    cfg.parseDecisionsResponse(parseJSON(`{"answers": [{}]}`)).shouldThrow!MalformedResponseException;
    cfg.parseDecisionsResponse(
        parseJSON(`{"answers": [{"type": "predicate", "probability": 2}]}`),
    ).shouldThrow!MalformedResponseException;
}

@Name("System One converts all question types and restores OpenAI answer semantics")
unittest
{
    SystemOneModelConfig cfg = new SystemOneModelConfig("jev-latest");
    DecisionQuestion predicate;
    predicate.instructions = "Urgent?";
    DecisionQuestion choice;
    choice.type = DecisionType.Choice;
    choice.name = "route";
    choice.instructions = "Choose a route";
    choice.choices = [DecisionOption(JSONValue("billing"), "Payments"),
        DecisionOption(JSONValue("support"), "Technical help")];
    DecisionQuestion score;
    score.type = DecisionType.Score;
    score.name = "severity";
    score.instructions = "How severe?";
    score.levels = [DecisionLevel("Low", "Minor"), DecisionLevel("High", "Blocking")];
    DecisionQuestion[] questions = [predicate, choice, score];
    JSONValue payload = cfg.buildDecisionsPayload(parseJSON(`{"ticket": "Help"}`), questions);
    payload["state"]["ticket"].str.should == "Help";
    payload["questions"]["route"]["criteria"]["billing"].str.should == "Payments";
    payload["questions"]["severity"]["criteria"][0].str.should == "Low: Minor";
    assert("input" !in payload);
    string predicateName;
    foreach (string name, JSONValue question; payload["questions"].object)
    {
        if (question["type"].str == "noul")
            predicateName = name;
    }

    JSONValue json = parseJSON(`{
        "model": "jev-1.13.0", "answers": {
            "route": {"type": "choice", "choice": "support",
                      "probabilities": {"billing": 0.1, "support": 0.9}},
            "severity": {"type": "score", "score": 0.75, "confidence": 0.5,
                         "probabilities": {"1": 0.75, "0": 0.25},
                         "legend": {"0": "Low: Minor", "1": "High: Blocking"}}
        }, "usage": {"input_tokens": 12, "output_tokens": 4, "cost": 0.001}
    }`);
    json["answers"][predicateName] = parseJSON(`{"type": "noul", "noul": 0.9}`);
    Decision ret = cfg.parseDecisionsResponse(json, questions);
    ret.answer.probability.should == 0.9;
    ret.answer.name.should == "";
    ret.answer("route").choice.str.should == "support";
    assert(ret.answer("route").confidence.isNaN);
    ret.answer("severity").probabilities[0].value.integer.should == 0;
    ret.answer("severity").probabilities[0].label.should == "Low";
    ret.usage.totalTokens.should == 16;
    ret.raw.should == json;
}

@Name("System One preserves boolean choices distinct from string choices")
unittest
{
    SystemOneModelConfig cfg = new SystemOneModelConfig("jev-latest");
    DecisionQuestion question;
    question.name = "choice";
    question.type = DecisionType.Choice;
    question.instructions = "Pick a value";
    question.choices = [DecisionOption(JSONValue(true)), DecisionOption(JSONValue("true"))];
    JSONValue payload = cfg.buildDecisionsPayload(JSONValue("test"), [question]);
    payload["questions"]["choice"]["criteria"].object.length.should == 2;
    JSONValue json = parseJSON(`{"answers": {"choice": {
        "type": "choice", "choice": "true", "probabilities": {"true": 0.9, "\"true\"": 0.1}
    }}}`);
    Decision ret = cfg.parseDecisionsResponse(json, [question]);
    ret.answer.choice.type.should == JSONType.true_;
    ret.answer.probabilities[1].value.type.should == JSONType.string;
}

@Name("System One enforces shared limits and TypeSafe reuses its configuration")
unittest
{
    TypeSafe ep = new TypeSafe();
    assert(cast(SystemOneModelConfig)ep.config("jev-latest") !is null);
    assert(ep.config("jev-latest") is ep.config("jev-latest"));
    SystemOneModelConfig cfg = new SystemOneModelConfig("jev-latest");
    DecisionQuestion question;
    question.instructions = "Urgent?";
    DecisionQuestion[] questions = new DecisionQuestion[255];
    questions[] = question;
    cfg.buildDecisionsPayload(JSONValue("text"), questions)["questions"].object.length.should == 255;
    cfg.buildDecisionsPayload(JSONValue("text"), questions~question).shouldThrow!FormatException;

    question.type = DecisionType.Score;
    question.levels = new DecisionLevel[10];
    question.levels[] = DecisionLevel("Level");
    cfg.buildDecisionsPayload(JSONValue("text"), [question]);
    question.levels ~= DecisionLevel("Extra");
    cfg.buildDecisionsPayload(JSONValue("text"), [question]).shouldThrow!FormatException;

    question.type = DecisionType.Choice;
    question.levels = null;
    question.choices = new DecisionOption[255];
    foreach (i; 0..question.choices.length)
        question.choices[i] = DecisionOption(JSONValue(i.to!string));

    cfg.buildDecisionsPayload(JSONValue("text"), [question]);
    question.choices ~= DecisionOption(JSONValue("extra"));
    cfg.buildDecisionsPayload(JSONValue("text"), [question]).shouldThrow!FormatException;
}

@Name("Decision validation rejects mismatched answers without changing raw responses")
unittest
{
    ModelConfig cfg = new ModelConfig("gpt-6-luna");
    DecisionQuestion question;
    question.name = "urgent";
    question.instructions = "Urgent?";
    cfg.parseDecisionsResponse(parseJSON(`{"answers": []}`), [question])
        .shouldThrow!MalformedResponseException;
    cfg.parseDecisionsResponse(parseJSON(`{"answers": []}`))
        .shouldThrow!MalformedResponseException;
    cfg.parseDecisionsResponse(
        parseJSON(`{"answers": [{"type": "predicate", "name": "other", "probability": 0.5}]}`),
        [question],
    ).shouldThrow!MalformedResponseException;
    cfg.parseDecisionsResponse(parseJSON(`{"error": {"message": "failed"}}`))
        .shouldThrow!EndpointException;

    SystemOneModelConfig systemOne = new SystemOneModelConfig("jev-latest");
    JSONValue json = parseJSON(`{"answers": {"urgent": {"type": "noul", "noul": 0.5}}}`);
    Decision ret = systemOne.parseDecisionsResponse(json, [question]);
    ret.answer.raw["type"].str.should == "noul";
    json["answers"]["urgent"]["type"].str.should == "noul";
    systemOne.parseDecisionsResponse(parseJSON(`{"answers": {}}`), [question])
        .shouldThrow!MalformedResponseException;
    (new TypeSafe()._completions(cfg, JSONValue.init)).shouldThrow!EndpointException;
    (new TypeSafe()._embeddings(cfg, JSONValue.init)).shouldThrow!EndpointException;
    (new Claude("http://localhost")._decisions(cfg, JSONValue.init)).shouldThrow!EndpointException;
    (new LiteLLM()._decisions(JSONValue.init)).shouldThrow!Exception;
}

