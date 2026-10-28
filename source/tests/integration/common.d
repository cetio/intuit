module tests.integration.common;

import intuit;
import tests.integration.settings;
import unit_threaded;

import std.conv : to;
import std.json : JSONType, JSONValue;
import std.math : isFinite;
import std.string : strip;
import core.time : dur;

enum completionPrompt = "Reply with a short greeting.";
enum toolPrompt = "Call add with a=2 and b=3. Use the tool result to answer briefly.";
enum decisionState = "The customer cannot sign in to their account.";

int add(int a, int b)
{
    return a + b;
}

void applyTimeouts(IEndpoint endpoint, TestSettings options)
{
    endpoint.connectTimeout(dur!"seconds"(10));
    endpoint.operationTimeout(dur!"seconds"(options.timeoutSeconds));
}

void applyTimeouts(IRouter router, TestSettings options)
{
    router.connectTimeout(dur!"seconds"(10));
    router.operationTimeout(dur!"seconds"(options.timeoutSeconds));
}

// Endpoint errors carry raw response bodies, so diagnostics stay limited to
// the exception class, HTTP status, and configuration context; no headers,
// keys, or response content is ever reported.
void sanitized(string scenario, void delegate() request)
{
    try
        request();
    catch (EndpointException error)
    {
        throw new Exception(
            scenario~" failed ("~error.classinfo.name~", HTTP status "~error.status.to!string
                ~"). Check the configured URL, credentials, quota, and model.",
        );
    }
    catch (MalformedResponseException error)
    {
        throw new Exception(
            scenario~" failed ("~error.classinfo.name
                ~"). The endpoint returned an unexpected response shape.",
        );
    }
}

void validateCatalog(ModelDetails[string] catalog)
{
    catalog.length.shouldBeGreaterThan(0);
    foreach (string id, ModelDetails details; catalog)
    {
        id.strip.length.shouldBeGreaterThan(0);
        id.should == details.id;
        details.name.strip.length.shouldBeGreaterThan(0);
    }
}

void validateCompletion(Completion ret, ref Context ctx)
{
    ret.choices.length.shouldBeGreaterThan(0);
    ret.choice.text.strip.length.shouldBeGreaterThan(0);
    ret.usage.modelName.strip.length.shouldBeGreaterThan(0);
    ret.usage.latency.isFinite.shouldBeTrue;
    (ret.usage.latency >= 0).shouldBeTrue;
    ctx.length.should == 2;
    ctx.messages[0].role.should == Role.User;
    ctx.messages[1].role.should == Role.Assistant;
}

// The model's actual returned arguments invoke the registered tool; no
// replacement arguments or automatic tool policies are used.
void toolRoundTrip(
    ref Context ctx,
    ref ToolRegistry tools,
    ModelConfig cfg,
    Completion delegate() request,
)
{
    tools.add!add();
    cfg.setRequiredTool(tools.get("add"));

    ctx.user(toolPrompt);
    Completion first = request();

    first.choices.length.shouldBeGreaterThan(0);
    first.choice.toolCalls.length.should == 1;
    ToolCall call = first.choice.toolCalls[0];
    call.name.should == "add";
    call.id.strip.length.shouldBeGreaterThan(0);
    call.arguments.type.should == JSONType.object;
    call.arguments["a"].integer.should == 2;
    call.arguments["b"].integer.should == 3;

    JSONValue result = tools.get(call.name).impl(call.arguments);
    result.integer.should == 5;

    ctx.tool(call.id, result);
    cfg.setToolLiability("none");
    Completion second = request();

    second.choices.length.shouldBeGreaterThan(0);
    second.choice.text.strip.length.shouldBeGreaterThan(0);
    second.choice.toolCalls.length.should == 0;

    ctx.length.should == 4;
    ctx.messages[1].role.should == Role.Assistant;
    ctx.messages[2].role.should == Role.Tool;
    ctx.messages[3].role.should == Role.Assistant;
    JSONValue history = ctx.toJSON();
    history[1]["tool_calls"][0]["id"].str.should == call.id;
    history[2]["tool_call_id"].str.should == call.id;
}

// Applies the hosted free-price guard when an endpoint URL is the canonical
// OpenRouter API route: free mode verifies zero pricing in the real catalog
// and pins provider.max_price to zero for every request the config builds.
// Any other URL cannot be cost-verified and is the operator's responsibility.
void guardDirectProvider(
    IEndpoint endpoint,
    string model,
    Modality[] outputs,
    TestSettings options,
    string[string] env,
)
{
    if (normalizeBaseUrl(endpoint.url) != "https://openrouter.ai/api")
        return;

    OpenRouter catalogRouter = new OpenRouter(env.get("OPENROUTER_API_KEY", null));
    applyTimeouts(catalogRouter, options);
    ModelDetails[string] catalog;
    sanitized("OpenRouter safety catalog request", {
        catalog = catalogRouter.catalog;
    });
    requireModel(catalog, model, outputs, options.allowPaid);
    if (!options.allowPaid)
    {
        ModelConfig cfg = endpoint.config(model);
        if (cfg.params.type != JSONType.object)
            cfg.params = JSONValue.emptyObject;
        cfg.params["provider"] = zeroPriceCap();
    }
}

DecisionQuestion[] integrationQuestions()
{
    DecisionQuestion predicate;
    predicate.type = DecisionType.Predicate;
    predicate.name = "login_issue";
    predicate.instructions = "Does the customer report a sign-in problem?";

    DecisionQuestion choice;
    choice.type = DecisionType.Choice;
    choice.name = "team";
    choice.instructions = "Which team should handle this ticket?";
    choice.choices = [
        DecisionOption(JSONValue("account"), "Sign-in and account access"),
        DecisionOption(JSONValue("billing"), "Invoices and payments"),
    ];

    DecisionQuestion score;
    score.type = DecisionType.Score;
    score.name = "severity";
    score.instructions = "How severe is this ticket?";
    score.levels = [
        DecisionLevel("Low", "Minor inconvenience"),
        DecisionLevel("High", "Cannot use the service"),
    ];
    return [predicate, choice, score];
}

void validateDecision(Decision ret)
{
    ret.answers.length.should == 3;

    ret.answers[0].name.should == "login_issue";
    ret.answers[0].type.should == DecisionType.Predicate;
    ret.answers[0].probability.isFinite.shouldBeTrue;
    (ret.answers[0].probability >= 0 && ret.answers[0].probability <= 1).shouldBeTrue;

    ret.answers[1].name.should == "team";
    ret.answers[1].type.should == DecisionType.Choice;
    ret.answers[1].choice.shouldBeIn([JSONValue("account"), JSONValue("billing")]);

    ret.answers[2].name.should == "severity";
    ret.answers[2].type.should == DecisionType.Score;
    ret.answers[2].score.isFinite.shouldBeTrue;
    (ret.answers[2].score >= 0 && ret.answers[2].score <= 1).shouldBeTrue;

    ret.raw.type.should == JSONType.object;
    ret.usage.modelName.strip.length.shouldBeGreaterThan(0);
}

void validatePredicateDecision(Decision ret)
{
    ret.answers.length.should == 1;
    ret.answers[0].name.should == "login_issue";
    ret.answers[0].type.should == DecisionType.Predicate;
    ret.answers[0].probability.isFinite.shouldBeTrue;
    (ret.answers[0].probability >= 0 && ret.answers[0].probability <= 1).shouldBeTrue;
    ret.raw.type.should == JSONType.object;
    ret.usage.modelName.strip.length.shouldBeGreaterThan(0);
}

void validateEmbeddings(Embedding!float single, Embedding!float[] batch)
{
    single.value.length.shouldBeGreaterThan(0);
    batch.length.should == 2;
    batch[0].value.length.should == single.value.length;
    batch[1].value.length.should == single.value.length;
    foreach (float element; single.value)
        element.isFinite.shouldBeTrue;
    foreach (Embedding!float vector; batch)
        foreach (float element; vector.value)
            element.isFinite.shouldBeTrue;
}
