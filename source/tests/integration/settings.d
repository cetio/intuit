module tests.integration.settings;

import intuit.router.details : ModelDetails, Modality;

import std.algorithm.searching : canFind;
import std.conv : to;
import std.json : JSONValue;
import std.math : isFinite;
import std.stdio : stderr;
import std.string : endsWith;

struct TestSettings
{
    bool allowPaid;
    int maxTokens;
    int timeoutSeconds;

    this(string[string] env)
    {
        string* paidFlag = "INTUIT_TEST_ALLOW_PAID" in env;
        if (paidFlag is null || paidFlag.length == 0 || *paidFlag == "0")
            allowPaid = false;
        else if (*paidFlag == "1")
            allowPaid = true;
        else
            throw new Exception("INTUIT_TEST_ALLOW_PAID must be \"0\" or \"1\".");

        maxTokens = positiveInt(env, "INTUIT_TEST_MAX_TOKENS", 256);
        timeoutSeconds = positiveInt(env, "INTUIT_TEST_TIMEOUT_SECONDS", 90);
    }
}

private int positiveInt(string[string] env, string name, int fallback)
{
    string* configured = name in env;
    if (configured is null || configured.length == 0)
        return fallback;

    int parsed;
    try
        parsed = (*configured).to!int;
    catch (Exception)
        throw new Exception(name~" must be a positive integer.");
    if (parsed <= 0)
        throw new Exception(name~" must be a positive integer.");
    return parsed;
}

bool providerScenario(
    string[string] env,
    string label,
    string urlName,
    string modelName,
    out string url,
    out string model,
)
{
    url = env.get(urlName, null);
    model = env.get(modelName, null);
    if (url.length == 0 && model.length == 0)
    {
        stderr.writeln("SKIP ", label, ": missing ", urlName);
        return false;
    }
    if (url.length == 0)
        throw new Exception(
            label~" requested with "~modelName~" but is missing "~urlName~".",
        );
    if (model.length == 0)
    {
        stderr.writeln("SKIP ", label, ": missing ", modelName);
        return false;
    }
    return true;
}

string normalizeBaseUrl(string url)
{
    while (url.length > 0 && url[$ - 1] == '/')
        url = url[0 .. $ - 1];
    return url;
}

bool openCodeFreeInferenceRequiresFirstPartyClient(string url, string model)
{
    if (normalizeBaseUrl(url) != "https://opencode.ai")
        return false;
    return model.endsWith("-free") || model == "big-pickle";
}

JSONValue zeroPriceCap()
{
    JSONValue maxPrice = JSONValue.emptyObject;
    maxPrice["prompt"] = JSONValue(0);
    maxPrice["completion"] = JSONValue(0);

    JSONValue provider = JSONValue.emptyObject;
    provider["max_price"] = maxPrice;
    return provider;
}

ModelDetails requireModel(
    ModelDetails[string] catalog,
    string model,
    Modality[] outputs,
    bool allowPaid = false,
)
{
    ModelDetails* details = model in catalog;
    if (details is null || details.id != model)
        throw new Exception("Model is absent from the OpenRouter catalog: "~model);
    if (!details.inputModalities.canFind(Modality.Text))
        throw new Exception("Catalog model lacks text input: "~model);

    bool hasOutput;
    foreach (Modality output; outputs)
        hasOutput = hasOutput || details.outputModalities.canFind(output);
    if (!hasOutput)
        throw new Exception("Catalog model lacks the required output modality: "~model);

    if (!allowPaid
        && (!details.promptCost.isFinite || !details.completionCost.isFinite
            || details.promptCost != 0 || details.completionCost != 0))
    {
        throw new Exception("Catalog does not confirm zero-cost inference for "~model~".");
    }
    return *details;
}
