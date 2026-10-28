module tests.integration.config;

import intuit;
import tests.integration.settings;
import unit_threaded;

import std.json : JSONValue;

@Name("Integration settings default to free inference and bounded requests")
unittest
{
    TestSettings options = TestSettings(null);
    options.allowPaid.shouldBeFalse;
    options.maxTokens.should == 256;
    options.timeoutSeconds.should == 90;
}

@Name("Paid inference requires the explicit one setting")
unittest
{
    string[string] env = ["INTUIT_TEST_ALLOW_PAID": "1"];
    TestSettings(env).allowPaid.shouldBeTrue;
    env["INTUIT_TEST_ALLOW_PAID"] = "0";
    TestSettings(env).allowPaid.shouldBeFalse;
    env["INTUIT_TEST_ALLOW_PAID"] = "true";
    TestSettings(env).shouldThrow!Exception;
}

@Name("Positive integer settings reject zero negative and non-numeric values")
unittest
{
    string[string] env = ["INTUIT_TEST_MAX_TOKENS": "64"];
    TestSettings(env).maxTokens.should == 64;
    env["INTUIT_TEST_MAX_TOKENS"] = "0";
    TestSettings(env).shouldThrow!Exception;
    env["INTUIT_TEST_MAX_TOKENS"] = "-5";
    TestSettings(env).shouldThrow!Exception;
    env["INTUIT_TEST_MAX_TOKENS"] = "abc";
    TestSettings(env).shouldThrow!Exception;
    env.remove("INTUIT_TEST_MAX_TOKENS");
    TestSettings(env).timeoutSeconds.should == 90;
}

@Name("Provider scenarios skip absent config and reject partial config")
unittest
{
    string url;
    string model;
    string[string] env;
    providerScenario(
        env, "sample", "INTUIT_TEST_SAMPLE_URL", "INTUIT_TEST_SAMPLE_MODEL", url, model,
    ).shouldBeFalse;

    env["INTUIT_TEST_SAMPLE_MODEL"] = "selected";
    providerScenario(
        env, "sample", "INTUIT_TEST_SAMPLE_URL", "INTUIT_TEST_SAMPLE_MODEL", url, model,
    ).shouldThrow!Exception;

    env["INTUIT_TEST_SAMPLE_URL"] = "http://127.0.0.1:1234";
    env.remove("INTUIT_TEST_SAMPLE_MODEL");
    providerScenario(
        env, "sample", "INTUIT_TEST_SAMPLE_URL", "INTUIT_TEST_SAMPLE_MODEL", url, model,
    ).shouldBeFalse;

    env["INTUIT_TEST_SAMPLE_MODEL"] = "selected";
    providerScenario(
        env, "sample", "INTUIT_TEST_SAMPLE_URL", "INTUIT_TEST_SAMPLE_MODEL", url, model,
    ).shouldBeTrue;
    url.should == "http://127.0.0.1:1234";
    model.should == "selected";
}

@Name("Base URLs lose trailing slashes before endpoint construction")
unittest
{
    normalizeBaseUrl("http://127.0.0.1:1234/").should == "http://127.0.0.1:1234";
    normalizeBaseUrl("https://openrouter.ai/api//").should == "https://openrouter.ai/api";
    normalizeBaseUrl("https://models.dev").should == "https://models.dev";
}

@Name("OpenCode free inference restriction only matches its hosted free models")
unittest
{
    openCodeFreeInferenceRequiresFirstPartyClient(
        "https://opencode.ai/", "deepseek-v4.1-flash-free",
    ).shouldBeTrue;
    openCodeFreeInferenceRequiresFirstPartyClient(
        "https://opencode.ai", "big-pickle",
    ).shouldBeTrue;
    openCodeFreeInferenceRequiresFirstPartyClient(
        "https://opencode.ai", "deepseek-v4.1-flash",
    ).shouldBeFalse;
    openCodeFreeInferenceRequiresFirstPartyClient(
        "http://127.0.0.1:1234", "deepseek-v4.1-flash-free",
    ).shouldBeFalse;
}

@Name("The zero-price provider cap reaches actual request payloads")
unittest
{
    JSONValue cap = zeroPriceCap();
    cap["max_price"]["prompt"].integer.should == 0;
    cap["max_price"]["completion"].integer.should == 0;

    ModelConfig cfg = new ModelConfig("model");
    cfg.params = JSONValue.emptyObject;
    cfg.params["provider"] = cap;
    JSONValue payload = cfg.buildCompletionPayload(JSONValue("input"));
    payload["provider"]["max_price"]["prompt"].integer.should == 0;
    payload["provider"]["max_price"]["completion"].integer.should == 0;
}

private ModelDetails freeDetails(string id = "sample/free")
{
    ModelDetails details;
    details.id = id;
    details.name = "Sample";
    details.inputModalities = [Modality.Text];
    details.outputModalities = [Modality.Text];
    details.promptCost = 0;
    details.completionCost = 0;
    return details;
}

@Name("Model enforcement requires catalog identity modality and zero pricing")
unittest
{
    ModelDetails[string] catalog = ["sample/free": freeDetails()];
    requireModel(catalog, "sample/free", [Modality.Text]).id.should == "sample/free";
    requireModel(catalog, "sample/free", [Modality.Text], true).id.should == "sample/free";

    requireModel(catalog, "other/model", [Modality.Text]).shouldThrow!Exception;
    requireModel(catalog, "sample/free", [Modality.Embedding]).shouldThrow!Exception;

    ModelDetails[string] mismatched = ["sample/free": freeDetails("sample/different")];
    requireModel(mismatched, "sample/free", [Modality.Text]).shouldThrow!Exception;

    ModelDetails embedding = freeDetails("sample/embed");
    embedding.outputModalities = [Modality.Embeddings];
    catalog["sample/embed"] = embedding;
    requireModel(
        catalog, "sample/embed", [Modality.Embedding, Modality.Embeddings],
    ).id.should == "sample/embed";
}

@Name("Free mode rejects nonzero and unknown prices regardless of input mode")
unittest
{
    ModelDetails[string] catalog;

    ModelDetails paidInput = freeDetails("sample/paid-input");
    paidInput.promptCost = 0.02;
    catalog["sample/paid-input"] = paidInput;
    requireModel(catalog, "sample/paid-input", [Modality.Text]).shouldThrow!Exception;
    requireModel(catalog, "sample/paid-input", [Modality.Text], true).id
        .should == "sample/paid-input";

    ModelDetails paidOutput = freeDetails("sample/paid-output");
    paidOutput.completionCost = 0.02;
    catalog["sample/paid-output"] = paidOutput;
    requireModel(catalog, "sample/paid-output", [Modality.Text]).shouldThrow!Exception;

    ModelDetails unknownInput = freeDetails("sample/unknown-input");
    unknownInput.promptCost = double.nan;
    catalog["sample/unknown-input"] = unknownInput;
    requireModel(catalog, "sample/unknown-input", [Modality.Text]).shouldThrow!Exception;

    ModelDetails unknownOutput = freeDetails("sample/unknown-output");
    unknownOutput.completionCost = double.nan;
    catalog["sample/unknown-output"] = unknownOutput;
    requireModel(catalog, "sample/unknown-output", [Modality.Text]).shouldThrow!Exception;

    ModelDetails noText = freeDetails("sample/no-text");
    noText.inputModalities = [Modality.Image];
    catalog["sample/no-text"] = noText;
    requireModel(catalog, "sample/no-text", [Modality.Text]).shouldThrow!Exception;
}
