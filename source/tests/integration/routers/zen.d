module tests.integration.routers.zen;

version(integration)
{
    import intuit;
    import tests.integration.common;
    import tests.integration.settings;
    import unit_threaded;

    import std.process : environment;
    import std.stdio : stderr;
    import std.string : endsWith;

    private enum defaultModel = "nemotron-3.5-lightning-free";

    @Name("OpenCode Zen catalog keys match model details") @Serial
    unittest
    {
        string[string] env = environment.toAA();
        TestSettings options = TestSettings(env);

        Zen router = new Zen(
            options.allowPaid ? env.get("ZEN_API_KEY", null) : null,
            env.get("INTUIT_TEST_ZEN_URL", "https://opencode.ai"),
        );
        applyTimeouts(router, options);

        ModelDetails[string] catalog;
        sanitized("OpenCode Zen catalog request", {
            catalog = router.catalog;
        });
        validateCatalog(catalog);

        string model = env.get("INTUIT_TEST_ZEN_MODEL", defaultModel);
        if (model !in catalog)
            throw new Exception(
                "Selected OpenCode Zen model is absent from the configured catalog: "~model,
            );
        // The catalog has no pricing fields, so free-mode safety is the id
        // policy plus sending no key rather than verified zero prices.
        if (!options.allowPaid && !model.endsWith("-free") && model != "big-pickle")
            throw new Exception(
                "OpenCode Zen free mode permits only *-free model ids or big-pickle, not "~model~".",
            );
    }

    @Name("OpenCode Zen completion maintains user and assistant history") @Serial
    unittest
    {
        string[string] env = environment.toAA();
        TestSettings options = TestSettings(env);

        string url = env.get("INTUIT_TEST_ZEN_URL", "https://opencode.ai");
        string model = env.get("INTUIT_TEST_ZEN_MODEL", defaultModel);
        if (openCodeFreeInferenceRequiresFirstPartyClient(url, model))
        {
            stderr.writeln(
                "SKIP OpenCode Zen completion: free chat models are only available through OpenCode clients",
            );
            return;
        }

        Zen router = new Zen(
            options.allowPaid ? env.get("ZEN_API_KEY", null) : null,
            url,
        );
        applyTimeouts(router, options);

        ModelDetails[string] catalog;
        sanitized("OpenCode Zen catalog request", {
            catalog = router.catalog;
        });
        if (model !in catalog)
            throw new Exception(
                "Selected OpenCode Zen model is absent from the configured catalog: "~model,
            );
        if (!options.allowPaid && !model.endsWith("-free") && model != "big-pickle")
            throw new Exception(
                "OpenCode Zen free mode permits only *-free model ids or big-pickle, not "~model~".",
            );

        router.active(model);
        router.config().maxTokens = options.maxTokens;

        Completion ret;
        sanitized("OpenCode Zen completion request", {
            ret = completions(router, completionPrompt, 0);
        });
        validateCompletion(ret, router.context);
    }

    @Name("OpenCode Zen optional tool round trip returns a final text answer") @Serial
    unittest
    {
        string[string] env = environment.toAA();
        TestSettings options = TestSettings(env);

        string model = env.get("INTUIT_TEST_ZEN_TOOL_MODEL", null);
        if (model.length == 0)
        {
            stderr.writeln("SKIP OpenCode Zen tool round trip: missing INTUIT_TEST_ZEN_TOOL_MODEL");
            return;
        }

        string url = env.get("INTUIT_TEST_ZEN_URL", "https://opencode.ai");
        if (openCodeFreeInferenceRequiresFirstPartyClient(url, model))
        {
            stderr.writeln(
                "SKIP OpenCode Zen tool round trip: free chat models are only available through OpenCode clients",
            );
            return;
        }

        Zen router = new Zen(
            options.allowPaid ? env.get("ZEN_API_KEY", null) : null,
            url,
        );
        applyTimeouts(router, options);

        ModelDetails[string] catalog;
        sanitized("OpenCode Zen catalog request", {
            catalog = router.catalog;
        });
        if (model !in catalog)
            throw new Exception(
                "Selected OpenCode Zen model is absent from the configured catalog: "~model,
            );
        if (!options.allowPaid && !model.endsWith("-free") && model != "big-pickle")
            throw new Exception(
                "OpenCode Zen free mode permits only *-free model ids or big-pickle, not "~model~".",
            );

        router.active(model);
        ModelConfig cfg = router.config();
        cfg.maxTokens = options.maxTokens;
        toolRoundTrip(router.context, router.tools, cfg, () {
            Completion ret;
            sanitized("OpenCode Zen tool completion request", {
                ret = completions(router, 0);
            });
            return ret;
        });
    }
}
