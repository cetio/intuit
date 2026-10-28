module tests.integration.providers.claude;

version(integration)
{
    import intuit;
    import tests.integration.common;
    import tests.integration.settings;
    import unit_threaded;

    import std.process : environment;
    import std.stdio : stderr;
    import std.string : strip;

    @Name("Claude endpoint lists unique models and caches configs") @Serial
    unittest
    {
        string[string] env = environment.toAA();
        TestSettings options = TestSettings(env);

        string url = env.get("INTUIT_TEST_CLAUDE_URL", null);
        if (url.length == 0)
        {
            stderr.writeln("SKIP Claude model listing: missing INTUIT_TEST_CLAUDE_URL");
            return;
        }

        Claude endpoint = new Claude(
            normalizeBaseUrl(url), env.get("ANTHROPIC_API_KEY", null),
        );
        applyTimeouts(endpoint, options);

        ModelConfig[] models;
        sanitized("Claude model listing request", {
            models = endpoint.available();
        });
        models.length.shouldBeGreaterThan(0);

        bool[string] names;
        foreach (ModelConfig cfg; models)
        {
            cfg.name.strip.length.shouldBeGreaterThan(0);
            cfg.name.shouldNotBeIn(names);
            names[cfg.name] = true;
        }
        (models[0] is endpoint.config(models[0].name)).shouldBeTrue;
    }

    @Name("Claude completion maintains user and assistant history") @Serial
    unittest
    {
        string[string] env = environment.toAA();
        TestSettings options = TestSettings(env);

        string url;
        string model;
        if (!providerScenario(
                env,
                "Claude completion",
                "INTUIT_TEST_CLAUDE_URL",
                "INTUIT_TEST_CLAUDE_MODEL",
                url,
                model,
            ))
            return;

        Claude endpoint = new Claude(
            normalizeBaseUrl(url), env.get("ANTHROPIC_API_KEY", null),
        );
        applyTimeouts(endpoint, options);
        endpoint.config(model).maxTokens = options.maxTokens;

        Context ctx;
        ctx.user(completionPrompt);
        Completion ret;
        sanitized("Claude completion request", {
            ret = completions(endpoint, model, ctx, 0);
        });
        validateCompletion(ret, ctx);
    }
}
