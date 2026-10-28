module tests.integration.providers.typesafe;

version(integration)
{
    import intuit;
    import tests.integration.common;
    import tests.integration.settings;
    import unit_threaded;

    import std.process : environment;
    import std.stdio : stderr;
    import std.string : strip;

    @Name("TypeSafe endpoint lists unique models and caches configs") @Serial
    unittest
    {
        string[string] env = environment.toAA();
        TestSettings options = TestSettings(env);

        string url = env.get("INTUIT_TEST_TYPESAFE_URL", null);
        if (url.length == 0)
        {
            stderr.writeln("SKIP TypeSafe model listing: missing INTUIT_TEST_TYPESAFE_URL");
            return;
        }

        TypeSafe endpoint = new TypeSafe(
            normalizeBaseUrl(url), env.get("TYPESAFE_API_KEY", null),
        );
        applyTimeouts(endpoint, options);

        ModelConfig[] models;
        sanitized("TypeSafe model listing request", {
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

    @Name("TypeSafe decisions answer all three questions") @Serial
    unittest
    {
        string[string] env = environment.toAA();
        TestSettings options = TestSettings(env);

        string url;
        string model;
        if (!providerScenario(
                env,
                "TypeSafe decisions",
                "INTUIT_TEST_TYPESAFE_URL",
                "INTUIT_TEST_TYPESAFE_DECISION_MODEL",
                url,
                model,
            ))
            return;

        TypeSafe endpoint = new TypeSafe(
            normalizeBaseUrl(url), env.get("TYPESAFE_API_KEY", null),
        );
        applyTimeouts(endpoint, options);

        Decision ret;
        sanitized("TypeSafe decision request", {
            ret = decisions(endpoint, model, decisionState, integrationQuestions());
        });
        validateDecision(ret);
    }
}
