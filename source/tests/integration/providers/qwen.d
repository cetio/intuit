module tests.integration.providers.qwen;

version(integration)
{
    import intuit;
    import tests.integration.common;
    import tests.integration.settings;
    import unit_threaded;

    import std.process : environment;
    import std.stdio : stderr;
    import std.string : strip;

    @Name("Qwen endpoint lists unique models and caches configs") @Serial
    unittest
    {
        string[string] env = environment.toAA();
        TestSettings options = TestSettings(env);

        string url = env.get("INTUIT_TEST_QWEN_URL", null);
        if (url.length == 0)
        {
            stderr.writeln("SKIP Qwen model listing: missing INTUIT_TEST_QWEN_URL");
            return;
        }

        Qwen endpoint = new Qwen(
            normalizeBaseUrl(url), env.get("QWEN_API_KEY", null),
        );
        applyTimeouts(endpoint, options);

        ModelConfig[] models;
        sanitized("Qwen model listing request", {
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

    @Name("Qwen completion maintains user and assistant history") @Serial
    unittest
    {
        string[string] env = environment.toAA();
        TestSettings options = TestSettings(env);

        string url;
        string model;
        if (!providerScenario(
                env,
                "Qwen completion",
                "INTUIT_TEST_QWEN_URL",
                "INTUIT_TEST_QWEN_MODEL",
                url,
                model,
            ))
            return;

        Qwen endpoint = new Qwen(
            normalizeBaseUrl(url), env.get("QWEN_API_KEY", null),
        );
        applyTimeouts(endpoint, options);
        guardDirectProvider(endpoint, model, [Modality.Text], options, env);
        endpoint.config(model).maxTokens = options.maxTokens;

        Context ctx;
        ctx.user(completionPrompt);
        Completion ret;
        sanitized("Qwen completion request", {
            ret = completions(endpoint, model, ctx, 0);
        });
        validateCompletion(ret, ctx);
    }

    @Name("Qwen tool round trip returns a final text answer") @Serial
    unittest
    {
        string[string] env = environment.toAA();
        TestSettings options = TestSettings(env);

        string url;
        string model;
        if (!providerScenario(
                env,
                "Qwen tool round trip",
                "INTUIT_TEST_QWEN_URL",
                "INTUIT_TEST_QWEN_TOOL_MODEL",
                url,
                model,
            ))
            return;

        Qwen endpoint = new Qwen(
            normalizeBaseUrl(url), env.get("QWEN_API_KEY", null),
        );
        applyTimeouts(endpoint, options);
        guardDirectProvider(endpoint, model, [Modality.Text], options, env);

        Context ctx;
        ModelConfig cfg = endpoint.config(model);
        cfg.maxTokens = options.maxTokens;
        toolRoundTrip(ctx, endpoint.tools, cfg, () {
            Completion ret;
            sanitized("Qwen tool completion request", {
                ret = completions(endpoint, model, ctx, 0);
            });
            return ret;
        });
    }

    @Name("Qwen decisions answer all three questions") @Serial
    unittest
    {
        string[string] env = environment.toAA();
        TestSettings options = TestSettings(env);

        string url;
        string model;
        if (!providerScenario(
                env,
                "Qwen decisions",
                "INTUIT_TEST_QWEN_URL",
                "INTUIT_TEST_QWEN_DECISION_MODEL",
                url,
                model,
            ))
            return;

        Qwen endpoint = new Qwen(
            normalizeBaseUrl(url), env.get("QWEN_API_KEY", null),
        );
        applyTimeouts(endpoint, options);
        guardDirectProvider(endpoint, model, [Modality.Decisions], options, env);

        Decision ret;
        sanitized("Qwen decision request", {
            ret = decisions(endpoint, model, decisionState, integrationQuestions());
        });
        validateDecision(ret);
    }

    @Name("Qwen embeddings produce finite matching-dimension vectors") @Serial
    unittest
    {
        string[string] env = environment.toAA();
        TestSettings options = TestSettings(env);

        string url;
        string model;
        if (!providerScenario(
                env,
                "Qwen embeddings",
                "INTUIT_TEST_QWEN_URL",
                "INTUIT_TEST_QWEN_EMBEDDING_MODEL",
                url,
                model,
            ))
            return;

        Qwen endpoint = new Qwen(
            normalizeBaseUrl(url), env.get("QWEN_API_KEY", null),
        );
        applyTimeouts(endpoint, options);
        guardDirectProvider(
            endpoint, model, [Modality.Embedding, Modality.Embeddings], options, env,
        );

        Embedding!float single;
        Embedding!float[] batch;
        sanitized("Qwen embedding request", {
            single = embeddings(endpoint, model, "hello");
        });
        sanitized("Qwen batch embedding request", {
            batch = embeddings(endpoint, model, ["hello", "world"]);
        });
        validateEmbeddings(single, batch);
    }
}
