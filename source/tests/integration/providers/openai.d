module tests.integration.providers.openai;

version(integration)
{
    import intuit;
    import tests.integration.common;
    import tests.integration.settings;
    import unit_threaded;

    import std.process : environment;
    import std.stdio : stderr;
    import std.string : strip;

    @Name("OpenAI endpoint lists unique models and caches configs") @Serial
    unittest
    {
        string[string] env = environment.toAA();
        TestSettings options = TestSettings(env);

        string url = env.get("INTUIT_TEST_OPENAI_URL", null);
        if (url.length == 0)
        {
            stderr.writeln("SKIP OpenAI model listing: missing INTUIT_TEST_OPENAI_URL");
            return;
        }

        OpenAI endpoint = new OpenAI(
            normalizeBaseUrl(url), env.get("OPENAI_API_KEY", null),
        );
        applyTimeouts(endpoint, options);

        ModelConfig[] models;
        sanitized("OpenAI model listing request", {
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

    @Name("OpenAI completion maintains user and assistant history") @Serial
    unittest
    {
        string[string] env = environment.toAA();
        TestSettings options = TestSettings(env);

        string url;
        string model;
        if (!providerScenario(
                env,
                "OpenAI completion",
                "INTUIT_TEST_OPENAI_URL",
                "INTUIT_TEST_OPENAI_MODEL",
                url,
                model,
            ))
            return;

        OpenAI endpoint = new OpenAI(
            normalizeBaseUrl(url), env.get("OPENAI_API_KEY", null),
        );
        applyTimeouts(endpoint, options);
        guardDirectProvider(endpoint, model, [Modality.Text], options, env);
        endpoint.config(model).maxTokens = options.maxTokens;

        Context ctx;
        ctx.user(completionPrompt);
        Completion ret;
        sanitized("OpenAI completion request", {
            ret = completions(endpoint, model, ctx, 0);
        });
        validateCompletion(ret, ctx);
    }

    @Name("OpenAI tool round trip returns a final text answer") @Serial
    unittest
    {
        string[string] env = environment.toAA();
        TestSettings options = TestSettings(env);

        string url;
        string model;
        if (!providerScenario(
                env,
                "OpenAI tool round trip",
                "INTUIT_TEST_OPENAI_URL",
                "INTUIT_TEST_OPENAI_TOOL_MODEL",
                url,
                model,
            ))
            return;

        OpenAI endpoint = new OpenAI(
            normalizeBaseUrl(url), env.get("OPENAI_API_KEY", null),
        );
        applyTimeouts(endpoint, options);
        guardDirectProvider(endpoint, model, [Modality.Text], options, env);

        Context ctx;
        ModelConfig cfg = endpoint.config(model);
        cfg.maxTokens = options.maxTokens;
        toolRoundTrip(ctx, endpoint.tools, cfg, () {
            Completion ret;
            sanitized("OpenAI tool completion request", {
                ret = completions(endpoint, model, ctx, 0);
            });
            return ret;
        });
    }

    @Name("OpenAI decisions answer all three questions") @Serial
    unittest
    {
        string[string] env = environment.toAA();
        TestSettings options = TestSettings(env);

        string url;
        string model;
        if (!providerScenario(
                env,
                "OpenAI decisions",
                "INTUIT_TEST_OPENAI_URL",
                "INTUIT_TEST_OPENAI_DECISION_MODEL",
                url,
                model,
            ))
            return;

        OpenAI endpoint = new OpenAI(
            normalizeBaseUrl(url), env.get("OPENAI_API_KEY", null),
        );
        applyTimeouts(endpoint, options);
        guardDirectProvider(endpoint, model, [Modality.Decisions], options, env);

        Decision ret;
        sanitized("OpenAI decision request", {
            ret = decisions(endpoint, model, decisionState, integrationQuestions());
        });
        validateDecision(ret);
    }

    @Name("OpenAI embeddings produce finite matching-dimension vectors") @Serial
    unittest
    {
        string[string] env = environment.toAA();
        TestSettings options = TestSettings(env);

        string url;
        string model;
        if (!providerScenario(
                env,
                "OpenAI embeddings",
                "INTUIT_TEST_OPENAI_URL",
                "INTUIT_TEST_OPENAI_EMBEDDING_MODEL",
                url,
                model,
            ))
            return;

        OpenAI endpoint = new OpenAI(
            normalizeBaseUrl(url), env.get("OPENAI_API_KEY", null),
        );
        applyTimeouts(endpoint, options);
        guardDirectProvider(
            endpoint, model, [Modality.Embedding, Modality.Embeddings], options, env,
        );

        Embedding!float single;
        Embedding!float[] batch;
        sanitized("OpenAI embedding request", {
            single = embeddings(endpoint, model, "hello");
        });
        sanitized("OpenAI batch embedding request", {
            batch = embeddings(endpoint, model, ["hello", "world"]);
        });
        validateEmbeddings(single, batch);
    }
}
