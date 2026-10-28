module tests.integration.routers.openrouter;

version(integration)
{
    import intuit;
    import tests.integration.common;
    import tests.integration.settings;
    import unit_threaded;

    import std.algorithm.searching : canFind;
    import std.process : environment;
    import std.stdio : stderr;

    private enum defaultModel = "nvidia/nemotron-3.5-lightning:free";
    private enum defaultDecisionModel = "respan/span-01-lite:free";

    @Name("OpenRouter catalog keys match details and configs are cached") @Serial
    unittest
    {
        string[string] env = environment.toAA();
        TestSettings options = TestSettings(env);

        OpenRouter router = new OpenRouter(
            env.get("OPENROUTER_API_KEY", null),
            env.get("INTUIT_TEST_OPENROUTER_URL", "https://openrouter.ai"),
        );
        applyTimeouts(router, options);

        ModelDetails[string] catalog;
        sanitized("OpenRouter catalog request", {
            catalog = router.catalog;
        });
        validateCatalog(catalog);

        string model = env.get("INTUIT_TEST_OPENROUTER_MODEL", defaultModel);
        ModelConfig first = router.config(model);
        (first is router.config(model)).shouldBeTrue;

        ModelDetails* details = model in catalog;
        if (details !is null && details.contextLength > 0)
        {
            router.active(model);
            router.context.compactor.maxTokens.should == details.contextLength;
        }
    }

    @Name("OpenRouter completion maintains user and assistant history") @Serial
    unittest
    {
        string[string] env = environment.toAA();
        TestSettings options = TestSettings(env);
        if (env.get("OPENROUTER_API_KEY", null).length == 0)
        {
            stderr.writeln("SKIP OpenRouter completion: missing OPENROUTER_API_KEY");
            return;
        }

        OpenRouter router = new OpenRouter(
            env.get("OPENROUTER_API_KEY", null),
            env.get("INTUIT_TEST_OPENROUTER_URL", "https://openrouter.ai"),
        );
        applyTimeouts(router, options);

        ModelDetails[string] catalog;
        sanitized("OpenRouter catalog request", {
            catalog = router.catalog;
        });
        string model = env.get("INTUIT_TEST_OPENROUTER_MODEL", defaultModel);
        requireModel(catalog, model, [Modality.Text], options.allowPaid);
        if (!options.allowPaid)
            router.provider = zeroPriceCap();

        router.active(model);
        router.config().maxTokens = options.maxTokens;

        Completion ret;
        sanitized("OpenRouter completion request", {
            ret = completions(router, completionPrompt, 0);
        });
        validateCompletion(ret, router.context);
    }

    @Name("OpenRouter tool round trip returns a final text answer") @Serial
    unittest
    {
        string[string] env = environment.toAA();
        TestSettings options = TestSettings(env);
        if (env.get("OPENROUTER_API_KEY", null).length == 0)
        {
            stderr.writeln("SKIP OpenRouter tool round trip: missing OPENROUTER_API_KEY");
            return;
        }

        OpenRouter router = new OpenRouter(
            env.get("OPENROUTER_API_KEY", null),
            env.get("INTUIT_TEST_OPENROUTER_URL", "https://openrouter.ai"),
        );
        applyTimeouts(router, options);

        ModelDetails[string] catalog;
        sanitized("OpenRouter catalog request", {
            catalog = router.catalog;
        });
        string model = env.get(
            "INTUIT_TEST_OPENROUTER_TOOL_MODEL",
            env.get("INTUIT_TEST_OPENROUTER_MODEL", defaultModel),
        );
        ModelDetails details = requireModel(
            catalog, model, [Modality.Text], options.allowPaid,
        );
        if (!details.capabilities.canFind(ModelCapability.Tools)
            || !details.capabilities.canFind(ModelCapability.ToolChoice))
        {
            throw new Exception(
                "Configured OpenRouter tool model lacks tools/tool_choice support: "~model,
            );
        }
        if (!options.allowPaid)
            router.provider = zeroPriceCap();

        router.active(model);
        ModelConfig cfg = router.config();
        cfg.maxTokens = options.maxTokens;
        toolRoundTrip(router.context, router.tools, cfg, () {
            Completion ret;
            sanitized("OpenRouter tool completion request", {
                ret = completions(router, 0);
            });
            return ret;
        });
    }

    @Name("OpenRouter predicate decision leaves context unchanged") @Serial
    unittest
    {
        string[string] env = environment.toAA();
        TestSettings options = TestSettings(env);
        if (env.get("OPENROUTER_API_KEY", null).length == 0)
        {
            stderr.writeln("SKIP OpenRouter decisions: missing OPENROUTER_API_KEY");
            return;
        }

        OpenRouter router = new OpenRouter(
            env.get("OPENROUTER_API_KEY", null),
            env.get("INTUIT_TEST_OPENROUTER_URL", "https://openrouter.ai"),
        );
        applyTimeouts(router, options);

        ModelDetails[string] catalog;
        sanitized("OpenRouter catalog request", {
            catalog = router.catalog;
        });
        string model = env.get("INTUIT_TEST_OPENROUTER_DECISION_MODEL", defaultDecisionModel);
        requireModel(catalog, model, [Modality.Decisions], options.allowPaid);
        if (!options.allowPaid)
            router.provider = zeroPriceCap();

        router.active(model);
        router.context.user("Preexisting context must be unchanged.");
        const size_t historyLength = router.context.length;

        // The zero-cost Respan default supports Noul predicates only. The
        // offline System One tests cover choice and score question mapping.
        DecisionQuestion[] questions = integrationQuestions();
        questions = questions[0 .. 1];
        Decision ret;
        sanitized("OpenRouter decision request", {
            ret = decisions(router, decisionState, questions);
        });
        validatePredicateDecision(ret);
        router.context.length.should == historyLength;
    }

    @Name("OpenRouter embeddings produce finite matching-dimension vectors") @Serial
    unittest
    {
        string[string] env = environment.toAA();
        TestSettings options = TestSettings(env);
        if (env.get("OPENROUTER_API_KEY", null).length == 0)
        {
            stderr.writeln("SKIP OpenRouter embeddings: missing OPENROUTER_API_KEY");
            return;
        }
        string model = env.get("INTUIT_TEST_OPENROUTER_EMBEDDING_MODEL", null);
        if (model.length == 0)
        {
            stderr.writeln("SKIP OpenRouter embeddings: missing INTUIT_TEST_OPENROUTER_EMBEDDING_MODEL");
            return;
        }

        OpenRouter router = new OpenRouter(
            env.get("OPENROUTER_API_KEY", null),
            env.get("INTUIT_TEST_OPENROUTER_URL", "https://openrouter.ai"),
        );
        applyTimeouts(router, options);

        ModelDetails[string] catalog;
        sanitized("OpenRouter catalog request", {
            catalog = router.catalog;
        });
        requireModel(
            catalog, model, [Modality.Embedding, Modality.Embeddings], options.allowPaid,
        );
        if (!options.allowPaid)
            router.provider = zeroPriceCap();

        router.active(model);
        Embedding!float single;
        Embedding!float[] batch;
        sanitized("OpenRouter embedding request", {
            single = embeddings(router, "hello");
        });
        sanitized("OpenRouter batch embedding request", {
            batch = embeddings(router, ["hello", "world"]);
        });
        validateEmbeddings(single, batch);
    }
}
