module tests.integration.routers.litellm;

version(integration)
{
    import intuit;
    import tests.integration.common;
    import tests.integration.settings;
    import unit_threaded;

    import std.process : environment;
    import std.stdio : stderr;

    @Name("Configured LiteLLM proxy catalog keys match model details") @Serial
    unittest
    {
        string[string] env = environment.toAA();
        TestSettings options = TestSettings(env);

        string url = env.get("INTUIT_TEST_LITELLM_URL", null);
        if (url.length == 0)
        {
            stderr.writeln("SKIP LiteLLM catalog: missing INTUIT_TEST_LITELLM_URL");
            return;
        }

        LiteLLM router = new LiteLLM(url, env.get("LITELLM_API_KEY", null));
        applyTimeouts(router, options);

        ModelDetails[string] catalog;
        sanitized("LiteLLM catalog request", {
            catalog = router.catalog;
        });
        validateCatalog(catalog);
    }
}
