module tests.integration.routers.modelsdev;

version(integration)
{
    import intuit;
    import tests.integration.common;
    import tests.integration.settings;
    import unit_threaded;

    import std.process : environment;

    @Name("Models.dev catalog keys match model details") @Serial
    unittest
    {
        string[string] env = environment.toAA();
        TestSettings options = TestSettings(env);

        ModelsDev router = new ModelsDev(
            env.get("INTUIT_TEST_MODELSDEV_URL", "https://models.dev"),
        );
        applyTimeouts(router, options);

        ModelDetails[string] catalog;
        sanitized("Models.dev catalog request", {
            catalog = router.catalog;
        });
        validateCatalog(catalog);
    }
}
