module tests.common;

version(integration)
{
    import intuit;
    import std.variant : Variant;

    struct TestConfig
    {
        /// Since tests are very likely to be impure, we use a delegate to create the instance.
        /// This is called for every run with the endpoint.
        /// Must return a Variant containing an uncast IEndpoint.
        Variant delegate() instance;
        /// The name of the model to use for this endpoint.
        string modelName;
    }

    static TestConfig[] configs;

    static this()
    {
        configs ~= TestConfig(() => Variant(new OpenAI("http://127.0.0.1:1234")), "gpt-oss-20b");
        configs ~= TestConfig(() => Variant(new Qwen("http://127.0.0.1:1234")), "qwen3-30b-a3b");
        configs ~= TestConfig(() => Variant(new Claude("https://api.anthropic.com", null)), "claude-sonnet-4-5");
    }

    /// Run a test with the first available endpoint.
    void testOnce(void delegate(IEndpoint endpoint, string modelName) dg)
    {
        foreach (config; configs)
        {
            dg(config.instance().get!IEndpoint, config.modelName);
            return;
        }
    }

    /// Run a test with the first available endpoint of type E.
    void testOnce(E)(void delegate(E endpoint, string modelName) dg)
        if (is(E : IEndpoint))
    {
        foreach (config; configs)
        {
            E* instance = config.instance().peek!E;
            if (instance != null)
            {
                dg(instance, config.modelName);
                return;
            }
        }
    }
}
