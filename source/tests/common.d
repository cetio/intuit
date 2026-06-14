module tests.common;

version(integration)
{
    import intuit;
    import std.variant : Variant;

    struct TestConfig
    {
        Variant delegate() instance;
        string model;
    }

    static TestConfig[] configs;

    static this()
    {
        configs ~= TestConfig(() => Variant(new OpenAI("http://127.0.0.1:1234")), "gpt-oss-20b");
        configs ~= TestConfig(() => Variant(new Qwen("http://127.0.0.1:1234")), "qwen3-30b-a3b");
        configs ~= TestConfig(() => Variant(new Claude("https://api.anthropic.com", null)), "claude-sonnet-4-5");
    }

    void testOnce(void delegate(IEndpoint endpoint, string model) dg)
    {
        foreach (config; configs)
        {
            dg(config.instance().get!IEndpoint, config.model);
            return;
        }
    }

    void testOnce(E)(void delegate(E endpoint, string model) dg)
        if (is(E : IEndpoint))
    {
        foreach (config; configs)
        {
            E* instance = config.instance().peek!E;
            if (instance != null)
            {
                dg(instance, config.model);
                return;
            }
        }
    }
}
