module tests.router.local;

import intuit.context;
import intuit.model;
import intuit.provider : IEndpoint;
import intuit.response;
import intuit.router;
import intuit.tool;
import unit_threaded;

import std.exception : assertThrown;
import std.json : JSONValue;

private final class StubModel : IModel
{
    private string _name;
    private string _owner;

    this(string name)
    {
        this._name = name;
    }

    override ref string name()
        => _name;

    override ref string owner()
        => _owner;

    override JSONValue completionsJSON(JSONValue input, ToolRegistry tools = ToolRegistry.init)
        => JSONValue.emptyObject;

    override JSONValue embeddingsJSON(JSONValue input)
        => JSONValue.emptyObject;

    override Completion parseCompletions(JSONValue response)
        => Completion.init;

    override JSONValue parseEmbeddings(JSONValue response)
        => JSONValue.emptyArray;
}

private final class StubEndpoint : IEndpoint
{
    private string _name = "stub";
    private string _url;
    private string _key;
    private ToolRegistry _tools;

    override ref string name()
        => _name;

    override ref string url()
        => _url;

    override ref string key()
        => _key;

    override ref ToolRegistry tools()
        => _tools;

    override IModel[] available()
        => null;

    override IModel model(string name)
        => new StubModel(name);

    override JSONValue _completions(IModel model, JSONValue payload)
        => JSONValue.emptyObject;

    override JSONValue _embeddings(IModel model, JSONValue payload)
        => JSONValue.emptyArray;

    override CompletionStream _stream(IModel model, JSONValue payload)
        => new CompletionStream(model.name, null);
}

@Name("active model sets compactor token limit from catalog")
unittest
{
    LocalRouter router = new LocalRouter(new StubEndpoint());
    router.active("gpt-4o");

    assert(router.active == "gpt-4o");
    assert(router.context.compactor.maxTokens == catalog["gpt-4o"].contextWindow);
}

@Name("switching active model re-limits but keeps context messages")
unittest
{
    LocalRouter router = new LocalRouter(new StubEndpoint());
    router.active("gpt-4o");
    router.context.user("hello");

    router.active("claude-3-5-sonnet-20241022");

    assert(router.context.length == 1);
    assert(router.context.compactor.maxTokens == catalog["claude-3-5-sonnet-20241022"].contextWindow);
}

@Name("unknown active model and missing active model throw")
unittest
{
    LocalRouter router = new LocalRouter(new StubEndpoint());

    assertThrown!Exception(router.active("does-not-exist"));
    assertThrown!Exception(router.model());
    assertThrown!Exception(completions(router, "hi"));
}

@Name("catalog exposes one model per supported provider")
unittest
{
    LocalRouter router = new LocalRouter(new StubEndpoint());

    LocalModelDetails[] openai = router.filter(d => d.provider == "openai");
    assert(openai.length == 1);
    assert(known("qwen2.5-72b-instruct"));
    assert(router.available().length == 3);
}
