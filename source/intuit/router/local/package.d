/// Local router exposing static model metadata over a single backing endpoint.
module intuit.router.local;

public import intuit.router.local.catalog;

import intuit.context;
import intuit.model;
import intuit.provider : IEndpoint;
import intuit.response;
import intuit.router;
import intuit.tool;

import std.json : JSONValue;

/// Router backed by a single endpoint with a static, queryable model catalog.
class LocalRouter : IRouter
{
private:
    string _name;
    IEndpoint _backing;
    Context _context;
    string _active;
    IModel _activeModel;

public:
    /**
     * Constructs a local router over a backing endpoint.
     *
     * Params:
     *  backing = The endpoint that fulfills requests.
     *  name = Display name for the router.
     */
    this(IEndpoint backing, string name = "LocalRouter")
    {
        this._name = name;
        this._backing = backing;
        this._context.compactor = new Compactor();
    }

    override ref string name()
        => _name;

    override ref ToolRegistry tools()
        => _backing.tools();

    override ref Context context()
        => _context;

    override string active()
        => _active;

    override void active(string name)
    {
        LocalModelDetails details = lookup(name);
        _active = name;
        _activeModel = _backing.model(name);
        _context.compactor.maxTokens = details.contextWindow;
    }

    override IModel model()
    {
        if (_active is null)
            throw new Exception("Router has no active model set.");
        return _activeModel;
    }

    override JSONValue _completions(JSONValue payload)
        => _backing._completions(_activeModel, payload);

    override JSONValue _embeddings(JSONValue payload)
        => _backing._embeddings(_activeModel, payload);

    override CompletionStream _stream(JSONValue payload)
        => _backing._stream(_activeModel, payload);

    /// Lists every model in the catalog.
    LocalModelDetails[] available()
    {
        LocalModelDetails[] ret;
        foreach (details; catalog.byValue)
            ret ~= details;
        return ret;
    }

    /// Gets the catalog details for a model by name.
    LocalModelDetails details(string name)
        => lookup(name);

    /**
     * Filters the catalog by a predicate.
     *
     * Params:
     *  pred = The predicate to apply to each entry.
     *
     * Returns:
     *  The matching catalog entries.
     */
    LocalModelDetails[] filter(bool delegate(LocalModelDetails) pred)
    {
        LocalModelDetails[] ret;
        foreach (details; catalog.byValue)
        {
            if (pred(details))
                ret ~= details;
        }
        return ret;
    }

    /**
     * Sorts the catalog by a comparison delegate.
     *
     * Params:
     *  less = Returns true if the first entry should precede the second.
     *
     * Returns:
     *  The sorted catalog entries.
     */
    LocalModelDetails[] sort(bool delegate(LocalModelDetails, LocalModelDetails) less)
    {
        import std.algorithm.sorting : sort;

        LocalModelDetails[] ret = available();
        ret.sort!((a, b) => less(a, b));
        return ret;
    }
}
