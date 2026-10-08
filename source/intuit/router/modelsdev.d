/// Unauthenticated model catalog router backed by the models.dev API.
module intuit.router.modelsdev;

import intuit.context;
import intuit.json : fromJSON;
import intuit.model;
import intuit.provider : request;
import intuit.router.details;
import intuit.router;
import intuit.tool;

import std.json : JSONType, JSONValue;
import std.net.curl : HTTP;
import core.time : Duration;

class ModelsDev : IRouter
{
private:
    string _name;
    string _url;
    ToolRegistry _tools;
    Context _context;
    HTTP _http;
    ModelDetails[string] _catalog;

public:
    /**
     * Constructs a models.dev catalog router.
     *
     * Params:
     *  url = The base URL of models.dev, defaulting to the public host.
     *  name = Display name for the router.
     */
    this(string url = "https://models.dev", string name = "models.dev")
    {
        this._name = name;
        this._url = url;
        this._http = HTTP();
        this._context.compactor = new Compactor();
    }

    override void operationTimeout(Duration timeout)
    {
        _http.operationTimeout = timeout;
    }

    override void connectTimeout(Duration timeout)
    {
        _http.connectTimeout = timeout;
    }

    override ref string name()
        => _name;

    override ref ToolRegistry tools()
        => _tools;

    override ref Context context()
        => _context;

    override string active()
        => null;

    override void active(string modelName)
    {
        throw new Exception("models.dev router currently only supports catalog access.");
    }

    override ModelConfig config()
    {
        throw new Exception("models.dev router currently only supports catalog access.");
    }

    override ModelConfig config(string modelName)
    {
        throw new Exception("models.dev router currently only supports catalog access.");
    }

    override ModelConfig[] configs()
    {
        throw new Exception("models.dev router currently only supports catalog access.");
    }

    override ModelDetails[string] catalog()
    {
        if (_catalog.length == 0)
            refresh();
        return _catalog;
    }

    override JSONValue _completions(JSONValue payload)
    {
        throw new Exception("models.dev router currently only supports catalog access.");
    }

    override JSONValue _decisions(JSONValue payload)
    {
        throw new Exception("models.dev router currently only supports catalog access.");
    }

    override JSONValue _embeddings(JSONValue payload)
    {
        throw new Exception("models.dev router currently only supports catalog access.");
    }

    /// Re-fetches the model catalog from `/api.json?type=all`.
    override void refresh()
    {
        JSONValue json = _http.request(HTTP.Method.get, _url~"/api.json?type=all");
        _catalog = null;
        foreach (providerId, provider; json.object)
        {
            if (provider.type != JSONType.object || "models" !in provider
                || provider["models"].type != JSONType.object)
                continue;

            foreach (item; provider["models"].object.byValue)
            {
                ModelDetails details = parseDetails(providerId, item);
                if (details.id.length > 0)
                    _catalog[details.id] = details;
            }
        }
    }

private:
    /// Parses a single provider model into ModelDetails.
    static ModelDetails parseDetails(string providerId, JSONValue item)
    {
        ModelDetails ret;
        if (item.type != JSONType.object)
            return ret;

        ret.id = "id" in item ? item["id"].str : null;
        if (ret.id.length == 0)
            return ret;
        ret.id = providerId~"/"~ret.id;

        if ("name" in item && item["name"].type == JSONType.string)
            ret.name = item["name"].str;
        if (ret.name.length == 0)
            ret.name = ret.id[providerId.length + 1..$];

        if ("description" in item && item["description"].type == JSONType.string)
            ret.description = item["description"].str;

        if ("limit" in item && item["limit"].type == JSONType.object)
        {
            JSONValue limit = item["limit"];
            if ("context" in limit && limit["context"].type == JSONType.integer)
                ret.contextLength = cast(size_t)limit["context"].integer;
            if ("output" in limit && limit["output"].type == JSONType.integer)
                ret.maxCompletionTokens = cast(size_t)limit["output"].integer;
        }

        if ("modalities" in item && item["modalities"].type == JSONType.object)
        {
            JSONValue modalities = item["modalities"];
            if ("input" in modalities && modalities["input"].type == JSONType.array)
            {
                foreach (entry; modalities["input"].array)
                {
                    if (entry.type != JSONType.string)
                        continue;
                    ret.inputModalities ~= cast(Modality)entry.str;
                }
            }

            if ("output" in modalities && modalities["output"].type == JSONType.array)
            {
                foreach (entry; modalities["output"].array)
                {
                    if (entry.type != JSONType.string)
                        continue;
                    ret.outputModalities ~= cast(Modality)entry.str;
                }
            }
        }

        if ("type" in item && item["type"].type == JSONType.string)
        {
            string modelType = item["type"].str;
            if (modelType == "decision")
                ret.outputModalities ~= Modality.Decisions;
            if (modelType == "embedding")
                ret.outputModalities ~= Modality.Embedding;
            if (modelType == "reranking")
                ret.outputModalities ~= Modality.Rerank;
        }

        if ("tool_call" in item && item["tool_call"].type == JSONType.true_)
            ret.capabilities ~= ModelCapability.Tools;
        if ("reasoning" in item && item["reasoning"].type == JSONType.true_)
            ret.capabilities ~= ModelCapability.Reasoning;
        if ("structured_output" in item && item["structured_output"].type == JSONType.true_)
            ret.capabilities ~= ModelCapability.StructuredOutputs;
        if ("temperature" in item && item["temperature"].type == JSONType.true_)
            ret.capabilities ~= ModelCapability.Temperature;

        if ("cost" in item && item["cost"].type == JSONType.object)
        {
            JSONValue cost = item["cost"];
            if ("input" in cost && cost["input"].type != JSONType.null_)
                ret.promptCost = cost["input"].fromJSON!double;
            if ("output" in cost && cost["output"].type != JSONType.null_)
                ret.completionCost = cost["output"].fromJSON!double;
        }

        return ret;
    }
}
