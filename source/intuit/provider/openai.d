/// OpenAI-compatible endpoint implementation.
module intuit.provider.openai;

public import intuit.provider;
import intuit.model;
import intuit.tool;

import std.net.curl : HTTP;
import std.json : JSONType, JSONValue;

/// OpenAI-compatible LLM endpoint.
class OpenAI : IEndpoint
{
private:
    string _name;
    string _key;
    ToolRegistry _tools;

protected:
    string _url;
    HTTP _http;
    ModelConfig[] _configs;

    string[string] buildHeaders()
    {
        string[string] ret;
        ret["Content-Type"] = "application/json";
        if (_key.length > 0)
            ret["Authorization"] = "Bearer "~_key;
        return ret;
    }

public:
    /**
     * Constructs an OpenAI endpoint.
     *
     * The provided URL is used as-is and the caller is responsible for
     * supplying the correct base URL for the endpoint.
     *
     * Params:
     *  url = The base URL of the endpoint.
     *  key = Optional API key.
     *  name = Display name for the endpoint.
     */
    this(string url, string key = null, string name = "OpenAI")
    {
        _name = name;
        _url = url;
        _key = key;
        _http = HTTP();
    }

    override ref string name()
        => _name;

    override ref string url()
        => _url;

    override ref string key()
        => _key;

    override ref ToolRegistry tools()
        => _tools;

    override ModelConfig[] available()
    {
        JSONValue json = _http.request(HTTP.Method.get, _url~"/v1/models", buildHeaders());
        if ("data" in json && json["data"].type == JSONType.array)
        {
            foreach (item; json["data"].array)
            {
                string name = "id" in item ? item["id"].str : null;
                if (name is null)
                    continue;

                bool found = false;
                foreach (cfg; _configs)
                {
                    if (cfg.name == name)
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                    _configs ~= new ModelConfig(name);
            }
        }
        return _configs;
    }

    override ModelConfig config(string modelName)
    {
        foreach (cfg; _configs)
        {
            if (cfg.name == modelName)
                return cfg;
        }
        
        ModelConfig cfg = new ModelConfig(modelName);
        _configs ~= cfg;
        return cfg;
    }

    override ModelConfig[] configs()
        => _configs;

    override JSONValue _completions(ModelConfig cfg, JSONValue payload)
    {
        return _http.request(
            HTTP.Method.post,
            _url~"/v1/chat/completions",
            buildHeaders(),
            payload,
        );
    }

    override JSONValue _embeddings(ModelConfig cfg, JSONValue payload)
    {
        return _http.request(
            HTTP.Method.post,
            _url~"/v1/embeddings",
            buildHeaders(),
            payload,
        );
    }
}
