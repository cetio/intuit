/// Qwen-compatible endpoint implementation.
module intuit.provider.qwen.endpoint;

public import intuit.provider;
import intuit.error : EndpointError;
import intuit.model;
import intuit.provider.openai;
import intuit.provider.qwen.model;
import intuit.tool;
import conductor.http : Response, send;

import std.net.curl : HTTP;
import std.json : JSONType, JSONValue, parseJSON;
import std.string : assumeUTF;

/// Qwen-compatible LLM endpoint, extending OpenAI with Qwen-specific models.
class Qwen : OpenAI
{
    /**
     * Constructs a Qwen endpoint.
     *
     * Params:
     *  url = The base URL of the endpoint.
     *  key = Optional API key.
     *  name = Display name for the endpoint.
     */
    this(string url, string key = null, string name = "Qwen")
    {
        super(url, key, name);
    }

    override ModelConfig[] available()
    {
        JSONValue json = request(HTTP.Method.get, "models");
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
                {
                    ModelConfig cfg = new QwenModelConfig(name);
                    _configs ~= cfg;
                }
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

        ModelConfig cfg = new QwenModelConfig(modelName);
        _configs ~= cfg;
        return cfg;
    }
}
