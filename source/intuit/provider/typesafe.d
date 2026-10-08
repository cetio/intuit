module intuit.provider.typesafe;

import intuit.exception : EndpointException, FormatException;
import intuit.model : ModelConfig;
import intuit.provider : request;
import intuit.provider.openai : OpenAI;
import intuit.provider.systemone : SystemOneModelConfig;

import std.json : JSONType, JSONValue;
import std.net.curl : HTTP;

public:

class TypeSafe : OpenAI
{
public:
    this(string url = "https://api.typesafe.ai", string key = null, string name = "TypeSafe")
    {
        super(url, key, name);
    }

    override ModelConfig[] available()
    {
        JSONValue json = _http.request(HTTP.Method.get, _url~"/v1/models", buildHeaders());
        if (json.type != JSONType.object || "models" !in json || json["models"].type != JSONType.array)
            throw new FormatException("Expected a TypeSafe models array.");

        foreach (item; json["models"].array)
        {
            if (item.type == JSONType.object && "name" in item && item["name"].type == JSONType.string)
                config(item["name"].str);
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

        ModelConfig ret = new SystemOneModelConfig(modelName);
        _configs ~= ret;
        return ret;
    }

    override JSONValue _decisions(ModelConfig cfg, JSONValue payload)
    {
        return _http.request(
            HTTP.Method.post,
            _url~"/v1/systemone",
            buildHeaders(),
            payload,
        );
    }

    override JSONValue _completions(ModelConfig cfg, JSONValue payload)
    {
        throw new EndpointException(
            "POST",
            "chat/completions",
            0,
            "not supported",
            "TypeSafe does not support completions.",
        );
    }

    override JSONValue _embeddings(ModelConfig cfg, JSONValue payload)
    {
        throw new EndpointException(
            "POST",
            "embeddings",
            0,
            "not supported",
            "TypeSafe does not support embeddings.",
        );
    }
}
