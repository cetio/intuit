/// Qwen-compatible endpoint implementation.
module intuit.qwen.endpoint;

import intuit.error : EndpointError;
import intuit.endpoint;
import intuit.model;
import intuit.openai.endpoint;
import intuit.qwen.model;
import intuit.tool;
import conductor.http : Response, send;
import std.net.curl : HTTP;
import std.json : JSONType, JSONValue, parseJSON;
import std.string : assumeUTF;

/// Qwen-compatible LLM endpoint, extending OpenAI with Qwen-specific models.
class Qwen : OpenAI
{
public:
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

    override IModel[] available()
    {
        JSONValue json = request(HTTP.Method.get, "models");
        IModel[] models;
        if ("data" in json && json["data"].type == JSONType.array)
        {
            foreach (item; json["data"].array)
            {
                string name = "id" in item ? item["id"].str : null;
                if (name !is null)
                {
                    string owner = "owned_by" in item ? item["owned_by"].str : null;
                    models ~= new QwenModel(name, owner);
                }
            }
        }
        return models;
    }

    /// Creates a QwenModel by name.
    override IModel model(string name)
        => new QwenModel(name);
}
