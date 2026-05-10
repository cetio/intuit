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

class Qwen : OpenAI
{
public:
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
            foreach (m; json["data"].array)
            {
                string name = "id" in m ? m["id"].str : null;
                if (name !is null)
                {
                    string owner = "owned_by" in m ? m["owned_by"].str : null;
                    models ~= new QwenModel(name, owner);
                }
            }
        }
        return models;
    }

    override IModel model(string name)
        => new QwenModel(name);
}
