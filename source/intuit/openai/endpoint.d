module intuit.openai.endpoint;

import intuit.endpoint;
import intuit.model;
import intuit.openai.model;
import intuit.utils;
import std.net.curl : HTTP;
import std.string : assumeUTF;

class OpenAI : IEndpoint
{
    private string _name;
    private string _url;
    private string _key;
    HTTP http;

    this(string name, string url, string key = null)
    {
        this._name = name;
        this._url = url;
        this._key = key;
        rebuildHTTP();
    }

    override string name() => _name;
    override void name(string value) { _name = value; }
    override string url() => _url;
    override void url(string value) { _url = value; }
    override void key(string value)
    {
        _key = value;
        rebuildHTTP();
    }

    override IModel[] available()
    {
        IModel[] models;
        http.get(
            _url~"v1/models",
            (ubyte[] data) {
                JSONValue json = data.assumeUTF().parseJSON();
                if ("data" in json)
                {
                    foreach (m; json["data"].array)
                    {
                        string name = "id" in m ? m["id"].str : null;
                        string owner = "owned_by" in m ? m["owned_by"].str : null;
                        models ~= new OpenAIModel(name, owner);
                    }
                }
            },
            (ubyte[] data) => throw new Exception("Unable to retrieve available models!")
        );
        return models;
    }

    override JSONValue _completions(IModel model, JSONValue payload)
    {
        JSONValue ret;
        http.post(
            _url~"v1/chat/completions",
            (ubyte[] data) {
                ret = data.assumeUTF().parseJSON();
            },
            (ubyte[] data) => throw new Exception("Connection to endpoint failed!"),
            payload
        );
        return ret;
    }

    override JSONValue _embeddings(IModel model, JSONValue payload)
    {
        JSONValue ret;
        http.post(
            _url~"v1/embeddings",
            (ubyte[] data) {
                ret = data.assumeUTF().parseJSON();
            },
            (ubyte[] data) => throw new Exception("Connection to endpoint failed!"),
            payload
        );
        return ret;
    }

private:
    void rebuildHTTP()
    {
        http = HTTP();
        if (_key !is null)
            http.addRequestHeader("Authorization", "Bearer "~_key);
    }
}
