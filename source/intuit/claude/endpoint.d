module intuit.claude.endpoint;

import intuit.claude.model;
import intuit.response;
import intuit.utils;
import std.net.curl;
import std.string : assumeUTF;
import std.traits : isStaticArray, isDynamicArray;

class Claude
{
    string url;
    HTTP http;
    Model[string] models;
    enum string apiVersion = "2023-06-01";

    this(string url, string key = null)
    {
        this.url = url;
        http = HTTP();
        if (key != null)
        {
            http.addRequestHeader("x-api-key", key);
            http.addRequestHeader("anthropic-version", apiVersion);
            http.addRequestHeader("content-type", "application/json");
        }
    }

    Model fetch(string name)
    {
        if (name in models)
            return models[name];
        models[name] = new Model(name);
        return models[name];
    }

    Model[] available()
    {
        http.get(
            url~"v1/models",
            (ubyte[] data) {
                string str = data.assumeUTF();
                JSONValue json = str.parseJSON();
                if ("data" in json)
                {
                    foreach (m; json["data"].array)
                    {
                        string name = "id" in m ? m["id"].str : null;
                        string owner = "owned_by" in m ? m["owned_by"].str : "anthropic";
                        if (name.length > 0 && name !in models)
                            models[name] = new Model(name, owner);
                    }
                }
            },
            (ubyte[] data) => throw new Exception("Unable to retrieve available models!")
        );
        return models.values;
    }

    Completion completions(T)(string name, T data)
    {
        Model model = fetch(name);
        JSONValue json = model.messagesJSON(data);
        Completion ret;

        http.post(
            url~"v1/messages",
            (ubyte[] data) {
                string str = data.assumeUTF();
                ret = model.parseCompletions(str.parseJSON());
            },
            (ubyte[] data) => throw new Exception("Connection to endpoint failed!"),
            json
        );
        return ret;
    }
}
