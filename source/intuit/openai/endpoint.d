module intuit.openai.endpoint;

import intuit.openai.model;
import intuit.response;
import intuit.utils;
import std.net.curl;
import std.string : assumeUTF;
import std.traits : isStaticArray, isDynamicArray;
import std.algorithm : map;
import std.array : array;

class OpenAI
{
    string url;
    HTTP http;
    Model[string] models;

    this(string url, string key = null)
    {
        this.url = url;
        http = HTTP();
        if (key != null)
            http.addRequestHeader("Authorization", "Bearer "~key);
    }

    Model fetch(string name)
    {
        if (name in models)
            return models[name];
        else
        {
            if (models.values.length < available.length && name in models)
                return models[name];
        }

        throw new Exception("Model '"~name~"' not available.");
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
                        string owner = "owned_by" in m ? m["owned_by"].str : null;

                        if (name !in models)
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
        JSONValue json = model.completionsJSON(data);
        Completion ret;

        http.post(
            url~"v1/chat/completions",
            (ubyte[] data) {
                string str = data.assumeUTF();
                ret = model.parseCompletions(str.parseJSON());
            },
            (ubyte[] data) => throw new Exception("Connection to endpoint failed!"),
            json
        );
        return ret;
    }

    auto embeddings(A = float, B)(string name, B data)
    {
        Model model = fetch(name);
        JSONValue json = model.embeddingsJSON(data);
        static if (!is(B == string) && (isDynamicArray!B || isStaticArray!B))
            Embedding!A[] ret;
        else
            Embedding!A ret;

        http.post(
            url~"v1/embeddings",
            (ubyte[] data) {
                string str = data.assumeUTF();
                static if (!is(B == string) && (isDynamicArray!B || isStaticArray!B))
                    ret = model.parseEmbeddingsBatch!A(str.parseJSON());
                else
                    ret = model.parseEmbeddings!A(str.parseJSON());
            },
            (ubyte[] data) => throw new Exception("Connection to endpoint failed!"),
            json
        );
        return ret;
    }
}
