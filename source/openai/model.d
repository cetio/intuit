module intuit.openai.model;

import intuit.response;
import intuit.utils;
import std.variant;
import std.conv;
import std.traits : isIntegral;

class Model
{
    string name;
    string owner;
    //Tool[] _tools;

    double temperature = double.nan;
    double topP = double.nan;
    long maxTokens = -1;
    string[] stop;
    double presencePenalty = double.nan;
    double frequencyPenalty = double.nan;
    long n = 1;
    bool stream = false;
    long[long] logitBias;
    long seed = 0;
    string encodingFormat = "float";
    long dimensions = 0;

    this(string name, string owner)
    {
        this.name = name;
        this.owner = owner;
    }

    JSONValue completionsJSON(T)(T data)
    {
        JSONValue ret = JSONValue.emptyObject;
        ret["model"] = JSONValue(name);
        ret["max_tokens"] = JSONValue(maxTokens);

        // Options
        if (temperature !is double.nan) ret["temperature"] = JSONValue(temperature);
        if (topP !is double.nan) ret["top_p"] = JSONValue(topP);
        if (stop.length > 0)
        {
            JSONValue arr = JSONValue.emptyArray;
            foreach (s; stop) arr.array ~= JSONValue(s);
            ret["stop"] = arr;
        }
        if (presencePenalty !is double.nan) ret["presence_penalty"] = JSONValue(presencePenalty);
        if (frequencyPenalty !is double.nan) ret["frequency_penalty"] = JSONValue(frequencyPenalty);
        if (n > 1) ret["n"] = JSONValue(n);
        if (stream) ret["stream"] = JSONValue(stream);
        if (logitBias.length > 0)
        {
            JSONValue bias = JSONValue.emptyObject;
            foreach (k, v; logitBias) bias[k.to!string] = JSONValue(v);
            ret["logit_bias"] = bias;
        }
        if (seed > 0) ret["seed"] = JSONValue(seed);

        // Payload
        // If data is already a JSONValue array (messages), use it directly
        static if (is(T == JSONValue))
        {
            if (data.type == JSONType.array)
            {
                ret["messages"] = data;
            }
            else
            {
                // Fallback: treat as single message
                ret["messages"] = JSONValue.emptyArray;
                ret["messages"].array ~= JSONValue.emptyObject;
                ret["messages"][0]["role"] = JSONValue("user");
                ret["messages"][0]["content"] = data;
            }
        }
        else
        {
            // Original behavior: single user message
            ret["messages"] = JSONValue.emptyArray;
            ret["messages"].array ~= JSONValue.emptyObject;
            ret["messages"][0]["role"] = JSONValue("user");
            ret["messages"][0]["content"] = data.toJSON();
        }
        return ret;
    }

    JSONValue embeddingsJSON(T)(T data)
    {
        JSONValue ret = JSONValue.emptyObject;
        ret["model"] = JSONValue(name);

        // Options
        if (encodingFormat != "float") ret["encoding_format"] = JSONValue(encodingFormat);
        if (dimensions > 0) ret["dimensions"] = JSONValue(dimensions);

        // Payload
        ret["input"] = data.toJSON();
        return ret;
    }

    Completion parseCompletions(JSONValue json)
    {
        Completion ret;
        if ("error" in json)
        {
            if (json["error"].type == JSONType.string)
                throw new Exception(json["error"].str);
            else if (json["error"].type == JSONType.object && "message" in json["error"])
                throw new Exception(json["error"]["message"].str);
            else
                throw new Exception("Critical error parsing error JSON "~json.toPrettyString());
        }

        if ("choices" in json)
        {
            foreach (c; json["choices"].array)
            {
                Choice choice;
                JSONValue msg = ("message" in c) ? c["message"] : c["delta"];

                string content = ("content" in msg && !msg["content"].isNull) 
                    ? msg["content"].str 
                    : cast(string)null;
                choice.data = Variant(content);
                
                bool hasFinishReason = ("finish_reason" in c && !c["finish_reason"].isNull);
                choice.finishReason = hasFinishReason
                    ? cast(FinishReason)c["finish_reason"].str
                    : FinishReason.Unknown;
                
                bool hasLogProbs = ("log_probs" in c && !c["log_probs"].isNull);
                choice.logProbs = hasLogProbs
                    ? cast(float)c["log_probs"].floating
                    : float.nan;
                ret.choices ~= choice;
            }
        }
        return ret;
    }

    Embedding!T parseEmbeddings(T)(JSONValue json)
    {
        Embedding!T ret;
        if ("error" in json)
        {
            if (json["error"].type == JSONType.string)
                throw new Exception(json["error"].str);
            else if (json["error"].type == JSONType.object && "message" in json["error"])
                throw new Exception(json["error"]["message"].str);
            else
                throw new Exception("Critical error parsing error JSON "~json.toPrettyString());
        }

        if ("data" in json && json["data"].array.length > 0)
        {
            JSONValue data = json["data"].array[0];
            if ("embedding" in data)
            {
                ret.value = new T[](data["embedding"].array.length);
                foreach (i, v; data["embedding"].array)
                {
                    static if (isIntegral!T)
                        ret.value[i] = cast(T)v.integer;
                    else
                        ret.value[i] = cast(T)v.floating;
                }
            }
        }
        return ret;
    }

    Embedding!T[] parseEmbeddingsBatch(T)(JSONValue json)
    {
        Embedding!T[] ret;
        if ("error" in json)
        {
            if (json["error"].type == JSONType.string)
                throw new Exception(json["error"].str);
            else if (json["error"].type == JSONType.object && "message" in json["error"])
                throw new Exception(json["error"]["message"].str);
            else
                throw new Exception("Critical error parsing error JSON "~json.toPrettyString());
        }

        if ("data" in json && json["data"].array.length > 0)
        {
            foreach (data; json["data"].array)
            {
                Embedding!T emb;
                if ("embedding" in data)
                {
                    emb.value = new T[](data["embedding"].array.length);
                    foreach (i, v; data["embedding"].array)
                    {
                        static if (isIntegral!T)
                            emb.value[i] = cast(T)v.integer;
                        else
                            emb.value[i] = cast(T)v.floating;
                    }
                }
                ret ~= emb;
            }
        }
        return ret;
    }

    override string toString() const
    {
        return "Model("~name~", "~owner~")";
    }
}
