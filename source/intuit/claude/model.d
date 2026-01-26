module intuit.claude.model;

import intuit.response;
import intuit.utils;
import std.variant;
import std.conv;
import std.traits : isIntegral;

class Model
{
    string name;
    string owner;

    double temperature = double.nan;
    double topP = double.nan;
    long maxTokens = 4096;
    string[] stopSequences;
    double topK = double.nan;
    string system;
    bool stream = false;

    this(string name, string owner = null)
    {
        this.name = name;
        this.owner = owner is null ? "anthropic" : owner;
    }

    JSONValue messagesJSON(T)(T data)
    {
        JSONValue ret = JSONValue.emptyObject;
        ret["model"] = JSONValue(name);
        ret["max_tokens"] = JSONValue(maxTokens);

        if (system.length > 0) ret["system"] = JSONValue(system);
        if (temperature !is double.nan) ret["temperature"] = JSONValue(temperature);
        if (topP !is double.nan) ret["top_p"] = JSONValue(topP);
        if (topK !is double.nan) ret["top_k"] = JSONValue(topK);
        if (stopSequences.length > 0)
        {
            JSONValue arr = JSONValue.emptyArray;
            foreach (s; stopSequences) arr.array ~= JSONValue(s);
            ret["stop_sequences"] = arr;
        }
        if (stream) ret["stream"] = JSONValue(stream);

        JSONValue messages = JSONValue.emptyArray;
        static if (is(T == JSONValue))
        {
            if (data.type == JSONType.array)
            {
                foreach (m; data.array)
                    messages.array ~= toClaudeMessage(m);
            }
            else
                messages.array ~= userMessage(data);
        }
        else
            messages.array ~= userMessage(data.toJSON());
            
        ret["messages"] = messages;
        return ret;
    }

    private static JSONValue toClaudeMessage(JSONValue m)
    {
        JSONValue obj = JSONValue.emptyObject;
        if ("role" in m) obj["role"] = m["role"];
        if ("content" in m) obj["content"] = m["content"];
        return obj;
    }

    private static JSONValue userMessage(JSONValue content)
    {
        JSONValue obj = JSONValue.emptyObject;
        obj["role"] = JSONValue("user");
        obj["content"] = content;
        return obj;
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

        if ("content" in json && json["content"].array.length > 0)
        {
            Choice choice;
            string textPart;
            string thinkingPart;
            foreach (block; json["content"].array)
            {
                if (block.type != JSONType.object || !("type" in block)) continue;
                string blockType = block["type"].str;
                if (blockType == "text" && "text" in block && !block["text"].isNull)
                    textPart ~= block["text"].str;
                else if (blockType == "thinking" && "thinking" in block && !block["thinking"].isNull)
                    thinkingPart ~= block["thinking"].str;
            }
            if (thinkingPart.length > 0 && textPart.length > 0)
                choice.data = Variant([textPart, thinkingPart]);
            else if (thinkingPart.length > 0)
                choice.data = Variant([cast(string)"", thinkingPart]);
            else
                choice.data = Variant(textPart);

            bool hasStopReason = ("stop_reason" in json && !json["stop_reason"].isNull);
            choice.finishReason = hasStopReason
                ? cast(FinishReason)json["stop_reason"].str
                : FinishReason.Unknown;
            choice.logProbs = float.nan;
            ret.choices ~= choice;
        }
        return ret;
    }

    override string toString() const
    {
        return "Model("~name~", "~owner~")";
    }
}
