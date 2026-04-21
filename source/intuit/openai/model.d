module intuit.openai.model;

import intuit.model;
import intuit.response;
import intuit.utils;
import std.conv : to;
import std.json : JSONValue, JSONType;

class OpenAIModel : IModel
{
    string _name;
    string owner;

    double _temperature = double.nan;
    double _topP = double.nan;
    long _maxTokens = -1;
    string[] _stop;
    double _presencePenalty = double.nan;
    double _frequencyPenalty = double.nan;
    long _n = 1;
    bool _stream = false;
    long[long] _logitBias;
    long _seed = 0;
    string _encodingFormat = "float";
    long _dimensions = 0;

    this(string name, string owner = null)
    {
        this._name = name;
        this.owner = owner;
    }

    override string name() => _name;

    override string toString() const
        => "OpenAIModel("~_name~", "~owner~")";

    // Per-setting lambda accessors (chainable). Each has a plain-value overload.

    OpenAIModel temperature(double delegate(OpenAIModel) fn)
    { _temperature = fn(this); return this; }
    OpenAIModel temperature(double v)
    { _temperature = v; return this; }

    OpenAIModel topP(double delegate(OpenAIModel) fn)
    { _topP = fn(this); return this; }
    OpenAIModel topP(double v)
    { _topP = v; return this; }

    OpenAIModel maxTokens(long delegate(OpenAIModel) fn)
    { _maxTokens = fn(this); return this; }
    OpenAIModel maxTokens(long v)
    { _maxTokens = v; return this; }

    OpenAIModel stop(string[] delegate(OpenAIModel) fn)
    { _stop = fn(this); return this; }
    OpenAIModel stop(string[] v)
    { _stop = v; return this; }

    OpenAIModel presencePenalty(double delegate(OpenAIModel) fn)
    { _presencePenalty = fn(this); return this; }
    OpenAIModel presencePenalty(double v)
    { _presencePenalty = v; return this; }

    OpenAIModel frequencyPenalty(double delegate(OpenAIModel) fn)
    { _frequencyPenalty = fn(this); return this; }
    OpenAIModel frequencyPenalty(double v)
    { _frequencyPenalty = v; return this; }

    OpenAIModel n(long delegate(OpenAIModel) fn)
    { _n = fn(this); return this; }
    OpenAIModel n(long v)
    { _n = v; return this; }

    OpenAIModel stream(bool delegate(OpenAIModel) fn)
    { _stream = fn(this); return this; }
    OpenAIModel stream(bool v)
    { _stream = v; return this; }

    OpenAIModel logitBias(long[long] delegate(OpenAIModel) fn)
    { _logitBias = fn(this); return this; }
    OpenAIModel logitBias(long[long] v)
    { _logitBias = v; return this; }

    OpenAIModel seed(long delegate(OpenAIModel) fn)
    { _seed = fn(this); return this; }
    OpenAIModel seed(long v)
    { _seed = v; return this; }

    OpenAIModel encodingFormat(string delegate(OpenAIModel) fn)
    { _encodingFormat = fn(this); return this; }
    OpenAIModel encodingFormat(string v)
    { _encodingFormat = v; return this; }

    OpenAIModel dimensions(long delegate(OpenAIModel) fn)
    { _dimensions = fn(this); return this; }
    OpenAIModel dimensions(long v)
    { _dimensions = v; return this; }

    override JSONValue completionsJSON(JSONValue input)
    {
        JSONValue ret = JSONValue.emptyObject;
        ret["model"] = JSONValue(_name);
        ret["max_tokens"] = JSONValue(_maxTokens);

        if (_temperature !is double.nan) ret["temperature"] = JSONValue(_temperature);
        if (_topP !is double.nan) ret["top_p"] = JSONValue(_topP);
        if (_stop.length > 0)
        {
            JSONValue arr = JSONValue.emptyArray;
            foreach (s; _stop) arr.array ~= JSONValue(s);
            ret["stop"] = arr;
        }
        if (_presencePenalty !is double.nan) ret["presence_penalty"] = JSONValue(_presencePenalty);
        if (_frequencyPenalty !is double.nan) ret["frequency_penalty"] = JSONValue(_frequencyPenalty);
        if (_n > 1) ret["n"] = JSONValue(_n);
        if (_stream) ret["stream"] = JSONValue(_stream);
        if (_logitBias.length > 0)
        {
            JSONValue bias = JSONValue.emptyObject;
            foreach (k, v; _logitBias) bias[k.to!string] = JSONValue(v);
            ret["logit_bias"] = bias;
        }
        if (_seed > 0) ret["seed"] = JSONValue(_seed);

        if (input.type == JSONType.array)
        {
            ret["messages"] = input;
        }
        else
        {
            ret["messages"] = JSONValue.emptyArray;
            JSONValue msg = JSONValue.emptyObject;
            msg["role"] = JSONValue("user");
            msg["content"] = input;
            ret["messages"].array ~= msg;
        }
        return ret;
    }

    override JSONValue embeddingsJSON(JSONValue input)
    {
        JSONValue ret = JSONValue.emptyObject;
        ret["model"] = JSONValue(_name);
        if (_encodingFormat != "float") ret["encoding_format"] = JSONValue(_encodingFormat);
        if (_dimensions > 0) ret["dimensions"] = JSONValue(_dimensions);
        ret["input"] = input;
        return ret;
    }

    override Completion parseCompletions(JSONValue json)
    {
        Completion ret;
        checkError(json);

        if ("choices" in json)
        {
            foreach (c; json["choices"].array)
            {
                Choice choice;
                JSONValue msg = ("message" in c) ? c["message"] : c["delta"];

                string content = ("content" in msg && !msg["content"].isNull)
                    ? msg["content"].str
                    : cast(string)null;
                import std.variant : Variant;
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

    override JSONValue parseEmbeddings(JSONValue json)
    {
        checkError(json);

        JSONValue ret = JSONValue.emptyArray;
        if ("data" in json && json["data"].type == JSONType.array)
        {
            foreach (item; json["data"].array)
            {
                if ("embedding" in item)
                    ret.array ~= item["embedding"];
            }
        }
        return ret;
    }

private:
    static void checkError(JSONValue json)
    {
        if ("error" !in json)
            return;

        if (json["error"].type == JSONType.string)
            throw new Exception(json["error"].str);
        else if (json["error"].type == JSONType.object && "message" in json["error"])
            throw new Exception(json["error"]["message"].str);
        else
            throw new Exception("Critical error parsing error JSON "~json.toPrettyString());
    }
}
