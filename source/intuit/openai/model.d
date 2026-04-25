module intuit.openai.model;

import std.algorithm.searching : canFind;
import intuit.error : EndpointError;
import intuit.model;
import intuit.response;
import conductor.http : toJSON;
import std.conv : to;
import std.json : JSONValue, JSONType;
import std.math : isNaN;
import std.string : toLower;

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
    long[long] _logitBias;
    long _seed = 0;
    string _encodingFormat = "float";
    long _dimensions = 0;
    JSONValue _responseFormat;
    bool _hasResponseFormat;

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

    OpenAIModel responseFormat(JSONValue delegate(OpenAIModel) fn)
    { return responseFormat(fn(this)); }
    OpenAIModel responseFormat(JSONValue v)
    {
        _responseFormat = v;
        _hasResponseFormat = true;
        return this;
    }

    OpenAIModel jsonMode()
    {
        JSONValue format = JSONValue.emptyObject;
        format["type"] = JSONValue("json_object");
        return responseFormat(format);
    }

    OpenAIModel jsonSchema(string name, JSONValue schema, bool strict = true)
    {
        JSONValue format = JSONValue.emptyObject;
        format["type"] = JSONValue("json_schema");

        JSONValue spec = JSONValue.emptyObject;
        spec["name"] = JSONValue(name);
        spec["schema"] = schema;
        spec["strict"] = JSONValue(strict);
        format["json_schema"] = spec;
        return responseFormat(format);
    }

    override JSONValue completionsJSON(JSONValue input)
    {
        JSONValue ret = JSONValue.emptyObject;
        ret["model"] = JSONValue(_name);

        if (_maxTokens >= 0) ret["max_tokens"] = JSONValue(_maxTokens);
        if (!isNaN(_temperature)) ret["temperature"] = JSONValue(_temperature);
        if (!isNaN(_topP)) ret["top_p"] = JSONValue(_topP);
        if (_stop.length > 0)
        {
            JSONValue arr = JSONValue.emptyArray;
            foreach (s; _stop) arr.array ~= JSONValue(s);
            ret["stop"] = arr;
        }
        if (!isNaN(_presencePenalty)) ret["presence_penalty"] = JSONValue(_presencePenalty);
        if (!isNaN(_frequencyPenalty)) ret["frequency_penalty"] = JSONValue(_frequencyPenalty);
        if (_n > 1) ret["n"] = JSONValue(_n);
        if (_logitBias.length > 0)
        {
            JSONValue bias = JSONValue.emptyObject;
            foreach (k, v; _logitBias) bias[k.to!string] = JSONValue(v);
            ret["logit_bias"] = bias;
        }
        if (_seed > 0) ret["seed"] = JSONValue(_seed);
        if (_hasResponseFormat) ret["response_format"] = _responseFormat;

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
        ret.raw = json;
        checkError(json);

        if ("choices" in json && json["choices"].type == JSONType.array)
        {
            foreach (c; json["choices"].array)
            {
                Choice choice;
                choice.raw = c;
                JSONValue message = ("message" in c) ? c["message"] : (("delta" in c) ? c["delta"] : JSONValue.init);
                parseMessage(choice, message);
                choice.finishReason = parseFinishReason("finish_reason" in c ? c["finish_reason"] : JSONValue.init);

                if ("logprobs" in c && !c["logprobs"].isNull)
                    choice.logProbs = c["logprobs"];
                else if ("log_probs" in c && !c["log_probs"].isNull)
                    choice.logProbs = c["log_probs"];

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
    static void parseMessage(ref Choice choice, JSONValue message)
    {
        if (message.type != JSONType.object)
            return;

        if ("content" in message && !message["content"].isNull)
        {
            choice.content = message["content"];
            parseContent(message["content"], choice.text, choice.reasoning, false);
        }

        if ("reasoning" in message && !message["reasoning"].isNull)
            parseContent(message["reasoning"], choice.text, choice.reasoning, true);
    }

    static void parseContent(JSONValue value, ref string text, ref string reasoning, bool reasoningMode)
    {
        switch (value.type)
        {
        case JSONType.string:
            appendText(reasoningMode ? reasoning : text, value.str);
            return;

        case JSONType.array:
            foreach (part; value.array)
                parseContent(part, text, reasoning, reasoningMode);
            return;

        case JSONType.object:
            bool nextReasoning = reasoningMode;
            if ("type" in value && value["type"].type == JSONType.string)
                nextReasoning = isReasoningType(value["type"].str);

            if ("summary" in value && !value["summary"].isNull)
                parseContent(value["summary"], text, reasoning, true);
            if ("text" in value && !value["text"].isNull)
                parseContent(value["text"], text, reasoning, nextReasoning);
            if ("content" in value && !value["content"].isNull)
                parseContent(value["content"], text, reasoning, nextReasoning);
            if ("value" in value && !value["value"].isNull)
                parseContent(value["value"], text, reasoning, nextReasoning);
            return;

        default:
            return;
        }
    }

    static bool isReasoningType(string raw)
    {
        string kind = raw.toLower;
        return kind.canFind("reason")
            || kind.canFind("think")
            || kind.canFind("summary");
    }

    static void appendText(ref string target, string value)
    {
        if (value.length > 0)
            target ~= value;
    }

    static FinishReason parseFinishReason(JSONValue value)
    {
        if (value.type != JSONType.string)
            return FinishReason.Unknown;

        switch (value.str)
        {
        case "missing":
            return FinishReason.Missing;
        case "length":
            return FinishReason.Length;
        case "max_tokens":
            return FinishReason.Max_Tokens;
        case "content_filter":
            return FinishReason.ContentFilter;
        case "refusal":
            return FinishReason.Refusal;
        case "tool_call":
            return FinishReason.ToolCall;
        case "tool_use":
            return FinishReason.ToolUse;
        case "function_call":
            return FinishReason.FunctionCall;
        case "pause":
            return FinishReason.Pause;
        case "pause_turn":
            return FinishReason.PauseTurn;
        case "stop":
            return FinishReason.Stop;
        case "end_turn":
            return FinishReason.EndTurn;
        case "stop_sequence":
            return FinishReason.StopSequence;
        default:
            return FinishReason.Unknown;
        }
    }

    static void checkError(JSONValue json)
    {
        if ("error" !in json)
            return;

        if (json["error"].type == JSONType.string)
            throw new Exception(json["error"].str);
        else if (json["error"].type == JSONType.object && "message" in json["error"])
            throw new EndpointError("POST", "chat/completions", 0, "error", json["error"]["message"].str);
        else
            throw new EndpointError("POST", "chat/completions", 0, "error", json.toPrettyString());
    }
}
