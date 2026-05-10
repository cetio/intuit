module intuit.openai.model;

import std.algorithm.searching : canFind;
import intuit.error : EndpointError;
import intuit.model;
import intuit.response;
import intuit.tool;
import conductor.http : toJSON;
import std.conv : to;
import std.json : JSONValue, JSONType, parseJSON;
import std.math : isNaN;
import std.string : toLower;

class OpenAIModel : IModel
{
private:
    string _name;
    string _owner;

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

public:
    this(string name, string owner = null)
    {
        this._name = name;
        this._owner = owner;
    }

    override ref string name()
        => _name;

    override ref string owner()
        => _owner;

    override string toString() const
        => "OpenAIModel("~_name~", "~_owner~")";

    // Chainable and accessor properties for model settings.

    ref double temperature()
        => _temperature;

    OpenAIModel temperature(double val)
    {
        _temperature = val;
        return this;
    }

    ref double topP()
        => _topP;

    OpenAIModel topP(double val)
    {
        _topP = val;
        return this;
    }

    ref long maxTokens()
        => _maxTokens;

    OpenAIModel maxTokens(long val)
    {
        _maxTokens = val;
        return this;
    }


    ref string[] stop()
        => _stop;

    OpenAIModel stop(string[] val)
    {
        _stop = val;
        return this;
    }

    ref double presencePenalty()
        => _presencePenalty;

    OpenAIModel presencePenalty(double val)
    {
        _presencePenalty = val;
        return this;
    }

    ref double frequencyPenalty()
        => _frequencyPenalty;

    OpenAIModel frequencyPenalty(double val)
    {
        _frequencyPenalty = val;
        return this;
    }

    ref long n()
        => _n;

    OpenAIModel n(long val)
    {
        _n = val;
        return this;
    }

    ref long[long] logitBias()
        => _logitBias;

    OpenAIModel logitBias(long[long] val)
    {
        _logitBias = val;
        return this;
    }

    ref long seed()
        => _seed;

    OpenAIModel seed(long val)
    {
        _seed = val;
        return this;
    }

    ref string encodingFormat()
        => _encodingFormat;

    OpenAIModel encodingFormat(string val)
    {
        _encodingFormat = val;
        return this;
    }

    ref long dimensions()
        => _dimensions;

    OpenAIModel dimensions(long val)
    {
        _dimensions = val;
        return this;
    }

    OpenAIModel responseFormat(JSONValue val)
    {
        _responseFormat = val;
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

    override JSONValue completionsJSON(JSONValue input, ToolRegistry tools = ToolRegistry.init)
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

        Tool[] toolList = tools.list();
        if (toolList.length > 0)
        {
            JSONValue toolsArray = JSONValue.emptyArray;
            foreach (tool; toolList)
            {
                JSONValue toolObj = JSONValue.emptyObject;
                toolObj["type"] = JSONValue("function");
                JSONValue functionObj = JSONValue.emptyObject;
                functionObj["name"] = JSONValue(tool.name);
                functionObj["parameters"] = tool.schema;
                toolObj["function"] = functionObj;
                toolsArray.array ~= toolObj;
            }
            ret["tools"] = toolsArray;
        }

        if (input.type == JSONType.array)
            ret["messages"] = input;
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
            foreach (entry; json["choices"].array)
            {
                Choice choice;
                choice.raw = entry;
                JSONValue msg = ("message" in entry)
                    ? entry["message"]
                    : (("delta" in entry) ? entry["delta"] : JSONValue.init);
                parseMessage(choice, msg);
                choice.finishReason = parseFinishReason(
                    "finish_reason" in entry ? entry["finish_reason"] : JSONValue.init
                );

                if ("logprobs" in entry && !entry["logprobs"].isNull)
                    choice.logProbs = entry["logprobs"];
                else if ("log_probs" in entry && !entry["log_probs"].isNull)
                    choice.logProbs = entry["log_probs"];

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

        if ("tool_calls" in message && message["tool_calls"].type == JSONType.array)
        {
            foreach (toolCall; message["tool_calls"].array)
            {
                ToolCall tc;
                tc.id = ("id" in toolCall) ? toolCall["id"].str : "";
                if ("function" in toolCall && toolCall["function"].type == JSONType.object)
                {
                    JSONValue func = toolCall["function"];
                    tc.name = ("name" in func) ? func["name"].str : "";
                    if ("arguments" in func && func["arguments"].type == JSONType.string)
                        tc.arguments = parseJSON(func["arguments"].str);
                }
                choice.toolCalls ~= tc;
            }
        }
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
