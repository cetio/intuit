/// LiteLLM proxy router with dynamic model discovery over the `/v1/model/info` endpoint.
module intuit.router.litellm;

import intuit.context;
import intuit.exception : EndpointException;
import intuit.model;
import intuit.router.details;
import intuit.response;
import intuit.router;
import intuit.tool;
import conductor.http : Response, send;

import std.conv : to;
import std.json : JSONType, JSONValue, parseJSON;
import std.net.curl : HTTP;
import std.string : assumeUTF;

/**
 * Router backed by a LiteLLM proxy instance.
 *
 * For now this only exposes the proxy's model catalog via the
 * `/v1/model/info` endpoint.  Completions, embeddings, and streaming
 * are not yet wired through.
 */
class LiteLLM : IRouter
{
private:
    string _name;
    string _url;
    string _key;
    ToolRegistry _tools;
    Context _context;
    string _active;
    ModelConfig[string] _configs;
    HTTP _http;
    ModelDetails[string] _catalog;

public:
    /**
     * Constructs a LiteLLM router.
     *
     * Params:
     *  url = The base URL of the LiteLLM proxy, defaulting to localhost.
     *  key = Optional API key for the proxy.
     *  name = Display name for the router.
     */
    this(string url = "http://localhost:4000", string key = null, string name = "LiteLLM")
    {
        this._name = name;
        this._url = normalizeBaseUrl(url);
        this._key = key;
        this._http = HTTP();
        this._context.compactor = new Compactor();
    }

    override ref string name()
        => _name;

    override ref ToolRegistry tools()
        => _tools;

    override ref Context context()
        => _context;

    override string active()
        => _active;

    override void active(string modelName)
    {
        throw new Exception("LiteLLM router currently only supports catalog access.");
    }

    override ModelConfig config()
    {
        throw new Exception("LiteLLM router currently only supports catalog access.");
    }

    override ModelConfig config(string modelName)
    {
        throw new Exception("LiteLLM router currently only supports catalog access.");
    }

    override ModelConfig[] configs()
    {
        throw new Exception("LiteLLM router currently only supports catalog access.");
    }

    override ModelDetails[string] catalog()
    {
        if (_catalog.length == 0)
            refresh();
        return _catalog;
    }

    override JSONValue _completions(JSONValue payload)
    {
        throw new Exception("LiteLLM router currently only supports catalog access.");
    }

    override JSONValue _embeddings(JSONValue payload)
    {
        throw new Exception("LiteLLM router currently only supports catalog access.");
    }

    override CompletionStream _stream(JSONValue payload)
    {
        throw new Exception("LiteLLM router currently only supports catalog access.");
    }

    /// Re-fetches the model catalog.
    override void refresh()
    {
        JSONValue json = request(HTTP.Method.get, "model/info");
        _catalog = null;
        if ("data" in json && json["data"].type == JSONType.array)
        {
            foreach (item; json["data"].array)
            {
                ModelDetails details = parseDetails(item);
                if (details.id.length > 0)
                    _catalog[details.id] = details;
            }
        }
    }

private:
    JSONValue request(HTTP.Method method, string tail, JSONValue payload = JSONValue.init)
    {
        string target = resolve(tail);
        Response response;

        if (payload.type == JSONType.null_)
            response = send(_http, method, target, null, null, requestHeaders());
        else
        {
            response = send(
                _http,
                method,
                target,
                cast(const(ubyte)[])payload.toString(),
                "application/json",
                requestHeaders(),
            );
        }

        string content = response.content is null ? null : response.content.assumeUTF().idup;
        if (response.status < 200 || response.status >= 300)
            throw new EndpointException(method.to!string, target, response.status, response.reason, content);

        try
            return content.parseJSON();
        catch (Exception)
            throw new EndpointException(
                method.to!string,
                target,
                response.status,
                response.reason,
                content,
                "Endpoint returned invalid JSON.",
            );
    }

    /// Builds request headers including authorization.
    string[string] requestHeaders()
    {
        string[string] headers;
        if (_key !is null && _key.length > 0)
            headers["Authorization"] = "Bearer "~_key;
        return headers;
    }

    /// Constructs a full route from a path tail under the LiteLLM API prefix.
    string resolve(string tail)
    {
        while (tail.length > 0 && tail[0] == '/')
            tail = tail[1..$];
        return _url~"/v1/"~tail;
    }

    /// Normalizes a base URL by stripping trailing slashes and API suffixes.
    static string normalizeBaseUrl(string url)
    {
        string ret = url;
        while (ret.length > 0 && ret[$-1] == '/')
            ret = ret[0..$-1];
        if (ret.length >= 3 && ret[$-3..$] == "/v1")
            ret = ret[0..$-3];
        return ret;
    }

    /// Parses a single `/v1/model/info` entry into ModelDetails.
    static ModelDetails parseDetails(JSONValue item)
    {
        ModelDetails ret;
        if (item.type != JSONType.object)
            return ret;

        ret.id = "model_name" in item ? item["model_name"].str : null;

        JSONValue modelInfo;
        if ("model_info" in item && item["model_info"].type == JSONType.object)
            modelInfo = item["model_info"];

        ret.name = jsonString(modelInfo, "key");
        if (ret.name.length == 0)
            ret.name = ret.id;

        ret.contextLength = jsonSize(modelInfo, "max_input_tokens");
        if (ret.contextLength == 0)
            ret.contextLength = jsonSize(modelInfo, "max_tokens");

        ret.maxCompletionTokens = jsonSize(modelInfo, "max_output_tokens");
        if (ret.maxCompletionTokens == 0)
            ret.maxCompletionTokens = jsonSize(modelInfo, "max_tokens");

        ret.promptCost = jsonDouble(modelInfo, "input_cost_per_token");
        ret.completionCost = jsonDouble(modelInfo, "output_cost_per_token");

        string[] inputs;
        string[] outputs;

        inputs ~= "text";

        string mode = jsonString(modelInfo, "mode");
        if (mode == "chat" || mode == "completion")
            outputs ~= "text";
        else if (mode == "embedding")
            outputs ~= "embedding";
        else if (mode == "image_generation")
            outputs ~= "image";
        else if (mode == "audio_speech")
            outputs ~= "audio";
        else if (mode.length > 0)
            outputs ~= mode;

        if (jsonBool(modelInfo, "supports_vision"))
            inputs ~= "image";
        if (jsonBool(modelInfo, "supports_audio_input"))
            inputs ~= "audio";
        if (jsonBool(modelInfo, "supports_pdf_input"))
            inputs ~= "pdf";

        if (jsonBool(modelInfo, "supports_audio_output"))
            outputs ~= "audio";

        ret.inputModalities = inputs;
        ret.outputModalities = outputs;

        string[] parameters;
        if (jsonBool(modelInfo, "supports_function_calling"))
            parameters ~= "tools";
        if (jsonBool(modelInfo, "supports_tool_choice"))
            parameters ~= "tool_choice";
        if (jsonBool(modelInfo, "supports_response_schema"))
            parameters ~= "response_format";
        if (jsonBool(modelInfo, "supports_system_messages"))
            parameters ~= "system";
        if (jsonBool(modelInfo, "supports_reasoning"))
            parameters ~= "reasoning";
        if (jsonBool(modelInfo, "supports_prompt_caching"))
            parameters ~= "prompt_caching";
        if (jsonBool(modelInfo, "supports_web_search"))
            parameters ~= "web_search";

        ret.supportedParameters = parameters;

        return ret;
    }

    /// Reads a string field from a JSON object.
    static string jsonString(JSONValue obj, string key)
    {
        if (key !in obj || obj[key].type != JSONType.string)
            return null;
        return obj[key].str;
    }

    /// Reads a boolean field from a JSON object.
    static bool jsonBool(JSONValue obj, string key)
    {
        if (key !in obj)
            return false;
        JSONValue value = obj[key];
        if (value.type == JSONType.true_)
            return true;
        if (value.type == JSONType.false_)
            return false;
        return false;
    }

    /// Reads an integral field from a JSON object, accepting numbers and strings.
    static size_t jsonSize(JSONValue obj, string key)
    {
        if (key !in obj)
            return 0;

        JSONValue value = obj[key];
        switch (value.type)
        {
        case JSONType.integer:
            return cast(size_t)value.integer;
        case JSONType.uinteger:
            return cast(size_t)value.uinteger;
        case JSONType.float_:
            return cast(size_t)value.floating;
        case JSONType.string:
            try
                return value.str.to!size_t;
            catch (Exception)
                return 0;
        default:
            return 0;
        }
    }

    /// Reads a floating field from a JSON object, accepting numbers and strings.
    static double jsonDouble(JSONValue obj, string key)
    {
        if (key !in obj)
            return 0;

        JSONValue value = obj[key];
        switch (value.type)
        {
        case JSONType.float_:
            return value.floating;
        case JSONType.integer:
            return cast(double)value.integer;
        case JSONType.uinteger:
            return cast(double)value.uinteger;
        case JSONType.string:
            try
                return value.str.to!double;
            catch (Exception)
                return 0;
        default:
            return 0;
        }
    }
}
