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

        if ("key" in modelInfo && modelInfo["key"].type == JSONType.string)
            ret.name = modelInfo["key"].str;
        if (ret.name.length == 0)
            ret.name = ret.id;

        if ("max_input_tokens" in modelInfo && modelInfo["max_input_tokens"].type == JSONType.integer)
            ret.contextLength = cast(size_t)modelInfo["max_input_tokens"].integer;
        if (ret.contextLength == 0 && "max_tokens" in modelInfo
            && modelInfo["max_tokens"].type == JSONType.integer)
            ret.contextLength = cast(size_t)modelInfo["max_tokens"].integer;

        if ("max_output_tokens" in modelInfo && modelInfo["max_output_tokens"].type == JSONType.integer)
            ret.maxCompletionTokens = cast(size_t)modelInfo["max_output_tokens"].integer;
        if (ret.maxCompletionTokens == 0 && "max_tokens" in modelInfo
            && modelInfo["max_tokens"].type == JSONType.integer)
            ret.maxCompletionTokens = cast(size_t)modelInfo["max_tokens"].integer;

        if ("input_cost_per_token" in modelInfo && modelInfo["input_cost_per_token"].type == JSONType.float_)
            ret.promptCost = modelInfo["input_cost_per_token"].floating;
        if ("output_cost_per_token" in modelInfo && modelInfo["output_cost_per_token"].type == JSONType.float_)
            ret.completionCost = modelInfo["output_cost_per_token"].floating;

        string[] inputs;
        string[] outputs;

        inputs ~= "text";

        string mode;
        if ("mode" in modelInfo && modelInfo["mode"].type == JSONType.string)
            mode = modelInfo["mode"].str;
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

        if ("supports_vision" in modelInfo && modelInfo["supports_vision"].type == JSONType.true_)
            inputs ~= "image";
        if ("supports_audio_input" in modelInfo && modelInfo["supports_audio_input"].type == JSONType.true_)
            inputs ~= "audio";
        if ("supports_pdf_input" in modelInfo && modelInfo["supports_pdf_input"].type == JSONType.true_)
            inputs ~= "pdf";

        if ("supports_audio_output" in modelInfo && modelInfo["supports_audio_output"].type == JSONType.true_)
            outputs ~= "audio";

        ret.inputModalities = inputs;
        ret.outputModalities = outputs;

        // TODO: This is terrible.
        string[] parameters;
        if ("supports_function_calling" in modelInfo && modelInfo["supports_function_calling"].type == JSONType.true_)
            parameters ~= "tools";
        if ("supports_tool_choice" in modelInfo && modelInfo["supports_tool_choice"].type == JSONType.true_)
            parameters ~= "tool_choice";
        if ("supports_response_schema" in modelInfo && modelInfo["supports_response_schema"].type == JSONType.true_)
            parameters ~= "response_format";
        if ("supports_system_messages" in modelInfo && modelInfo["supports_system_messages"].type == JSONType.true_)
            parameters ~= "system";
        if ("supports_reasoning" in modelInfo && modelInfo["supports_reasoning"].type == JSONType.true_)
            parameters ~= "reasoning";
        if ("supports_prompt_caching" in modelInfo && modelInfo["supports_prompt_caching"].type == JSONType.true_)
            parameters ~= "prompt_caching";
        if ("supports_web_search" in modelInfo && modelInfo["supports_web_search"].type == JSONType.true_)
            parameters ~= "web_search";

        ret.supportedParameters = parameters;

        return ret;
    }
}
