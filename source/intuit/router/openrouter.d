/// OpenRouter router with dynamic model discovery over the OpenAI-compatible API.
module intuit.router.openrouter;

import intuit.context;
import intuit.exception : EndpointException;
import intuit.model;
import intuit.router.details;
import intuit.provider.openai;
import intuit.provider.systemone : SystemOneModelConfig;
import intuit.router;
import intuit.tool;

import std.conv : to;
import std.json : JSONType, JSONValue;
import std.net.curl : HTTP;
import std.string : join;

/// OpenRouter router implementation.
class OpenRouter : IRouter
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
    /// HTTP-Referer header used to identify the app to OpenRouter.
    string referer;
    /// X-OpenRouter-Title header used to identify the app to OpenRouter.
    string title;
    /// Marketplace categories sent via X-OpenRouter-Categories header.
    string[] categories;
    /// Routing strategy, e.g. "fallback".
    string route;
    /// Message transforms, e.g. ["middle-out"].
    string[] transforms;
    /// Whether to request reasoning tokens in responses.
    bool includeReasoning;
    /// Provider preferences object merged into every request.
    JSONValue provider;
    /// Plugins array merged into chat requests.
    JSONValue plugins;

    /**
     * Constructs an OpenRouter router.
     *
     * The provided URL is used as-is and the caller is responsible for
     * supplying the correct base URL for the endpoint.
     *
     * Params:
     *  key = The OpenRouter API key.
     *  url = The base URL, defaulting to the public OpenRouter host.
     *  name = Display name for the router.
     */
    this(string key, string url = "https://openrouter.ai", string name = "OpenRouter")
    {
        this._name = name;
        this._url = url;
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
        // The catalog is huge and dynamic, so only fetch it once on demand. Aliases
        // like "openrouter/auto" or "~openai/gpt-latest" may be absent; we still allow
        // them and simply leave the compactor limit untouched when metadata is missing.
        // NOTE: the Auto Router resolves the real model per request in the response
        // "model" field. We could pin the compactor limit to that resolved model once
        // the response types expose it.
        if (modelName !in _catalog && _catalog.length == 0)
            refresh();

        _active = modelName;
        config(modelName);
        if (auto details = modelName in _catalog)
            _context.compactor.maxTokens = details.contextLength;
    }

    override ModelConfig config()
    {
        if (_active is null)
            throw new Exception("Router has no active model set.");
        return _configs[_active];
    }

    override ModelConfig config(string modelName)
    {
        if (auto found = modelName in _configs)
            return *found;
        ModelConfig ret = new SystemOneModelConfig(modelName);
        _configs[modelName] = ret;
        return ret;
    }

    override ModelConfig[] configs()
    {
        ModelConfig[] ret;
        foreach (cfg; _configs.byValue)
            ret ~= cfg;
        return ret;
    }

    override ModelDetails[string] catalog()
    {
        if (_catalog.length == 0)
            refresh();
        return _catalog;
    }

    override JSONValue _completions(JSONValue payload)
        => _http.request(
            HTTP.Method.post,
            _url~"/api/v1/chat/completions",
            buildHeaders(),
            decorate(payload),
        );

    override JSONValue _decisions(JSONValue payload)
    {
        if (provider.type != JSONType.null_)
            payload["provider"] = provider;

        return _http.request(
            HTTP.Method.post,
            _url~"/api/alpha/decisions",
            buildHeaders(),
            payload,
        );
    }

    override JSONValue _embeddings(JSONValue payload)
    {
        if (provider.type != JSONType.null_)
            payload["provider"] = provider;
        return _http.request(
            HTTP.Method.post,
            _url~"/api/v1/embeddings",
            buildHeaders(),
            payload,
        );
    }

    /// Re-fetches the model catalog.
    override void refresh()
    {
        JSONValue json = _http.request(
            HTTP.Method.get,
            _url~"/api/v1/models?output_modalities=all",
            buildHeaders(),
        );
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
    string[string] buildHeaders()
    {
        string[string] ret;
        ret["Content-Type"] = "application/json";
        if (_key.length > 0)
            ret["Authorization"] = "Bearer "~_key;
        if (referer.length > 0)
            ret["HTTP-Referer"] = referer;
        if (title.length > 0)
            ret["X-OpenRouter-Title"] = title;
        if (categories.length > 0)
            ret["X-OpenRouter-Categories"] = categories.join(",");
        return ret;
    }

    /// Injects OpenRouter-only chat fields into a request payload.
    JSONValue decorate(JSONValue payload)
    {
        if (route.length > 0)
            payload["route"] = JSONValue(route);
        if (transforms.length > 0)
        {
            JSONValue arr = JSONValue.emptyArray;
            foreach (transform; transforms)
                arr.array ~= JSONValue(transform);
            payload["transforms"] = arr;
        }
        if (plugins.type != JSONType.null_)
            payload["plugins"] = plugins;
        if (provider.type != JSONType.null_)
            payload["provider"] = provider;
        if (includeReasoning)
        {
            JSONValue reasoning = "reasoning" in payload && payload["reasoning"].type == JSONType.object
                ? payload["reasoning"]
                : JSONValue.emptyObject;
            reasoning["exclude"] = JSONValue(false);
            payload["reasoning"] = reasoning;
        }
        return payload;
    }

    /// Parses a single `/models` entry into ModelDetails.
    static ModelDetails parseDetails(JSONValue item)
    {
        ModelDetails ret;
        if (item.type != JSONType.object)
            return ret;

        ret.id = "id" in item ? item["id"].str : null;
        ret.name = "name" in item ? item["name"].str : null;
        ret.description = "description" in item ? item["description"].str : null;

        if ("context_length" in item && item["context_length"].type == JSONType.integer)
            ret.contextLength = cast(size_t)item["context_length"].integer;

        if ("top_provider" in item && item["top_provider"].type == JSONType.object)
        {
            JSONValue topProvider = item["top_provider"];
            if (ret.contextLength == 0 && "context_length" in topProvider
                && topProvider["context_length"].type == JSONType.integer)
                ret.contextLength = cast(size_t)topProvider["context_length"].integer;
            if ("max_completion_tokens" in topProvider
                && topProvider["max_completion_tokens"].type == JSONType.integer)
                ret.maxCompletionTokens = cast(size_t)topProvider["max_completion_tokens"].integer;
        }

        if ("architecture" in item && item["architecture"].type == JSONType.object)
        {
            JSONValue architecture = item["architecture"];
            if ("input_modalities" in architecture
                && architecture["input_modalities"].type == JSONType.array)
            {
                foreach (entry; architecture["input_modalities"].array)
                {
                    if (entry.type != JSONType.string)
                        continue;
                    switch (entry.str)
                    {
                        case "text":
                            ret.inputModalities ~= Modality.Text;
                            break;
                        case "image":
                            ret.inputModalities ~= Modality.Image;
                            break;
                        case "audio":
                            ret.inputModalities ~= Modality.Audio;
                            break;
                        case "pdf":
                            ret.inputModalities ~= Modality.Pdf;
                            break;
                        case "embedding":
                            ret.inputModalities ~= Modality.Embedding;
                            break;
                        default:
                            break;
                    }
                }
            }

            if ("output_modalities" in architecture
                && architecture["output_modalities"].type == JSONType.array)
            {
                foreach (entry; architecture["output_modalities"].array)
                {
                    if (entry.type != JSONType.string)
                        continue;
                    switch (entry.str)
                    {
                        case "text":
                            ret.outputModalities ~= Modality.Text;
                            break;
                        case "image":
                            ret.outputModalities ~= Modality.Image;
                            break;
                        case "audio":
                            ret.outputModalities ~= Modality.Audio;
                            break;
                        case "pdf":
                            ret.outputModalities ~= Modality.Pdf;
                            break;
                        case "embedding":
                            ret.outputModalities ~= Modality.Embedding;
                            break;
                        case "decisions":
                            ret.outputModalities ~= Modality.Decisions;
                            break;
                        default:
                            break;
                    }
                }
            }
        }

        if ("supported_parameters" in item && item["supported_parameters"].type == JSONType.array)
        {
            foreach (entry; item["supported_parameters"].array)
            {
                if (entry.type != JSONType.string)
                    continue;
                switch (entry.str)
                {
                    case "audio":
                        ret.capabilities ~= ModelCapability.Audio;
                        break;
                    case "n":
                        ret.capabilities ~= ModelCapability.CandidateCount;
                        break;
                    case "frequency_penalty":
                        ret.capabilities ~= ModelCapability.FrequencyPenalty;
                        break;
                    case "function_call":
                        ret.capabilities ~= ModelCapability.FunctionCall;
                        break;
                    case "functions":
                        ret.capabilities ~= ModelCapability.Functions;
                        break;
                    case "include_reasoning":
                        ret.capabilities ~= ModelCapability.IncludeReasoning;
                        break;
                    case "logit_bias":
                        ret.capabilities ~= ModelCapability.LogitBias;
                        break;
                    case "logprobs":
                        ret.capabilities ~= ModelCapability.Logprobs;
                        break;
                    case "max_completion_tokens":
                        ret.capabilities ~= ModelCapability.MaxCompletionTokens;
                        break;
                    case "max_tokens":
                        ret.capabilities ~= ModelCapability.MaxTokens;
                        break;
                    case "min_p":
                        ret.capabilities ~= ModelCapability.MinP;
                        break;
                    case "modalities":
                        ret.capabilities ~= ModelCapability.Modalities;
                        break;
                    case "parallel_tool_calls":
                        ret.capabilities ~= ModelCapability.ParallelToolCalls;
                        break;
                    case "presence_penalty":
                        ret.capabilities ~= ModelCapability.PresencePenalty;
                        break;
                    case "prompt_caching":
                        ret.capabilities ~= ModelCapability.PromptCaching;
                        break;
                    case "reasoning":
                        ret.capabilities ~= ModelCapability.Reasoning;
                        break;
                    case "repetition_penalty":
                        ret.capabilities ~= ModelCapability.RepetitionPenalty;
                        break;
                    case "response_format":
                        ret.capabilities ~= ModelCapability.ResponseFormat;
                        break;
                    case "seed":
                        ret.capabilities ~= ModelCapability.Seed;
                        break;
                    case "service_tier":
                        ret.capabilities ~= ModelCapability.ServiceTier;
                        break;
                    case "stop":
                        ret.capabilities ~= ModelCapability.Stop;
                        break;
                    case "store":
                        ret.capabilities ~= ModelCapability.Store;
                        break;
                    case "structured_outputs":
                        ret.capabilities ~= ModelCapability.StructuredOutputs;
                        break;
                    case "system":
                        ret.capabilities ~= ModelCapability.System;
                        break;
                    case "temperature":
                        ret.capabilities ~= ModelCapability.Temperature;
                        break;
                    case "tool_choice":
                        ret.capabilities ~= ModelCapability.ToolChoice;
                        break;
                    case "tools":
                        ret.capabilities ~= ModelCapability.Tools;
                        break;
                    case "top_a":
                        ret.capabilities ~= ModelCapability.TopA;
                        break;
                    case "top_k":
                        ret.capabilities ~= ModelCapability.TopK;
                        break;
                    case "top_logprobs":
                        ret.capabilities ~= ModelCapability.TopLogprobs;
                        break;
                    case "top_p":
                        ret.capabilities ~= ModelCapability.TopP;
                        break;
                    case "user":
                        ret.capabilities ~= ModelCapability.User;
                        break;
                    case "verbosity":
                        ret.capabilities ~= ModelCapability.Verbosity;
                        break;
                    case "web_search":
                        ret.capabilities ~= ModelCapability.WebSearch;
                        break;
                    case "web_search_options":
                        ret.capabilities ~= ModelCapability.WebSearchOptions;
                        break;
                    default:
                        break;
                }
            }
        }

        if ("pricing" in item && item["pricing"].type == JSONType.object)
        {
            JSONValue pricing = item["pricing"];
            if ("prompt" in pricing)
                ret.promptCost = parseCost(pricing["prompt"]);
            if ("completion" in pricing)
                ret.completionCost = parseCost(pricing["completion"]);
        }

        return ret;
    }

    static double parseCost(JSONValue value)
    {
        double ret;
        switch (value.type)
        {
            case JSONType.float_:
                ret = value.floating;
                break;
            case JSONType.integer:
                ret = cast(double)value.integer;
                break;
            case JSONType.uinteger:
                ret = cast(double)value.uinteger;
                break;
            case JSONType.string:
                try
                {
                    ret = value.str.to!double;
                }
                catch (Exception)
                {
                    return ret;
                }
                break;
            default:
                break;
        }
        return ret;
    }
}
