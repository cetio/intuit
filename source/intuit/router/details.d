/// Universal model metadata shared across all routers.
module intuit.router.details;

/// Dynamic metadata for a single model, populated from provider catalogs.
struct ModelDetails
{
    /// The model slug, e.g. "openai/gpt-4o".
    string id;
    /// Human-readable display name.
    string name;
    /// Model description text.
    string description;
    /// Total context window in tokens; drives the compactor token limit.
    size_t contextLength;
    /// Maximum tokens the top provider can generate in a single response.
    size_t maxCompletionTokens;
    // TODO: Define enums for model capabilities.
    /// Supported input modalities, e.g. ["text", "image"].
    string[] inputModalities;
    /// Supported output modalities, e.g. ["text"].
    string[] outputModalities;
    /// OpenAI-compatible parameters the model accepts, e.g. ["tools", "temperature"].
    string[] supportedParameters;
    /// Cost in USD per input token.
    double promptCost;
    /// Cost in USD per output token.
    double completionCost;
}
