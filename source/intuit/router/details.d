/// Universal model metadata shared across all routers.
module intuit.router.details;

/// Input or output modality supported by a model.
enum Modality : string
{
    /// Plain text input or output.
    text = "text",
    /// Image input or output.
    image = "image",
    /// Audio input or output.
    audio = "audio",
    /// PDF document input.
    pdf = "pdf",
    /// Embedding vector output.
    embedding = "embedding",
}

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
    /// Supported input modalities, e.g. [Modality.text, Modality.image].
    Modality[] inputModalities;
    /// Supported output modalities, e.g. [Modality.text].
    Modality[] outputModalities;
    /// OpenAI-compatible parameters the model accepts, e.g. ["tools", "temperature"].
    string[] supportedParameters;
    /// Cost in USD per input token.
    double promptCost;
    /// Cost in USD per output token.
    double completionCost;
}
