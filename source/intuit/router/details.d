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

enum ModelCapability : string
{
    Audio = "audio",
    CandidateCount = "n",
    FrequencyPenalty = "frequency_penalty",
    FunctionCall = "function_call",
    Functions = "functions",
    IncludeReasoning = "include_reasoning",
    LogitBias = "logit_bias",
    Logprobs = "logprobs",
    MaxCompletionTokens = "max_completion_tokens",
    MaxTokens = "max_tokens",
    MinP = "min_p",
    Modalities = "modalities",
    ParallelToolCalls = "parallel_tool_calls",
    PresencePenalty = "presence_penalty",
    PromptCaching = "prompt_caching",
    Reasoning = "reasoning",
    RepetitionPenalty = "repetition_penalty",
    ResponseFormat = "response_format",
    Seed = "seed",
    ServiceTier = "service_tier",
    Stop = "stop",
    Store = "store",
    StructuredOutputs = "structured_outputs",
    System = "system",
    Temperature = "temperature",
    ToolChoice = "tool_choice",
    Tools = "tools",
    TopA = "top_a",
    TopK = "top_k",
    TopLogprobs = "top_logprobs",
    TopP = "top_p",
    User = "user",
    Verbosity = "verbosity",
    WebSearch = "web_search",
    WebSearchOptions = "web_search_options",
}

/// Dynamic metadata for a single model, populated from provider catalogs.
// TODO: TPS and proper pricing support for multiple providers.
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
    /// Normalized OpenAI-compatible parameters and features supported by the model.
    ModelCapability[] capabilities;
    /// Cost in USD per input token.
    double promptCost;
    /// Cost in USD per output token.
    double completionCost;
}
