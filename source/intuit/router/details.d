/// Universal model metadata shared across all routers.
module intuit.router.details;

/// Input or output modality supported by a model.
enum Modality : string
{
    /// Plain text input or output.
    Text = "text",
    /// Image input or output.
    Image = "image",
    /// Audio input or output.
    Audio = "audio",
    /// PDF document input.
    Pdf = "pdf",
    /// Embedding vector output.
    Embedding = "embedding",
    /// Structured decision output.
    Decisions = "decisions",
}

enum ModelCapability : string
{
    /// Supports audio processing. This does NOT guarantee audio input/output support; check modalities for that.
    Audio = "audio",
    /// Supports generating multiple candidate responses.
    CandidateCount = "n",
    /// Supports frequency penalty for token generation.
    FrequencyPenalty = "frequency_penalty",
    /// Supports function calling.
    FunctionCall = "function_call",
    /// Functions available for calling.
    Functions = "functions",
    /// Supports including reasoning in the response.
    IncludeReasoning = "include_reasoning",
    /// Supports logit bias for token generation.
    LogitBias = "logit_bias",
    /// Supports log probabilities for token generation.
    Logprobs = "logprobs",
    /// Supports maximum completion tokens.
    MaxCompletionTokens = "max_completion_tokens",
    /// Supports maximum tokens.
    MaxTokens = "max_tokens",
    /// Supports minimum probability for token generation.
    MinP = "min_p",
    /// Supports at least 1 non-essential modality.
    Modalities = "modalities",
    /// Supports parallel tool calls.
    ParallelToolCalls = "parallel_tool_calls",
    /// Supports presence penalty for token generation.
    PresencePenalty = "presence_penalty",
    /// Supports prompt caching.
    PromptCaching = "prompt_caching",
    /// Supports reasoning.
    Reasoning = "reasoning",
    /// Supports repetition penalty for token generation.
    RepetitionPenalty = "repetition_penalty",
    /// Supports response format.
    ResponseFormat = "response_format",
    /// Supports random seed for reproducible generation.
    Seed = "seed",
    /// Supports service tier (e.g., "standard", "priority").
    ServiceTier = "service_tier",
    /// Supports stop sequences for generation.
    Stop = "stop",
    /// Supports storage.
    Store = "store",
    /// Supports structured outputs.
    StructuredOutputs = "structured_outputs",
    /// Supports system prompt.
    System = "system",
    /// Supports temperature for generation.
    Temperature = "temperature",
    /// Supports tool choice.
    ToolChoice = "tool_choice",
    /// Supports tools.
    Tools = "tools",
    /// Supports top-A sampling.
    TopA = "top_a",
    /// Supports top-K sampling.
    TopK = "top_k",
    /// Supports top-logprobs.
    TopLogprobs = "top_logprobs",
    /// Supports top-P sampling.
    TopP = "top_p",
    /// Supports user identifier.
    User = "user",
    /// Supports verbosity level.
    Verbosity = "verbosity",
    /// Supports web search.
    WebSearch = "web_search",
    /// Supports web search options.
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
    /// Supported input modalities, e.g. [Modality.Text, Modality.Image].
    Modality[] inputModalities;
    /// Supported output modalities, e.g. [Modality.Text].
    Modality[] outputModalities;
    /// Normalized OpenAI-compatible parameters and features supported by the model.
    ModelCapability[] capabilities;
    /// Cost in USD per input token.
    double promptCost;
    /// Cost in USD per output token.
    double completionCost;
}
