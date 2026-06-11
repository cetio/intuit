/// Static model metadata catalog for the local router.
module intuit.router.local.catalog;

/// Router-defined metadata describing a single model.
struct LocalModelDetails
{
    /// The model identifier, e.g. "gpt-4o".
    string name;
    /// The provider that serves the model.
    string provider;
    /// Total context window in tokens; drives the compactor token limit.
    size_t contextWindow;
    /// Maximum tokens the model can generate in a single response.
    size_t maxOutputTokens;
    /// Cost in USD per one million input tokens.
    double inputCostPer1M;
    /// Cost in USD per one million output tokens.
    double outputCostPer1M;
    /// Advertised capabilities, e.g. ["tools", "vision"].
    string[] capabilities;
}

/// The known models indexed by name.
__gshared LocalModelDetails[string] catalog;

shared static this()
{
    catalog["gpt-4o"] = LocalModelDetails(
        "gpt-4o",
        "openai",
        128_000,
        16_384,
        2.50,
        10.00,
        ["tools", "vision"],
    );
    catalog["claude-3-5-sonnet-20241022"] = LocalModelDetails(
        "claude-3-5-sonnet-20241022",
        "anthropic",
        200_000,
        8_192,
        3.00,
        15.00,
        ["tools", "vision"],
    );
    catalog["qwen2.5-72b-instruct"] = LocalModelDetails(
        "qwen2.5-72b-instruct",
        "qwen",
        32_768,
        8_192,
        0.00,
        0.00,
        ["tools"],
    );
}

/// Returns true if a model is present in the catalog.
bool known(string name)
    => (name in catalog) !is null;

/**
 * Looks up the details for a model by name.
 *
 * Params:
 *  name = The model name.
 *
 * Returns:
 *  The matching LocalModelDetails.
 *
 * Throws:
 *  Exception if the model is not in the catalog.
 */
LocalModelDetails lookup(string name)
{
    if (auto details = name in catalog)
        return *details;
    throw new Exception("Model not found in catalog: "~name);
}
