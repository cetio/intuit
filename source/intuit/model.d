/// Base model interface for LLM providers.
module intuit.model;

import intuit.response;
import intuit.tool;
import std.json : JSONValue;

/// Interface implemented by all LLM model types.
interface IModel
{
    /// Gets the model name.
    ref string name();
    /// Gets the model owner.
    ref string owner();

    /**
     * Builds the JSON payload for a completions request.
     *
     * Params:
     *  input = The input messages or raw content.
     *  tools = Registered tools to include in the request.
     *
     * Returns:
     *  The JSON payload to send to the endpoint.
     */
    JSONValue completionsJSON(JSONValue input, ToolRegistry tools = ToolRegistry.init);

    /**
     * Builds the JSON payload for an embeddings request.
     *
     * Params:
     *  input = The input data to embed.
     *
     * Returns:
     *  The JSON payload to send to the endpoint.
     */
    JSONValue embeddingsJSON(JSONValue input);

    /**
     * Parses a raw completions response into a Completion struct.
     *
     * Params:
     *  response = The raw JSON response from the endpoint.
     *
     * Returns:
     *  The parsed Completion.
     */
    Completion parseCompletions(JSONValue response);

    /**
     * Parses a raw embeddings response into a JSON array of embeddings.
     *
     * Params:
     *  response = The raw JSON response from the endpoint.
     *
     * Returns:
     *  A JSON array containing the embedding vectors.
     */
    JSONValue parseEmbeddings(JSONValue response);
}
