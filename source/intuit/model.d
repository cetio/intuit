module intuit.model;

import intuit.response;
import intuit.tool;
import std.json : JSONValue;

interface IModel
{
    ref string name();
    ref string owner();

    JSONValue completionsJSON(JSONValue input, ToolRegistry tools = ToolRegistry.init);
    JSONValue embeddingsJSON(JSONValue input);

    Completion parseCompletions(JSONValue response);
    JSONValue parseEmbeddings(JSONValue response);
}
