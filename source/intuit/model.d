module intuit.model;

import intuit.response;
import std.json : JSONValue;

interface IModel
{
    string name();

    JSONValue completionsJSON(JSONValue input);
    JSONValue embeddingsJSON(JSONValue input);

    Completion parseCompletions(JSONValue response);
    JSONValue parseEmbeddings(JSONValue response);
}
