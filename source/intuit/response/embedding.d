/// Embedding vector type.
module intuit.response.embedding;

import intuit.exception : EndpointException;

import std.json : JSONType, JSONValue;
import std.traits : isIntegral;

/// Wrapper around an embedding vector that implicitly converts to its value.
struct Embedding(T)
{
    alias value this;

    /// The embedding vector values.
    T[] value;
}

package(intuit):
JSONValue parseEmbeddingResponse(JSONValue json)
{
    /// Throws if the JSON response contains an exception field.
    if ("error" in json)
    {
        if (json["error"].type == JSONType.string)
            throw new EndpointException("POST", "embeddings", 0, "error", json["error"].str);
        else if (json["error"].type == JSONType.object && "message" in json["error"])
            throw new EndpointException(
                "POST",
                "embeddings",
                0,
                "error",
                json["error"]["message"].str,
            );
        else
            throw new EndpointException("POST", "embeddings", 0, "error", json.toPrettyString());
    }

    JSONValue ret = JSONValue.emptyArray;
    if ("data" in json && json["data"].type == JSONType.array)
    {
        foreach (item; json["data"].array)
        {
            if ("embedding" in item)
                ret.array ~= item["embedding"];
        }
    }
    return ret;
}

/// Converts a JSON array into a typed vector.
T[] parseEmbeddingVector(T)(JSONValue vector)
{
    if (vector.type != JSONType.array)
        return null;

    T[] ret = new T[](vector.array.length);
    foreach (i, value; vector.array)
    {
        static if (isIntegral!T)
        {
            if (value.type == JSONType.integer)
                ret[i] = cast(T)value.integer;
            else if (value.type == JSONType.uinteger)
                ret[i] = cast(T)value.uinteger;
            else if (value.type == JSONType.float_)
                ret[i] = cast(T)value.floating;
        }
        else
        {
            if (value.type == JSONType.float_)
                ret[i] = cast(T)value.floating;
            else if (value.type == JSONType.integer)
                ret[i] = cast(T)value.integer;
            else if (value.type == JSONType.uinteger)
                ret[i] = cast(T)value.uinteger;
        }
    }
    return ret;
}
