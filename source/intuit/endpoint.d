module intuit.endpoint;

import intuit.context;
import intuit.model;
import intuit.response;
import intuit.utils;
import std.json : JSONValue, JSONType;
import std.traits : isArray, isIntegral;

interface IEndpoint
{
    string name();
    void name(string value);
    string url();
    void url(string value);
    void key(string value);

    IModel[] available();

    JSONValue _completions(IModel model, JSONValue payload);
    JSONValue _embeddings(IModel model, JSONValue payload);
}

Completion completions(E, M, D)(E ep, M model, D data)
    if (is(E : IEndpoint) && is(M : IModel))
{
    static if (is(D == Context))
        JSONValue input = data.messages;
    else
        JSONValue input = data.toJSON();

    JSONValue payload = model.completionsJSON(input);
    JSONValue resp = ep._completions(model, payload);
    return model.parseCompletions(resp);
}

Embedding!T embeddings(T = float, E, M, D)(E ep, M model, D data)
    if (is(E : IEndpoint) && is(M : IModel)
        && (is(D == string) || !isArray!D))
{
    JSONValue payload = model.embeddingsJSON(data.toJSON());
    JSONValue resp = ep._embeddings(model, payload);
    JSONValue arr = model.parseEmbeddings(resp);

    Embedding!T ret;
    if (arr.type == JSONType.array && arr.array.length > 0)
        ret.value = toVector!T(arr.array[0]);
    return ret;
}

Embedding!T[] embeddings(T = float, E, M, D)(E ep, M model, D data)
    if (is(E : IEndpoint) && is(M : IModel)
        && isArray!D && !is(D == string))
{
    JSONValue payload = model.embeddingsJSON(data.toJSON());
    JSONValue resp = ep._embeddings(model, payload);
    JSONValue arr = model.parseEmbeddings(resp);

    Embedding!T[] ret;
    if (arr.type == JSONType.array)
    {
        ret.length = arr.array.length;
        foreach (i, v; arr.array)
            ret[i].value = toVector!T(v);
    }
    return ret;
}

private T[] toVector(T)(JSONValue arr)
{
    if (arr.type != JSONType.array)
        return null;

    T[] ret = new T[](arr.array.length);
    foreach (i, v; arr.array)
    {
        static if (isIntegral!T)
        {
            if (v.type == JSONType.integer)
                ret[i] = cast(T)v.integer;
            else if (v.type == JSONType.uinteger)
                ret[i] = cast(T)v.uinteger;
            else
                ret[i] = cast(T)v.floating;
        }
        else
        {
            if (v.type == JSONType.float_)
                ret[i] = cast(T)v.floating;
            else if (v.type == JSONType.integer)
                ret[i] = cast(T)v.integer;
            else if (v.type == JSONType.uinteger)
                ret[i] = cast(T)v.uinteger;
        }
    }
    return ret;
}
