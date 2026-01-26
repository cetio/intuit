module intuit.utils;

public import std.json : JSONValue, JSONOptions, JSONType, JSONException, parseJSON;
static import std.json;
import std.traits;
import std.net.curl : HTTP;

void get(
    HTTP http,
    string url,
    void delegate(ubyte[]) success,
    void delegate(ubyte[]) failure)
{
    http.url = url;
    http.method = HTTP.Method.get;

    ubyte[] data;
    http.onReceive((ubyte[] tmp) {
        if (tmp.length > 0)
            data ~= tmp;
        return tmp.length;
    });

    if (http.perform() == 0 && success != null)
        success(data);
    else if (failure != null)
        failure(data);
}

void post(
    HTTP http,
    string url,
    void delegate(ubyte[]) success,
    void delegate(ubyte[]) failure,
    JSONValue json)
{
    http.url = url;
    http.method = HTTP.Method.post;
    http.setPostData(json.toString(JSONOptions.specialFloatLiterals), "application/json");

    ubyte[] data;
    http.onReceive((ubyte[] tmp) {
        if (tmp.length > 0)
            data ~= tmp;
        return tmp.length;
    });

    if (http.perform() == 0 && success != null)
        success(data);
    else if (failure != null)
        failure(data);
}

JSONValue toJSON(T)(T val)
{
    static if (__traits(compiles, { auto j = JSONValue(val); }))
        return JSONValue(val);

    JSONValue ret;
    static if (isAggregateType!T)
    {
        static foreach (F; FieldNameTuple!T)
            ret[F] = __traits(getMember, val, F).toJSON();
    }
    // TODO: Should handle associative arrays.
    else static if (isArray!T)
    {
        ret = JSONValue.emptyArray;
        foreach (u; val)
            ret.array ~= u.toJSON();
    }

    return ret;
}