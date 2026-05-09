module checks.examples;

import intuit;
import std.json : JSONValue, parseJSON;

unittest
{
    static assert(__traits(compiles, {
        auto endpoint = new OpenAI("https://api.openai.com", "key");
        auto model = cast(OpenAIModel)endpoint.model("gpt-4o-mini");
        Context ctx;
        ctx.system("You are helpful.").user("Say hello.");
        auto payload = model.jsonMode().completionsJSON(ctx.messages);
        auto schema = parseJSON(`{"type":"object"}`);
        auto schemaModel = new OpenAIModel("gpt-4o").jsonSchema("reply", schema);
        auto schemaPayload = schemaModel.completionsJSON(JSONValue("hello"));
    }));
}

unittest
{
    auto endpoint = new OpenAI("http://127.0.0.1:1234");
    assert(endpoint.available.length > 0);
}