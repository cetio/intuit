/// Decision request and response types.
module intuit.response.decision;

import intuit.exception : EndpointException, FormatException, MalformedResponseException;
import intuit.json : fromJSON;
import intuit.response.completion : Usage;

import std.json : JSONType, JSONValue;
import std.math : isFinite;

/// The kind of question a decision asks or answer provides.
enum DecisionType : string
{
    /// Probability estimate for a yes/no question.
    Predicate = "predicate",
    /// Selection among named options.
    Choice = "choice",
    /// Numeric score on a level scale.
    Score = "score",
    /// The model refused to answer.
    Refusal = "refusal"
}

/// A selectable option for a choice question.
struct DecisionOption
{
    /// The option value.
    JSONValue value;
    /// Optional human-readable description.
    string description;
}

/// A labeled level for a score question.
struct DecisionLevel
{
    /// The level label.
    string label;
    /// Optional human-readable description.
    string description;
}

/// A single question within a decision request.
struct DecisionQuestion
{
    /// The question type.
    DecisionType type;
    /// Optional unique question name.
    string name;
    /// Instructions given to the model.
    string instructions;
    /// Options for Choice questions.
    DecisionOption[] choices;
    /// Levels for Score questions.
    DecisionLevel[] levels;
}

/// A probability assigned to a single option or level.
struct DecisionProbability
{
    /// The option value or level index.
    JSONValue value;
    /// The assigned probability between zero and one.
    double probability = double.nan;
    /// The level label for score questions.
    string label;
}

/// The model's answer to a decision question.
struct DecisionAnswer
{
    /// The raw JSON value for this answer.
    JSONValue raw;
    /// The answer type.
    DecisionType type;
    /// The name of the question this answers.
    string name;
    /// Probability estimate for Predicate answers.
    double probability = double.nan;
    /// Selected option for Choice answers.
    JSONValue choice;
    /// Numeric score for Score answers.
    double score = double.nan;
    /// Optional model-reported confidence between zero and one.
    double confidence = double.nan;
    /// Per-option or per-level probability distribution.
    DecisionProbability[] probabilities;
}

/// Parsed result from a decisions request.
struct Decision
{
    /// The raw JSON response.
    JSONValue raw;
    /// All returned answers.
    DecisionAnswer[] answers;
    /// Token accounting for the request, when reported by the endpoint.
    Usage usage;

    /// Gets the answer at `index`.
    ref inout(DecisionAnswer) answer(size_t index = 0) inout
    {
        if (index >= answers.length)
            throw new Exception("Decision answer index is out of range.");
        return answers[index];
    }

    /// Gets the answer for the question named `name`.
    ref inout(DecisionAnswer) answer(string name) inout
    {
        foreach (ref entry; answers)
        {
            if (entry.name == name)
                return entry;
        }

        throw new Exception("Decision answer not found: "~name);
    }
}

package(intuit):
JSONValue decisionQuestions(DecisionQuestion[] questions)
{
    if (questions.length == 0)
        throw new FormatException("Decisions require at least one question.");

    JSONValue ret = JSONValue.emptyArray;
    bool[string] names;
    foreach (ref question; questions)
    {
        if (question.instructions.length == 0 || question.type == DecisionType.Refusal)
            throw new FormatException("Decision questions require instructions and a non-refusal type.");

        JSONValue entry = JSONValue.emptyObject;
        entry["type"] = JSONValue(cast(string)question.type);
        entry["instructions"] = JSONValue(question.instructions);
        if (question.name.length > 0)
        {
            if (question.name in names)
                throw new FormatException("Decision question names must be unique.");

            names[question.name] = true;
            entry["name"] = JSONValue(question.name);
        }

        if (question.type == DecisionType.Choice)
        {
            if (question.choices.length < 2)
                throw new FormatException("Choice questions require at least two choices.");

            entry["choices"] = JSONValue.emptyArray;
            bool[string] values;
            foreach (ref option; question.choices)
            {
                if (option.value.type != JSONType.string && option.value.type != JSONType.true_
                    && option.value.type != JSONType.false_)
                    throw new FormatException("Decision choice values must be strings or booleans.");
                if (option.value.toString() in values)
                    throw new FormatException("Decision choice values must be unique.");

                values[option.value.toString()] = true;
                JSONValue choice = JSONValue(["value": option.value]);
                if (option.description.length > 0)
                    choice["description"] = JSONValue(option.description);

                entry["choices"].array ~= choice;
            }
        }
        else if (question.type == DecisionType.Score)
        {
            if (question.levels.length < 2)
                throw new FormatException("Score questions require at least two levels.");

            entry["levels"] = JSONValue.emptyArray;
            foreach (ref level; question.levels)
            {
                if (level.label.length == 0)
                    throw new FormatException("Decision score levels require labels.");

                JSONValue score = JSONValue(["label": JSONValue(level.label)]);
                if (level.description.length > 0)
                    score["description"] = JSONValue(level.description);

                entry["levels"].array ~= score;
            }
        }

        ret.array ~= entry;
    }
    return ret;
}

Decision parseDecisionResponse(
    JSONValue json,
    DecisionQuestion[] questions = null,
    string modelName = null,
)
{
    if (json.type != JSONType.object)
        throw new MalformedResponseException("Expected a decision response object.", json.toString());
    if ("error" in json)
        throw new EndpointException("POST", "decisions", 0, "error", json.toString());
    if ("answers" !in json || json["answers"].type != JSONType.array)
        throw new MalformedResponseException("Expected a decision answers array.", json.toString());
    if (json["answers"].array.length == 0)
        throw new MalformedResponseException("Decision response contains no answers.", json.toString());
    if (questions.length > 0 && questions.length != json["answers"].array.length)
        throw new MalformedResponseException(
            "Decision answer count does not match the questions.",
            json.toString(),
        );

    Decision ret;
    ret.raw = json;
    foreach (i, entry; json["answers"].array)
    {
        if (entry.type != JSONType.object || "type" !in entry)
            throw new MalformedResponseException("Decision answers require a type.", json.toString());

        DecisionAnswer answer;
        try
            answer = fromJSON!DecisionAnswer(entry);
        catch (Exception error)
        {
            throw new MalformedResponseException(
                "Invalid decision answer: "~error.msg,
                json.toString(),
                entry.toString(),
            );
        }
        answer.raw = entry;
        final switch (answer.type)
        {
        case DecisionType.Predicate:
            if (!answer.probability.isFinite || answer.probability < 0 || answer.probability > 1)
                throw new MalformedResponseException(
                    "Decision probabilities must be between zero and one.",
                    json.toString(),
                );
            break;

        case DecisionType.Choice:
            if (answer.choice.type != JSONType.string && answer.choice.type != JSONType.true_
                && answer.choice.type != JSONType.false_)
                throw new MalformedResponseException(
                    "Decision choices must be strings or booleans.",
                    json.toString(),
                );
            break;

        case DecisionType.Score:
            if (!answer.score.isFinite || answer.score < 0)
                throw new MalformedResponseException("Invalid decision score.", json.toString());
            break;

        case DecisionType.Refusal:
            break;
        }

        if ("confidence" in entry
            && (!answer.confidence.isFinite || answer.confidence < 0 || answer.confidence > 1))
            throw new MalformedResponseException(
                "Decision probabilities must be between zero and one.",
                json.toString(),
            );

        foreach (ref probability; answer.probabilities)
        {
            if (!probability.probability.isFinite
                || probability.probability < 0
                || probability.probability > 1)
                throw new MalformedResponseException(
                    "Decision probabilities must be between zero and one.",
                    json.toString(),
                );
        }

        if (questions.length > 0)
        {
            if (answer.name != questions[i].name)
                throw new MalformedResponseException(
                    "Decision answer name does not match its question.",
                    json.toString(),
                );
            if (answer.type != DecisionType.Refusal && answer.type != questions[i].type)
                throw new MalformedResponseException(
                    "Decision answer type does not match its question.",
                    json.toString(),
                );

            if (answer.type == DecisionType.Choice)
            {
                bool found;
                foreach (ref option; questions[i].choices)
                    found = found || option.value == answer.choice;

                if (!found)
                    throw new MalformedResponseException(
                        "Decision selected an unknown choice.",
                        json.toString(),
                    );
            }
            else if (answer.type == DecisionType.Score
                && (answer.score < 0 || answer.score > questions[i].levels.length - 1))
                throw new MalformedResponseException(
                    "Decision score is outside its levels.",
                    json.toString(),
                );
        }

        ret.answers ~= answer;
    }

    ret.usage.modelName = "model" in json && json["model"].type == JSONType.string
        ? json["model"].str : modelName;
    if ("latency" in json)
    {
        try
            ret.usage.latency = fromJSON!float(json["latency"]);
        catch (Exception)
            throw new MalformedResponseException("Invalid decision latency.", json.toString());
    }

    // Reads an integral value from the JSON usage object.
    if ("usage" in json && json["usage"].type == JSONType.object)
    {
        JSONValue usage = json["usage"];
        ret.usage.promptTokens = "input_tokens" in usage
            ? fromJSON!size_t(usage["input_tokens"]) : 0;
        ret.usage.completionTokens = "output_tokens" in usage
            ? fromJSON!size_t(usage["output_tokens"]) : 0;
        ret.usage.totalTokens = "total_tokens" in usage
            ? fromJSON!size_t(usage["total_tokens"]) : 0;
        if ("total_tokens" !in usage)
            ret.usage.totalTokens = ret.usage.promptTokens + ret.usage.completionTokens;

        if ("input_tokens_details" in usage
            && usage["input_tokens_details"].type == JSONType.object
            && "cached_tokens" in usage["input_tokens_details"])
            ret.usage.cacheHits = fromJSON!size_t(usage["input_tokens_details"]["cached_tokens"]);

        if (ret.usage.cacheHits > ret.usage.promptTokens)
            throw new MalformedResponseException(
                "Cached decision tokens exceed input tokens.",
                json.toString(),
            );

        ret.usage.cacheMisses = ret.usage.promptTokens - ret.usage.cacheHits;
    }
    return ret;
}
