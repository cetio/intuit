module intuit.provider.systemone;

import intuit.exception : FormatException;
import intuit.model : ModelConfig;
import intuit.response.decision;

import std.conv : to;
import std.json : JSONType, JSONValue;

class SystemOneModelConfig : ModelConfig
{
    this(string name)
    {
        super(name);
    }

    override JSONValue buildDecisionsPayload(JSONValue input, DecisionQuestion[] questions)
    {
        if (input.type != JSONType.string && input.type != JSONType.object && input.type != JSONType.array)
            throw new FormatException("System One state must be text, an object, or an array.");

        if (questions.length > 255)
            throw new FormatException("System One supports at most 255 questions per request.");

        foreach (ref question; questions)
        {
            if (question.choices.length > 255 || question.levels.length > 10)
                throw new FormatException("System One supports at most 255 choices or 10 score levels.");
        }

        JSONValue serialized = decisionQuestions(questions);
        JSONValue ret = JSONValue.emptyObject;
        if (params.type == JSONType.object)
        {
            foreach (key, value; params.object)
                ret[key] = value;
        }

        ret["model"] = JSONValue(name);
        ret["state"] = input;
        ret["questions"] = JSONValue.emptyObject;
        foreach (i, ref question; questions)
        {
            string key = questionKey(question, i);
            if (key in ret["questions"])
                throw new FormatException("Decision question name conflicts with an unnamed question.");

            JSONValue entry = serialized[i];
            entry.object.remove("name");
            final switch (question.type)
            {
                case DecisionType.Predicate:
                    entry["type"] = JSONValue("noul");
                    break;

                case DecisionType.Choice:
                    entry.object.remove("choices");
                    entry["criteria"] = JSONValue.emptyObject;
                    foreach (ref option; question.choices)
                    {
                        entry["criteria"][optionKey(question, option)] = option.description.length > 0
                            ? JSONValue(option.description) : JSONValue(null);
                    }
                    break;

                case DecisionType.Score:
                    entry.object.remove("levels");
                    entry["criteria"] = JSONValue.emptyArray;
                    foreach (ref level; question.levels)
                    {
                        entry["criteria"].array ~= JSONValue(level.description.length > 0
                            ? level.label~": "~level.description : level.label);
                    }
                    break;

                case DecisionType.Refusal:
                    throw new FormatException("Refusal is not a question type.");
            }

            ret["questions"][key] = entry;
        }
        return ret;
    }

    override Decision parseDecisionsResponse(JSONValue json, DecisionQuestion[] questions = null)
    {
        if (json.type != JSONType.object || "error" in json)
            return super.parseDecisionsResponse(json, questions);
        if ("answers" !in json || json["answers"].type != JSONType.object)
            throw malformedResponse(json, "Expected a System One answers object.");
        if (questions.length == 0 || questions.length != json["answers"].object.length)
            throw malformedResponse(json, "System One answers must match the requested questions.");

        JSONValue normalized = JSONValue(json.object.dup);
        normalized["answers"] = JSONValue.emptyArray;
        foreach (i, ref question; questions)
        {
            string key = questionKey(question, i);
            if (key !in json["answers"] || json["answers"][key].type != JSONType.object)
                throw malformedResponse(json, "Missing System One answer: "~key);

            JSONValue entry = JSONValue(json["answers"][key].object.dup);
            if ("type" !in entry || entry["type"].type != JSONType.string)
                throw malformedResponse(json, "System One answers require a type.");

            entry["name"] = JSONValue(question.name);
            if (entry["type"].str == "noul")
            {
                if ("noul" !in entry)
                    throw malformedResponse(json, "Missing System One noul probability.");

                entry["type"] = JSONValue("predicate");
                entry["probability"] = entry["noul"];
            }
            else if (entry["type"].str == "choice")
            {
                if ("choice" !in entry || entry["choice"].type != JSONType.string)
                    throw malformedResponse(json, "System One choice answers require a string choice.");

                foreach (ref option; question.choices)
                {
                    if (optionKey(question, option) == entry["choice"].str)
                    {
                        entry["choice"] = option.value;
                        break;
                    }
                }
            }

            if ("probabilities" in entry)
            {
                JSONValue probabilities = entry["probabilities"];
                size_t count = question.type == DecisionType.Choice ? question.choices.length : question.levels.length;
                if (probabilities.type != JSONType.object || probabilities.object.length != count)
                    throw malformedResponse(json, "System One probabilities must match the requested options.");

                entry["probabilities"] = JSONValue.emptyArray;
                foreach (j; 0..count)
                {
                    string option = question.type == DecisionType.Choice
                        ? optionKey(question, question.choices[j]) : j.to!string;
                    if (option !in probabilities)
                        throw malformedResponse(json, "Missing System One option probability: "~option);

                    JSONValue probability = JSONValue.emptyObject;
                    probability["value"] = question.type == DecisionType.Choice
                        ? question.choices[j].value : JSONValue(cast(long)j);
                    probability["probability"] = probabilities[option];
                    if (question.type == DecisionType.Score)
                        probability["label"] = JSONValue(question.levels[j].label);

                    entry["probabilities"].array ~= probability;
                }
            }

            normalized["answers"].array ~= entry;
        }

        Decision ret = super.parseDecisionsResponse(normalized, questions);
        ret.raw = json;
        foreach (i, ref answer; ret.answers)
            answer.raw = json["answers"][questionKey(questions[i], i)];
        return ret;
    }

private:
    static string questionKey(DecisionQuestion question, size_t index)
        => question.name.length > 0 ? question.name : "__intuit_question_"~index.to!string;

    static string optionKey(DecisionQuestion question, DecisionOption option)
    {
        foreach (ref candidate; question.choices)
        {
            if (candidate.value.type != JSONType.string)
                return option.value.toString();
        }
        return option.value.str;
    }
}
