module intuit.response.decision;

import intuit.response.completion : Usage;

import std.json : JSONValue;

public:

enum DecisionType : string
{
    Predicate = "predicate",
    Choice = "choice",
    Score = "score",
    Refusal = "refusal"
}

struct DecisionOption
{
    JSONValue value;
    string description;
}

struct DecisionLevel
{
    string label;
    string description;
}

struct DecisionQuestion
{
    DecisionType type;
    string name;
    string instructions;
    DecisionOption[] choices;
    DecisionLevel[] levels;
}

struct DecisionProbability
{
    JSONValue value;
    double probability = double.nan;
    string label;
}

struct DecisionAnswer
{
    JSONValue raw;
    DecisionType type;
    string name;
    double probability = double.nan;
    JSONValue choice;
    double score = double.nan;
    double confidence = double.nan;
    DecisionProbability[] probabilities;
}

struct Decision
{
    JSONValue raw;
    DecisionAnswer[] answers;
    Usage usage;

    ref inout(DecisionAnswer) answer(size_t index = 0) inout
    {
        if (index >= answers.length)
            throw new Exception("Decision answer index is out of range.");
        return answers[index];
    }

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
