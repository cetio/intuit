/// Accumulates streaming Completion chunks into a single running Completion.
module intuit.stream.delta;

import intuit.response;
import std.json : JSONValue;

/// Merges delta fragments into a single accumulated Completion.
class DeltaAccumulator
{
    private Completion _acc;

    this()
    {
        _acc.raw = JSONValue.emptyObject;
        _acc.choices = [Choice.init];
    }

    /**
     * Merge a new chunk into the running completion.
     *
     * Text and reasoning are concatenated. Tool calls are appended.
     * Finish reason is captured from the first non-Unknown value.
     *
     * Params:
     *  chunk = A Completion parsed from a single SSE event.
     */
    void accumulate(Completion chunk)
    {
        if (chunk.choices.length == 0)
            return;

        // Ensure accumulated choices array can hold all indices.
        if (_acc.choices.length < chunk.choices.length)
        {
            size_t oldLen = _acc.choices.length;
            _acc.choices.length = chunk.choices.length;
            foreach (i; oldLen.._acc.choices.length)
                _acc.choices[i] = Choice.init;
        }

        foreach (i, choice; chunk.choices)
        {
            Choice target = _acc.choices[i];

            if (choice.text.length > 0)
                target.text ~= choice.text;

            if (choice.reasoning.length > 0)
                target.reasoning ~= choice.reasoning;

            if (choice.toolCalls.length > 0)
                target.toolCalls ~= choice.toolCalls;

            if (choice.finishReason != FinishReason.Unknown)
                target.finishReason = choice.finishReason;

            if (!choice.logProbs.isNull)
                target.logProbs = choice.logProbs;

            if (!choice.content.isNull)
                target.content = choice.content;

            if (!choice.raw.isNull)
                target.raw = choice.raw;

            _acc.choices[i] = target;
        }

        // Merge raw JSON if the chunk carries metadata (e.g., final usage block).
        if (!chunk.raw.isNull)
            _acc.raw = chunk.raw;
    }

    /// Gets the current accumulated state.
    ref Completion current()
        => _acc;
}
