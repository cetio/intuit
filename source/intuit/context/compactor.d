/// Context compaction policies based on token and message-count limits.
module intuit.context.compactor;

import intuit.context.message;
import intuit.response;

/// Strategy used to compact a context once a limit is exceeded.
enum CompactorStrategy
{
    Trim,
    Callback
}

/// Compacts a message list when configured token or message limits are exceeded.
class Compactor
{
    /// Maximum number of messages before compaction; 0 disables this limit.
    size_t maxMessages;
    /// Maximum number of tokens before compaction; 0 disables this limit.
    size_t maxTokens = 250_000;
    /// The compaction strategy to apply.
    CompactorStrategy strategy = CompactorStrategy.Trim;
    /// User-supplied compaction routine invoked for CompactorStrategy.Callback.
    IMessage[] delegate(IMessage[]) callback;

    /**
     * Compacts the given messages if a configured limit is exceeded.
     *
     * Params:
     *  messages = The current message list.
     *
     * Returns:
     *  The compacted message list, or the input unchanged when under limits.
     */
    IMessage[] compact(IMessage[] messages)
    {
        if (!over(messages))
            return messages;

        final switch (strategy)
        {
        case CompactorStrategy.Trim:
            return trim(messages);
        case CompactorStrategy.Callback:
            return callback is null ? messages : callback(messages);
        }
    }

    /**
     * Computes the token total of a message list.
     *
     * The total is the sum of every assistant turn's completion tokens plus the
     * most recent assistant turn's prompt tokens.
     *
     * Params:
     *  messages = The messages to measure.
     *
     * Returns:
     *  The estimated token total.
     */
    static size_t tokens(IMessage[] messages)
    {
        size_t completion;
        size_t latestPrompt;
        foreach (message; messages)
        {
            AssistantMessage assistant = cast(AssistantMessage)message;
            if (assistant is null)
                continue;

            completion += assistant.usage.completionTokens;
            latestPrompt = assistant.usage.promptTokens;
        }
        return completion + latestPrompt;
    }

private:
    /// Returns true if any configured limit is exceeded.
    bool over(IMessage[] messages)
    {
        if (maxMessages > 0 && messages.length > maxMessages)
            return true;
        if (maxTokens > 0 && tokens(messages) > maxTokens)
            return true;
        return false;
    }

    /// Removes the oldest non-system messages until back under every limit.
    IMessage[] trim(IMessage[] messages)
    {
        IMessage[] ret = messages.dup;
        while (over(ret))
        {
            ptrdiff_t oldest = -1;
            foreach (i, message; ret)
            {
                if (message.role != Role.System)
                {
                    oldest = i;
                    break;
                }
            }

            if (oldest < 0)
                break;

            ret = ret[0..oldest]~ret[oldest + 1..$];
        }
        return ret;
    }
}
