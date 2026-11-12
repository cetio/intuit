/// Context compaction policies based on token and message-count limits.
module intuit.context.compactor;

import intuit.context.message;
import intuit.response;

/// Strategy used to compact a context once a limit is exceeded.
enum CompactorStrategy
{
    /// Removes the oldest non-system messages.
    Trim,
    /// Invokes the user-supplied callback.
    Callback
}

/// Compacts a message list when configured token or message limits are exceeded.
class Compactor
{
    /// Maximum number of messages before compaction. Disabled if set to 0.
    size_t maxMessages;
    /// Maximum number of tokens before compaction. Disabled if set to 0.
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
        size_t completion;
        size_t latestPrompt;
        size_t latestAssistant = messages.length;
        if (maxTokens > 0)
        {
            foreach (i, message; messages)
            {
                AssistantMessage assistant = cast(AssistantMessage)message;
                if (assistant is null)
                    continue;
                
                Usage usage = assistant.usage;
                completion += usage.completionTokens;
                latestPrompt = usage.promptTokens;
                latestAssistant = i;
            }
        }

        size_t retained = messages.length;
        size_t firstRetained;
        size_t preservedPrompt;
        while (firstRetained < messages.length
            && ((maxMessages > 0 && retained > maxMessages)
                || (maxTokens > 0 && completion + latestPrompt > maxTokens)))
        {
            IMessage message = messages[firstRetained];
            AssistantMessage assistant;
            if (maxTokens > 0)
                assistant = cast(AssistantMessage)message;

            if (message.role == Role.System && assistant !is null)
                preservedPrompt = assistant.usage.promptTokens;
            else
            {
                --retained;
                if (assistant !is null)
                {
                    completion -= assistant.usage.completionTokens;
                    if (firstRetained == latestAssistant)
                        latestPrompt = preservedPrompt;
                }
            }
            ++firstRetained;
        }

        IMessage[] ret = new IMessage[retained];
        size_t index;
        foreach (message; messages[0..firstRetained])
        {
            if (message.role == Role.System)
                ret[index++] = message;
        }
        
        ret[index..$] = messages[firstRetained..$];
        return ret;
    }
}
