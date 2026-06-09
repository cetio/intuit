/// Server-Sent Events (SSE) parser for streaming HTTP responses.
module intuit.stream.sse;

import std.string;
import std.array;

/// A single parsed SSE event.
struct SseEvent
{
    /// Concatenated data lines. Empty for heartbeats or no data.
    string data;
}

/// Parses raw HTTP chunks into discrete SSE events.
class SseParser
{
    private string _buffer;

    /**
     * Feed new bytes and return all complete events found.
     *
     * Params:
     *  chunk = Raw bytes from the HTTP response.
     *
     * Returns:
     *  An array of complete events parsed since the last feed.
     */
    SseEvent[] feed(const(ubyte)[] chunk)
    {
        _buffer ~= cast(string)chunk;
        return extractEvents();
    }

    /**
     * Flush any trailing buffered data as a final event.
     *
     * Returns:
     *  Zero or one final SseEvent.
     */
    SseEvent[] flush()
    {
        SseEvent[] ret;
        if (_buffer.length > 0)
        {
            string eventData = parseEvent(_buffer);
            if (eventData.length > 0)
                ret ~= SseEvent(eventData);
            _buffer = null;
        }
        return ret;
    }

private:
    SseEvent[] extractEvents()
    {
        SseEvent[] ret;

        while (true)
        {
            // Events are separated by two consecutive line breaks.
            // Accept both \n\n and \r\n\r\n by normalizing to \n\n first.
            string normalized = _buffer.replace("\r\n", "\n");
            ptrdiff_t split = normalized.indexOf("\n\n");
            if (split < 0)
                break;

            string block = _buffer[0..split];
            _buffer = _buffer[split + 2..$];

            string eventData = parseEvent(block);
            if (eventData.length > 0)
                ret ~= SseEvent(eventData);
        }

        return ret;
    }

    static string parseEvent(string block)
    {
        string[] dataLines;
        foreach (line; block.splitLines)
        {
            line = line.strip;
            if (line.startsWith("data:"))
                dataLines ~= line[5..$].strip;
        }
        return dataLines.join("\n");
    }
}
