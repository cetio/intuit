/// Server-Sent Events (SSE) parser for streaming HTTP responses.
module intuit.stream.sse;

import std.string;
import std.array;

/// A single parsed SSE event.
struct SSEEvent
{
    /// Event type/name (e.g., "message_start", "content_block_delta").
    string event;
    /// Concatenated data lines. Empty for heartbeats or no data.
    string data;
}

/// Parses raw HTTP chunks into discrete SSE events.
class SSEParser
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
    SSEEvent[] feed(const(ubyte)[] chunk)
    {
        _buffer ~= cast(string)chunk;
        return extractEvents();
    }

    /**
     * Flush any trailing buffered data as a final event.
     *
     * Returns:
     *  Zero or one final SSEEvent.
     */
    SSEEvent[] flush()
    {
        SSEEvent[] ret;
        if (_buffer.length > 0)
        {
            SSEEvent event = parseEvent(_buffer);
            if (event.data.length > 0 || event.event.length > 0)
                ret ~= event;
            _buffer = null;
        }
        return ret;
    }

private:
    SSEEvent[] extractEvents()
    {
        SSEEvent[] ret;

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

            SSEEvent event = parseEvent(block);
            if (event.data.length > 0 || event.event.length > 0)
                ret ~= event;
        }

        return ret;
    }

    static SSEEvent parseEvent(string block)
    {
        SSEEvent ret;
        string[] dataLines;
        foreach (line; block.splitLines)
        {
            line = line.strip;
            if (line.startsWith("event:"))
                ret.event = line[6..$].strip;
            else if (line.startsWith("data:"))
                dataLines ~= line[5..$].strip;
        }
        ret.data = dataLines.join("\n");
        return ret;
    }
}
