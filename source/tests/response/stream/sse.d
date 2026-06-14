module tests.response.stream.sse;

import intuit.response.stream.sse;
import unit_threaded;

@Name("parses complete event in single chunk")
unittest
{
    SSEParser parser = new SSEParser();
    SSEEvent[] events = parser.feed(cast(const(ubyte)[])"data: hello\n\n");

    assert(events.length == 1);
    assert(events[0].data == "hello");
}

@Name("parses event split across chunks")
unittest
{
    SSEParser parser = new SSEParser();
    SSEEvent[] first = parser.feed(cast(const(ubyte)[])"data: hel");
    assert(first.length == 0);

    SSEEvent[] second = parser.feed(cast(const(ubyte)[])"lo\n\n");
    assert(second.length == 1);
    assert(second[0].data == "hello");
}

@Name("captures event type and data lines")
unittest
{
    SSEParser parser = new SSEParser();
    SSEEvent[] events = parser.feed(cast(const(ubyte)[])"event: message_start\ndata: {\"type\":\"start\"}\n\n");

    assert(events.length == 1);
    assert(events[0].event == "message_start");
    assert(events[0].data == "{\"type\":\"start\"}");
}

@Name("ignores id lines")
unittest
{
    SSEParser parser = new SSEParser();
    SSEEvent[] events = parser.feed(cast(const(ubyte)[])"id: 1\ndata: hello\n\n");

    assert(events.length == 1);
    assert(events[0].data == "hello");
    assert(events[0].event == "");
}

@Name("concatenates multiple data lines")
unittest
{
    SSEParser parser = new SSEParser();
    SSEEvent[] events = parser.feed(cast(const(ubyte)[])"data: line1\ndata: line2\n\n");

    assert(events.length == 1);
    assert(events[0].data == "line1\nline2");
}

@Name("handles [DONE] event")
unittest
{
    SSEParser parser = new SSEParser();
    SSEEvent[] events = parser.feed(cast(const(ubyte)[])"data: [DONE]\n\n");

    assert(events.length == 1);
    assert(events[0].data == "[DONE]");
}

@Name("handles CRLF line endings")
unittest
{
    SSEParser parser = new SSEParser();
    SSEEvent[] events = parser.feed(cast(const(ubyte)[])"data: hello\r\n\r\n");

    assert(events.length == 1);
    assert(events[0].data == "hello");
}

@Name("flush returns trailing partial event")
unittest
{
    SSEParser parser = new SSEParser();
    SSEEvent[] events = parser.feed(cast(const(ubyte)[])"data: hello");
    assert(events.length == 0);

    SSEEvent[] flushed = parser.flush();
    assert(flushed.length == 1);
    assert(flushed[0].data == "hello");
}

@Name("flush returns empty when nothing buffered")
unittest
{
    SSEParser parser = new SSEParser();
    SSEEvent[] flushed = parser.flush();
    assert(flushed.length == 0);
}

@Name("handles multiple events in one chunk")
unittest
{
    SSEParser parser = new SSEParser();
    SSEEvent[] events = parser.feed(cast(const(ubyte)[])"data: a\n\ndata: b\n\n");

    assert(events.length == 2);
    assert(events[0].data == "a");
    assert(events[1].data == "b");
}
