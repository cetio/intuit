module tests.stream.sse;

import intuit.stream.sse;
import unit_threaded;

@Name("parses complete event in single chunk")
unittest
{
    auto parser = new SseParser();
    auto events = parser.feed(cast(const(ubyte)[])"data: hello\n\n");

    assert(events.length == 1);
    assert(events[0].data == "hello");
}

@Name("parses event split across chunks")
unittest
{
    auto parser = new SseParser();
    auto first = parser.feed(cast(const(ubyte)[])"data: hel");
    assert(first.length == 0);

    auto second = parser.feed(cast(const(ubyte)[])"lo\n\n");
    assert(second.length == 1);
    assert(second[0].data == "hello");
}

@Name("ignores non-data lines")
unittest
{
    auto parser = new SseParser();
    auto events = parser.feed(cast(const(ubyte)[])"id: 1\ndata: hello\nevent: message\n\n");

    assert(events.length == 1);
    assert(events[0].data == "hello");
}

@Name("concatenates multiple data lines")
unittest
{
    auto parser = new SseParser();
    auto events = parser.feed(cast(const(ubyte)[])"data: line1\ndata: line2\n\n");

    assert(events.length == 1);
    assert(events[0].data == "line1\nline2");
}

@Name("handles [DONE] event")
unittest
{
    auto parser = new SseParser();
    auto events = parser.feed(cast(const(ubyte)[])"data: [DONE]\n\n");

    assert(events.length == 1);
    assert(events[0].data == "[DONE]");
}

@Name("handles CRLF line endings")
unittest
{
    auto parser = new SseParser();
    auto events = parser.feed(cast(const(ubyte)[])"data: hello\r\n\r\n");

    assert(events.length == 1);
    assert(events[0].data == "hello");
}

@Name("flush returns trailing partial event")
unittest
{
    auto parser = new SseParser();
    auto events = parser.feed(cast(const(ubyte)[])"data: hello");
    assert(events.length == 0);

    auto flushed = parser.flush();
    assert(flushed.length == 1);
    assert(flushed[0].data == "hello");
}

@Name("flush returns empty when nothing buffered")
unittest
{
    auto parser = new SseParser();
    auto flushed = parser.flush();
    assert(flushed.length == 0);
}

@Name("handles multiple events in one chunk")
unittest
{
    auto parser = new SseParser();
    auto events = parser.feed(cast(const(ubyte)[])"data: a\n\ndata: b\n\n");

    assert(events.length == 2);
    assert(events[0].data == "a");
    assert(events[1].data == "b");
}
