module tests.response.stream.sse;

import intuit.response.stream.sse;
import unit_threaded;

@Name("parses complete event in single chunk")
unittest
{
    SSEParser parser = new SSEParser();
    SSEEvent[] events = parser.feed(cast(const(ubyte)[])"data: hello\n\n");

    events.length.should == 1;
    events[0].data.should == "hello";
}

@Name("parses event split across chunks")
unittest
{
    SSEParser parser = new SSEParser();
    SSEEvent[] first = parser.feed(cast(const(ubyte)[])"data: hel");
    first.length.should == 0;

    SSEEvent[] second = parser.feed(cast(const(ubyte)[])"lo\n\n");
    second.length.should == 1;
    second[0].data.should == "hello";
}

@Name("captures event type and data lines")
unittest
{
    SSEParser parser = new SSEParser();
    SSEEvent[] events = parser.feed(cast(const(ubyte)[])"event: message_start\ndata: {\"type\":\"start\"}\n\n");

    events.length.should == 1;
    events[0].event.should == "message_start";
    events[0].data.should == "{\"type\":\"start\"}";
}

@Name("ignores id lines")
unittest
{
    SSEParser parser = new SSEParser();
    SSEEvent[] events = parser.feed(cast(const(ubyte)[])"id: 1\ndata: hello\n\n");

    events.length.should == 1;
    events[0].data.should == "hello";
    events[0].event.should == "";
}

@Name("concatenates multiple data lines")
unittest
{
    SSEParser parser = new SSEParser();
    SSEEvent[] events = parser.feed(cast(const(ubyte)[])"data: line1\ndata: line2\n\n");

    events.length.should == 1;
    events[0].data.should == "line1\nline2";
}

@Name("handles [DONE] event")
unittest
{
    SSEParser parser = new SSEParser();
    SSEEvent[] events = parser.feed(cast(const(ubyte)[])"data: [DONE]\n\n");

    events.length.should == 1;
    events[0].data.should == "[DONE]";
}

@Name("handles CRLF line endings")
unittest
{
    SSEParser parser = new SSEParser();
    SSEEvent[] events = parser.feed(cast(const(ubyte)[])"data: hello\r\n\r\n");

    events.length.should == 1;
    events[0].data.should == "hello";
}

@Name("flush returns trailing partial event")
unittest
{
    SSEParser parser = new SSEParser();
    SSEEvent[] events = parser.feed(cast(const(ubyte)[])"data: hello");
    events.length.should == 0;

    SSEEvent[] flushed = parser.flush();
    flushed.length.should == 1;
    flushed[0].data.should == "hello";
}

@Name("flush returns empty when nothing buffered")
unittest
{
    SSEParser parser = new SSEParser();
    SSEEvent[] flushed = parser.flush();
    flushed.length.should == 0;
}

@Name("handles multiple events in one chunk")
unittest
{
    SSEParser parser = new SSEParser();
    SSEEvent[] events = parser.feed(cast(const(ubyte)[])"data: a\n\ndata: b\n\n");

    events.length.should == 2;
    events[0].data.should == "a";
    events[1].data.should == "b";
}
