/// Thread-safe completion stream consumer.
module intuit.response.stream;

import intuit.response.completion;

import core.atomic;
import core.thread;
import std.json : JSONValue, JSONValue, parseJSON;

/**
 * Thread-safe completion stream consumer.
 *
 * This is the frontend for all Server-Sent Events (SSE) streaming
 * completions. Endpoint implementations feed parsed SSE chunks into
 * a CompletionStream, and callers consume them via `next()`, `collect()`,
 * or the per-chunk `callback` delegate.
 */
class CompletionStream
{
private:
    Completion[] _completions;
    shared size_t _length;
    shared size_t _index;
    shared bool _writer;

public:
    /// Internal delegate to start the stream.
    void delegate(CompletionStream) _commence;

    /// Accumulated metadata JSON.
    JSONValue json;
    /// Model name for the stream.
    string model;
    /// True when the stream has finished.
    bool complete;
    /// If set, an exception thrown by the background worker.
    Exception error;
    /// Callback invoked for each completion chunk.
    void delegate(Completion) callback;

    /**
     * Constructs a CompletionStream.
     *
     * Params:
     *  model = The model name.
     *  callback = Delegate called for each chunk.
     */
    this(string model, void delegate(Completion) callback)
    {
        this.model = model;
        this.callback = callback;
        this._completions = null;
        this._length = 0;
        this._index = 0;
        this._writer = false;
        this.complete = false;
        this.json = JSONValue.emptyObject;
    }

    /**
     * Gets the next available completion chunk, blocking until available.
     *
     * Returns:
     *  The next Completion chunk.
     *
     * Throws:
     *  Exception if the stream has not been initialized.
     */
    Completion next()
    {
        if (_commence is null)
            throw new Exception("Stream not initialized");

        if (error !is null)
            throw error;

        while (atomicLoad!(MemoryOrder.acq)(_writer))
            Thread.yield();

        size_t cur = atomicFetchAdd!(MemoryOrder.seq)(_index, 1);

        while (cur >= atomicLoad!(MemoryOrder.acq)(_length))
        {
            if (complete)
                throw new Exception("Stream exhausted");
            Thread.yield();
        }

        atomicFence!(MemoryOrder.acq);
        return _completions[cur];
    }

    /**
     * Collects `count` completion chunks.
     *
     * Params:
     *  count = Number of chunks to collect.
     *
     * Returns:
     *  An array of collected Completion chunks.
     */
    Completion[] collect(size_t count)
    {
        Completion[] ret;
        foreach (i; 0..count)
            ret ~= next();
        return ret;
    }

    /// Begins streaming using the internal commence delegate.
    void begin()
    {
        _commence(this);
    }

    /**
     * Sets the commence delegate and starts streaming.
     *
     * Params:
     *  cb = The delegate that drives the stream.
     */
    void commence(void delegate(CompletionStream) cb)
    {
        _commence = cb;
        begin();
    }

    /**
     * Pushes a new completion chunk into the stream.
     *
     * Params:
     *  val = The completion chunk to append.
     */
    void update(Completion val)
    {
        atomicStore!(MemoryOrder.rel)(_writer, true);
        atomicFence!(MemoryOrder.rel);

        _completions ~= val;
        atomicFetchAdd!(MemoryOrder.rel)(_length, 1);

        atomicFence!(MemoryOrder.rel);
        atomicStore!(MemoryOrder.rel)(_writer, false);
    }
}
