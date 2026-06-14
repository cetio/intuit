/// Exception types for LLM endpoint and parsing exceptions.
module intuit.exception;

import std.format : format;
import std.exception : basicExceptionCtors;

/// Thrown when an endpoint returns a non-success status or invalid response.
class EndpointException : Exception
{
    /// HTTP method used for the request.
    string method;
    /// Target route of the request.
    string route;
    /// HTTP status code returned by the endpoint.
    ushort status;
    /// Reason phrase from the HTTP response.
    string reason;
    /// Raw response body content.
    string content;

    /**
     * Constructs an EndpointException.
     *
     * Params:
     *  method = The HTTP method.
     *  route = The request route.
     *  status = The HTTP status code.
     *  reason = The HTTP reason phrase.
     *  content = The raw response body.
     *  detail = Optional detail message to prepend over content.
     */
    this(
        string method,
        string route,
        ushort status,
        string reason,
        string content,
        string detail = null,
    )
    {
        super(buildMessage(method, route, status, reason, content, detail));
        this.method = method;
        this.route = route;
        this.status = status;
        this.reason = reason;
        this.content = content;
    }

private:
    static string buildMessage(
        string method,
        string route,
        ushort status,
        string reason,
        string content,
        string detail,
    )
    {
        string message = format("%s %s failed (%s %s)", method, route, status, reason);
        if (detail !is null && detail.length > 0)
            message ~= ": "~detail;
        else if (content !is null && content.length > 0)
            message ~= ": "~content;
        return message;
    }
}

/// Thrown when parsing a completion response fails.
class ResponseFormatException : Exception
{
    /// The raw text that could not be parsed.
    string rawText;
    /// The candidate text that triggered the parse failure.
    string candidateText;

    /**
     * Constructs a ResponseFormatException.
     *
     * Params:
     *  message = The exception message.
     *  rawText = The raw unparsable text.
     *  candidateText = The candidate text that caused the failure.
     */
    this(string message, string rawText, string candidateText)
    {
        super(message);
        this.rawText = rawText;
        this.candidateText = candidateText;
    }
}

/// Thrown when a stream encounters a (protocol) fault.
/// This does not apply to streaming exceptions related to content.
class StreamException : Exception
{
    mixin basicExceptionCtors;
}

/// Thrown when formatting (JSON schema) validation fails.
class FormatException : Exception
{
    mixin basicExceptionCtors;
}
