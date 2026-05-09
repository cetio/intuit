module intuit.error;

import std.format : format;

public:

class EndpointError : Exception
{
    string method;
    string route;
    ushort status;
    string reason;
    string content;

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

class CompletionParseError : Exception
{
    string rawText;
    string candidateText;

    this(string message, string rawText, string candidateText)
    {
        super(message);
        this.rawText = rawText;
        this.candidateText = candidateText;
    }
}
