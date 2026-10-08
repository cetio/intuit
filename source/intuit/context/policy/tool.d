module intuit.context.policy.tool;

import intuit.tool : Tool;

import std.algorithm.searching : canFind;

enum ToolPolicyStatus
{
    None,
    Allowed,
    Denied,
    Pending,
    Failed
}

struct ToolPolicyResult
{
    ToolPolicyStatus status;
    string message;
}

struct ToolPolicy
{
public:
    string[] allowList;
    string[] denyList;
    ToolPolicyStatus delegate(Tool) evaluator;

    ref ToolPolicy allow(string toolName)
    {
        if (!allowList.canFind(toolName))
            allowList ~= toolName;
        return this;
    }

    ref ToolPolicy deny(string toolName)
    {
        if (!denyList.canFind(toolName))
            denyList ~= toolName;
        return this;
    }

    ToolPolicyStatus eval(Tool tool)
    {
        if (denyList.canFind(tool.name))
            return ToolPolicyStatus.Denied;
            
        if (allowList.canFind(tool.name))
            return ToolPolicyStatus.Allowed;

        if (evaluator !is null)
            return evaluator(tool);

        return ToolPolicyStatus.None;
    }
}
