/// Tool execution policy and outcome types.
module intuit.context.policy.tool;

import intuit.tool : Tool;

import std.algorithm.searching : canFind;

/// Outcome of tool policy evaluation for a tool call.
enum ToolPolicyStatus
{
    /// No policy rule matched.
    None,
    /// Tool execution is allowed.
    Allowed,
    /// Tool execution is denied.
    Denied,
    /// Tool execution awaits manual handling.
    Pending,
    /// Policy evaluation or tool execution failed.
    Failed
}

/// Policy evaluation outcome attached to a tool call.
struct ToolPolicyResult
{
    /// The policy decision.
    ToolPolicyStatus status;
    /// Human-readable explanation of the outcome.
    string message;
}

/// Allow and deny lists with an optional evaluator governing tool execution.
struct ToolPolicy
{
    /// Tool names always allowed.
    string[] allowList;
    /// Tool names always denied.
    string[] denyList;
    /// Custom policy delegate invoked when no list matches.
    ToolPolicyStatus delegate(Tool) evaluator;

    /**
     * Adds a tool name to the allow list.
     * 
     * Params:
     *   toolName = The name of the tool to allow.
     *
     * Returns: 
     *   A reference to this policy for chaining.
     */
    ref ToolPolicy allow(string toolName)
    {
        if (!allowList.canFind(toolName))
            allowList ~= toolName;
        return this;
    }

    /**
     * Adds a tool name to the deny list.
     * 
     * Params:
     *   toolName = The name of the tool to deny.
     *
     * Returns: 
     *   A reference to this policy for chaining.
     */
    ref ToolPolicy deny(string toolName)
    {
        if (!denyList.canFind(toolName))
            denyList ~= toolName;
        return this;
    }

    /**
     * Evaluates the policy for a tool. Deny takes precedence over allow, 
     * and the evaluator is invoked only when neither list matches.
     * 
     * Params:
     *   tool = The tool to evaluate.
     *
     * Returns:
     *   The policy decision.
     */
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
