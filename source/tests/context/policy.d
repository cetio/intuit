module tests.context.policy;

import intuit.context.policy.tool : ToolPolicy, ToolPolicyStatus;
import intuit.tool : Tool;
import unit_threaded;

import std.json : JSONValue;

private Tool makeTool(string name)
{
    Tool ret = new Tool(
        name,
        null,
        JSONValue.emptyObject,
        (JSONValue arguments) {
            return arguments;
        },
    );
    return ret;
}

@Name("ToolPolicy returns None when no matching rule or delegate exists")
unittest
{
    ToolPolicy policy;
    policy.allow("read");
    policy.deny("delete");

    policy.eval(makeTool("write")).should == ToolPolicyStatus.None;
}

@Name("ToolPolicy accepts a tool in its allow list")
unittest
{
    ToolPolicy policy;
    policy.allow("read");

    policy.eval(makeTool("read")).should == ToolPolicyStatus.Allowed;
}

@Name("ToolPolicy denies a tool in its deny list")
unittest
{
    ToolPolicy policy;
    policy.deny("delete");

    policy.eval(makeTool("delete")).should == ToolPolicyStatus.Denied;
}

@Name("ToolPolicy deny list takes precedence over its allow list")
unittest
{
    ToolPolicy policy;
    policy.allow("delete");
    policy.deny("delete");

    policy.eval(makeTool("delete")).should == ToolPolicyStatus.Denied;
}

@Name("ToolPolicy delegates evaluation when no list matches")
unittest
{
    ToolPolicy policy;
    policy.evaluator = (Tool tool) {
        return tool.name == "read" ? ToolPolicyStatus.Allowed : ToolPolicyStatus.Denied;
    };

    policy.eval(makeTool("read")).should == ToolPolicyStatus.Allowed;
    policy.eval(makeTool("write")).should == ToolPolicyStatus.Denied;
}

@Name("ToolPolicy list additions are idempotent")
unittest
{
    ToolPolicy policy;
    policy.allow("read");
    policy.allow("read");
    policy.deny("delete");
    policy.deny("delete");

    policy.allowList.length.should == 1;
    policy.denyList.length.should == 1;
}
