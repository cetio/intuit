module tests.auc.predict;

import intuit;
import intuit.openai : OpenAI;
import tests.auc.tools;
import std.json : JSONValue, parseJSON, JSONType;
import std.conv : to;

/**
 * Unit test for AI model to predict tramadol concentrations using tools.
 * 
 * This test provides the AI model with tools containing tramadol pharmacokinetic
 * data and calculation functions. The model is expected to:
 * 1. Call get_tramadol_pharmacokinetics to get PK parameters
 * 2. Call predict_tramadol_concentration to calculate concentrations at various time points
 * 3. Return a summary of the predictions
 * 
 * The test uses a loop to handle multi-turn tool calling, as the model may call tools
 * in multiple responses rather than all at once.
 * 
 * BASELINE PARAMETERS:
 * - Drug: Tramadol 50mg immediate-release oral tablet
 * - Patient: 70kg male
 * - Data sources: DailyMed (NIH) and PubMed peer-reviewed studies
 */

unittest
{
    // Set up OpenAI endpoint (using local model for testing)
    auto endpoint = new OpenAI("http://127.0.0.1:1234");
    auto model = endpoint.model("qwen/qwen3.5-9b");
    
    // Register tramadol tools
    endpoint.tools().add!predict_tramadol_concentration();
    endpoint.tools().add!get_tramadol_pharmacokinetics();
    
    // Create context with prompt for the model
    Context ctx;
    ctx.system("You are a pharmacokinetic expert. Use the available tools to predict tramadol plasma concentrations.");
    ctx.user("Predict the plasma concentration of tramadol at the following time points after " ~
        "oral administration: 0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 6, 8, 12, and 24 hours. " ~
        "Use a 50mg dose for a 70kg male patient. First get the pharmacokinetic parameters, " ~
        "then calculate concentrations at each time point.");
    
    // Loop to handle multi-turn tool calling
    bool called_pk = false;
    bool called_prediction = false;
    int max_iterations = 10;
    
    for (int i = 0; i < max_iterations; i++)
    {
        // Get completion from model
        Completion result = completions(endpoint, model, ctx);
        
        // Check if model made tool calls
        if (result.choice.toolCalls.length == 0)
        {
            // No more tool calls, model is done
            break;
        }
        
        // Execute tool calls
        foreach (tc; result.choice.toolCalls)
        {
            if (tc.name == "get_tramadol_pharmacokinetics")
            {
                called_pk = true;
            }
            if (tc.name == "predict_tramadol_concentration")
            {
                called_prediction = true;
            }

            import std.stdio : writeln;
            writeln("Tool: ", tc.name);
            writeln("Arguments: ", tc.arguments);
            
            Tool tool = endpoint.tools().get(tc.name);
            JSONValue tool_result = tool.execute(tc.arguments);
            ctx.tool(tc.id, tool_result);
        }
    }
    
    // Verify model used both tools (may have called them in different turns)
    assert(called_pk, "Model should have called get_tramadol_pharmacokinetics");
    assert(called_prediction, "Model should have called predict_tramadol_concentration");
    
    // Get final response
    Completion final_result = completions(endpoint, model, ctx);
    
    // Verify final response contains reasonable content
    string response_text = final_result.choice.text;
    assert(response_text.length > 0, "Model should provide a response");
}
