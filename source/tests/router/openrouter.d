module tests.router.openrouter;

version(integration)
{
    import intuit.router.details : ModelCapability, ModelDetails, Modality;
    import intuit.router.openrouter : OpenRouter;
    import std.algorithm.searching : canFind;
    import unit_threaded : Name;

    @Name("OpenRouter catalog maps API modality and capability values")
    unittest
    {
        OpenRouter router = new OpenRouter("");
        ModelDetails[string] catalog = router.catalog;
        bool supportsFileInput;
        bool supportsEmbeddingsOutput;
        bool supportsPrediction;

        assert(catalog.length > 0);
        foreach (details; catalog.byValue)
        {
            supportsFileInput |= details.inputModalities.canFind(Modality.File);
            supportsEmbeddingsOutput |= details.outputModalities.canFind(Modality.Embeddings);
            supportsPrediction |= details.capabilities.canFind(ModelCapability.Prediction);
        }

        assert(supportsFileInput);
        assert(supportsEmbeddingsOutput);
        assert(supportsPrediction);
    }
}
