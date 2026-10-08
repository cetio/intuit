module tests.router.modelsdev;

import intuit.exception : TransportException;
import intuit.router.modelsdev : ModelsDev;
import unit_threaded : Name, ShouldFailWith, shouldThrow;

import std.json : JSONValue;

@Name("ModelsDev rejects active model selection")
@ShouldFailWith!Exception
unittest
{
    ModelsDev router = new ModelsDev();
    router.active("model");
}

@Name("ModelsDev rejects current model configuration")
@ShouldFailWith!Exception
unittest
{
    ModelsDev router = new ModelsDev();
    router.config();
}

@Name("ModelsDev rejects named model configuration")
@ShouldFailWith!Exception
unittest
{
    ModelsDev router = new ModelsDev();
    router.config("model");
}

@Name("ModelsDev rejects listing model configurations")
@ShouldFailWith!Exception
unittest
{
    ModelsDev router = new ModelsDev();
    router.configs();
}

@Name("ModelsDev rejects completion requests")
@ShouldFailWith!Exception
unittest
{
    ModelsDev router = new ModelsDev();
    router._completions(JSONValue.init);
}

@Name("ModelsDev rejects decision requests")
@ShouldFailWith!Exception
unittest
{
    ModelsDev router = new ModelsDev();
    router._decisions(JSONValue.init);
}

@Name("ModelsDev rejects embedding requests")
@ShouldFailWith!Exception
unittest
{
    ModelsDev router = new ModelsDev();
    router._embeddings(JSONValue.init);
}

@Name("ModelsDev normalizes unsupported URL transport failures")
unittest
{
    ModelsDev router = new ModelsDev("unsupported://modelsdev-test");
    router.refresh().shouldThrow!TransportException;
}

version(integration)
{
    import intuit.router.details : ModelDetails, ModelCapability, Modality;
    import std.algorithm.searching : canFind;
    import unit_threaded : should;

    @Name("ModelsDev retrieves the public catalog without authentication")
    unittest
    {
        ModelsDev router = new ModelsDev();
        ModelDetails[string] catalog = router.catalog;
        bool supportsVideo;
        bool supportsTools;

        assert(catalog.length > 0);
        foreach (id, details; catalog)
        {
            id.should == details.id;
            supportsVideo |= details.inputModalities.canFind(Modality.Video)
                || details.outputModalities.canFind(Modality.Video);
            supportsTools |= details.capabilities.canFind(ModelCapability.Tools);
        }

        assert(supportsVideo);
        assert(supportsTools);
    }
}
