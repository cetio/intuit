module checks.embedding;

import intuit.response.embedding;

unittest
{
    import std.math : isClose;
    float[] a = [1.0f, 0.0f, 0.0f];
    float[] b = [1.0f, 0.0f, 0.0f];
    assert(isClose(cosineSimilarity(a, b), 1.0f, 1e-5f));
}

unittest
{
    import std.math : isClose;
    float[] a = [1.0f, 0.0f, 0.0f];
    float[] b = [0.0f, 1.0f, 0.0f];
    assert(isClose(cosineSimilarity(a, b), 0.0f, 1e-5f));
}

unittest
{
    import std.math : isClose, sqrt;
    float[] a = [1.0f, 1.0f, 0.0f];
    float[] b = [1.0f, 0.0f, 0.0f];
    assert(isClose(cosineSimilarity(a, b), 1.0f / sqrt(2.0f), 1e-5f));
}

unittest
{
    float[] a = [1.0f, 2.0f];
    float[] b = [1.0f, 2.0f, 3.0f];
    try
    {
        auto _ = cosineSimilarity(a, b);
        assert(false);
    }
    catch (Exception) {}
}

unittest
{
    import std.math : isClose;
    float[] a = [1.0f, 2.0f, 3.0f];
    float[] b = [4.0f, 5.0f, 6.0f];
    assert(isClose(dotProduct(a, b), 32.0f, 1e-5f));
}

unittest
{
    import std.math : isClose;
    float[] a = [0.0f, 0.0f, 0.0f];
    float[] b = [3.0f, 4.0f, 0.0f];
    assert(isClose(euclideanDistance(a, b), 5.0f, 1e-5f));
}

unittest
{
    import std.math : isClose;
    float[] v = [3.0f, 4.0f, 0.0f];
    ref float[] result = normalize(v);
    assert(result.ptr == v.ptr);
    assert(isClose(l2Norm(v), 1.0f, 1e-5f));
}

unittest
{
    import std.math : isClose, sqrt;
    float[][] embeddings = [
        [1.0f, 0.0f],
        [0.0f, 1.0f]
    ];
    float[] result = normMean(embeddings);
    float expected = 1.0f / sqrt(2.0f);
    assert(result.length == 2);
    assert(isClose(result[0], expected, 1e-5f));
    assert(isClose(result[1], expected, 1e-5f));
}

unittest
{
    float[][] embeddings = [
        [1.0f, 2.0f],
        [1.0f, 2.0f, 3.0f]
    ];
    try
    {
        auto _ = normMean(embeddings);
        assert(false);
    }
    catch (Exception) {}
}
