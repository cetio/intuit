module tests.embedding;

import intuit.response.embedding;
import unit_threaded;

import std.exception : assertThrown;
import std.math : isClose, sqrt;

@Name("cosine similarity of identical, orthogonal, and diagonal vectors")
unittest
{
    float[] identical = [1.0f, 0.0f, 0.0f];
    assert(isClose(cosineSimilarity(identical, identical), 1.0f, 1e-5f));

    float[] orthogonal = [0.0f, 1.0f, 0.0f];
    assert(isClose(cosineSimilarity(identical, orthogonal), 0.0f, 1e-5f));

    float[] diagonal = [1.0f, 1.0f, 0.0f];
    float[] axis = [1.0f, 0.0f, 0.0f];
    assert(isClose(cosineSimilarity(diagonal, axis), 1.0f / sqrt(2.0f), 1e-5f));
}

@Name("dot product and euclidean distance")
unittest
{
    float[] first = [1.0f, 2.0f, 3.0f];
    float[] second = [4.0f, 5.0f, 6.0f];
    assert(isClose(dotProduct(first, second), 32.0f, 1e-5f));

    float[] origin = [0.0f, 0.0f, 0.0f];
    float[] point = [3.0f, 4.0f, 0.0f];
    assert(isClose(euclideanDistance(origin, point), 5.0f, 1e-5f));
}

@Name("in-place normalization preserves pointer and yields unit norm")
unittest
{
    float[] vector = [3.0f, 4.0f, 0.0f];
    ref float[] normalized = normalize(vector);
    assert(normalized.ptr == vector.ptr);
    assert(isClose(l2Norm(vector), 1.0f, 1e-5f));
}

@Name("norm mean of two orthogonal unit vectors")
unittest
{
    float[][] embeddings = [
        [1.0f, 0.0f],
        [0.0f, 1.0f]
    ];
    float[] mean = normMean(embeddings);
    float expected = 1.0f / sqrt(2.0f);
    assert(mean.length == 2);
    assert(isClose(mean[0], expected, 1e-5f));
    assert(isClose(mean[1], expected, 1e-5f));
}

@Name("mismatched vector lengths throw")
unittest
{
    float[] shortVec = [1.0f, 2.0f];
    float[] longVec = [1.0f, 2.0f, 3.0f];
    assertThrown!Exception(cosineSimilarity(shortVec, longVec));

    float[][] mismatched = [[1.0f, 2.0f], [1.0f, 2.0f, 3.0f]];
    assertThrown!Exception(normMean(mismatched));
}
