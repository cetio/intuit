module tests.response.embedding;

import intuit.response.embedding;
import unit_threaded;

import std.math : isClose, sqrt;

@Name("cosine similarity of identical, orthogonal, and diagonal vectors")
unittest
{
    float[] identical = [1.0f, 0.0f, 0.0f];
    isClose(cosineSimilarity(identical, identical), 1.0f, 1e-5f).should == true;

    float[] orthogonal = [0.0f, 1.0f, 0.0f];
    isClose(cosineSimilarity(identical, orthogonal), 0.0f, 1e-5f).should == true;

    float[] diagonal = [1.0f, 1.0f, 0.0f];
    float[] axis = [1.0f, 0.0f, 0.0f];
    isClose(cosineSimilarity(diagonal, axis), 1.0f / sqrt(2.0f), 1e-5f).should == true;
}

@Name("dot product and euclidean distance")
unittest
{
    float[] first = [1.0f, 2.0f, 3.0f];
    float[] second = [4.0f, 5.0f, 6.0f];
    isClose(dotProduct(first, second), 32.0f, 1e-5f).should == true;

    float[] origin = [0.0f, 0.0f, 0.0f];
    float[] point = [3.0f, 4.0f, 0.0f];
    isClose(euclideanDistance(origin, point), 5.0f, 1e-5f).should == true;
}

@Name("in-place normalization preserves pointer and yields unit norm")
unittest
{
    float[] vector = [3.0f, 4.0f, 0.0f];
    ref float[] normalized = vector.normalize();
    normalized.ptr.should == vector.ptr;
    isClose(vector.l2Norm(), 1.0f, 1e-5f).should == true;
}

@Name("norm mean of two orthogonal unit vectors")
unittest
{
    float[][] embeddings = [
        [1.0f, 0.0f],
        [0.0f, 1.0f]
    ];
    float[] mean = embeddings.normMean();
    float expected = 1.0f / sqrt(2.0f);
    mean.length.should == 2;
    isClose(mean[0], expected, 1e-5f).should == true;
    isClose(mean[1], expected, 1e-5f).should == true;
}

@Name("mismatched vector lengths throw")
unittest
{
    float[] shortVec = [1.0f, 2.0f];
    float[] longVec = [1.0f, 2.0f, 3.0f];
    cosineSimilarity(shortVec, longVec).shouldThrow!Exception;

    float[][] mismatched = [[1.0f, 2.0f], [1.0f, 2.0f, 3.0f]];
    mismatched.normMean().shouldThrow!Exception;
}
