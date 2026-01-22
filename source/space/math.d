module intuit.space.math;

import inteli.avxintrin;
import std.math : sqrt;
import std.conv : to;

package float hsum(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps!1(v);

    hi = _mm_add_ps(lo, hi);
    hi = _mm_hadd_ps(hi, hi);
    hi = _mm_hadd_ps(hi, hi);
    
    return hi[0];
}

float cosineSimilarity(T : U[], U)(T a, T b)
    if (is(U == float))
{
    if (a.length != b.length)
        throw new Exception("Vectors must be of the same length for cosine similarity!");
    
    if (a.length == 0)
        throw new Exception("Cannot compute cosine similarity of empty vectors!");
    
    float dot = 0;
    float ma = 0;
    float mb = 0;

    float* pa = a.ptr;
    float* pb = b.ptr;
    size_t simdCount = a.length / 8;
    foreach (i; 0..simdCount)
    {
        __m256 va = _mm256_loadu_ps(pa);
        __m256 vb = _mm256_loadu_ps(pb);
        pa += 8;
        pb += 8;

        dot += hsum(_mm256_mul_ps(va, vb));
        ma += hsum(_mm256_mul_ps(va, va));
        mb += hsum(_mm256_mul_ps(vb, vb));
    }

    size_t remainder = a.length - (simdCount * 8);
    if (remainder > 0)
    {
        for (size_t i = 0; i < remainder; i++)
        {
            float va = *pa;
            float vb = *pb;
            dot += va * vb;
            ma += va * va;
            mb += vb * vb;
            pa++;
            pb++;
        }
    }

    if (ma <= 1e-12 || mb <= 1e-12)
        throw new Exception("Cannot compute cosine similarity: one or both vectors have zero norm!");

    return dot / (sqrt(ma) * sqrt(mb));
}

float[] normMean(T : U[][], U)(T embeddings)
    if (is(U == float))
{
    size_t n = embeddings.length;
    if (n == 0)
        return new float[0];

    size_t d = embeddings[0].length;
    
    foreach (e; embeddings)
    {
        if (e.length != d)
        {
            string msg = "All embeddings must have the same length! Expected " ~ 
                         d.to!string ~ ", got " ~ e.length.to!string;
            throw new Exception(msg);
        }
    }
    
    float[] sum = new float[d];
    sum[] = 0.0;

    foreach (e; embeddings)
    {
        float* pe = e.ptr;
        float* ps = sum.ptr;
        size_t simdCount = d / 8;
        
        foreach (i; 0..simdCount)
        {
            __m256 ve = _mm256_loadu_ps(pe);
            __m256 vs = _mm256_loadu_ps(ps);
            __m256 result = _mm256_add_ps(ve, vs);
            _mm256_storeu_ps(ps, result);
            pe += 8;
            ps += 8;
        }
        
        size_t remainder = d - (simdCount * 8);
        if (remainder > 0)
        {
            for (size_t i = 0; i < remainder; i++)
            {
                *ps += *pe;
                ps++;
                pe++;
            }
        }
    }

    float invN = 1.0f / cast(float)n;
    __m256 invNVec = _mm256_set1_ps(invN);
    float* ps = sum.ptr;
    size_t simdCount = d / 8;
    
    foreach (i; 0..simdCount)
    {
        __m256 vs = _mm256_loadu_ps(ps);
        __m256 result = _mm256_mul_ps(vs, invNVec);
        _mm256_storeu_ps(ps, result);
        ps += 8;
    }
    
    size_t remainder = d - (simdCount * 8);
    if (remainder > 0)
    {
        for (size_t i = 0; i < remainder; i++)
        {
            *ps *= invN;
            ps++;
        }
    }

    double sq = 0.0;
    float* psum = sum.ptr;
    size_t simdCountNorm = d / 8;
    
    foreach (i; 0..simdCountNorm)
    {
        __m256 vs = _mm256_loadu_ps(psum);
        __m256 vsq = _mm256_mul_ps(vs, vs);
        float partial = hsum(vsq);
        sq += cast(double)partial;
        psum += 8;
    }
    
    size_t remainderNorm = d - (simdCountNorm * 8);
    if (remainderNorm > 0)
    {
        for (size_t i = 0; i < remainderNorm; i++)
        {
            float v = *psum;
            sq += cast(double)v * cast(double)v;
            psum++;
        }
    }

    double norm = sqrt(sq);
    if (norm <= 1e-12)
        return sum;

    float invNorm = cast(float)(1.0 / norm);
    __m256 invNormVec = _mm256_set1_ps(invNorm);
    float* pnorm = sum.ptr;
    size_t simdCountFinal = d / 8;
    
    foreach (i; 0..simdCountFinal)
    {
        __m256 vs = _mm256_loadu_ps(pnorm);
        __m256 result = _mm256_mul_ps(vs, invNormVec);
        _mm256_storeu_ps(pnorm, result);
        pnorm += 8;
    }
    
    size_t remainderFinal = d - (simdCountFinal * 8);
    if (remainderFinal > 0)
    {
        for (size_t i = 0; i < remainderFinal; i++)
        {
            *pnorm *= invNorm;
            pnorm++;
        }
    }

    return sum;
}

float dotProduct(T : U[], U)(T a, T b)
    if (is(U == float))
{
    if (a.length != b.length)
        throw new Exception("Vectors must be of the same length for dot product!");
    
    float dot = 0;

    float* pa = a.ptr;
    float* pb = b.ptr;
    size_t simdCount = a.length / 8;
    foreach (i; 0..simdCount)
    {
        __m256 va = _mm256_loadu_ps(pa);
        __m256 vb = _mm256_loadu_ps(pb);
        pa += 8;
        pb += 8;

        dot += hsum(_mm256_mul_ps(va, vb));
    }

    size_t remainder = a.length - (simdCount * 8);
    if (remainder > 0)
    {
        for (size_t i = 0; i < remainder; i++)
        {
            dot += *pa * *pb;
            pa++;
            pb++;
        }
    }

    return dot;
}

float euclideanDistance(T : U[], U)(T a, T b)
    if (is(U == float))
{
    if (a.length != b.length)
        throw new Exception("Vectors must be of the same length for Euclidean distance!");
    
    float sqDist = 0;

    float* pa = a.ptr;
    float* pb = b.ptr;
    size_t simdCount = a.length / 8;
    foreach (i; 0..simdCount)
    {
        __m256 va = _mm256_loadu_ps(pa);
        __m256 vb = _mm256_loadu_ps(pb);
        pa += 8;
        pb += 8;

        __m256 diff = _mm256_sub_ps(va, vb);
        __m256 sq = _mm256_mul_ps(diff, diff);
        sqDist += hsum(sq);
    }

    size_t remainder = a.length - (simdCount * 8);
    if (remainder > 0)
    {
        for (size_t i = 0; i < remainder; i++)
        {
            float diff = *pa - *pb;
            sqDist += diff * diff;
            pa++;
            pb++;
        }
    }

    return sqrt(sqDist);
}

float l2Norm(T : U[], U)(T v)
    if (is(U == float))
{
    if (v.length == 0)
        return 0.0f;
    
    double sq = 0.0;

    float* pv = v.ptr;
    size_t simdCount = v.length / 8;
    foreach (i; 0..simdCount)
    {
        __m256 vv = _mm256_loadu_ps(pv);
        __m256 vsq = _mm256_mul_ps(vv, vv);
        float partial = hsum(vsq);
        sq += cast(double)partial;
        pv += 8;
    }

    size_t remainder = v.length - (simdCount * 8);
    if (remainder > 0)
    {
        for (size_t i = 0; i < remainder; i++)
        {
            float val = *pv;
            sq += cast(double)val * cast(double)val;
            pv++;
        }
    }

    return cast(float)sqrt(sq);
}

ref T normalize(T : U[], U)(ref T v)
    if (is(U == float))
{
    float norm = l2Norm(v);
    if (norm <= 1e-12)
        throw new Exception("Cannot normalize zero vector!");
    
    float invNorm = 1.0f / norm;
    __m256 invNormVec = _mm256_set1_ps(invNorm);
    float* pv = v.ptr;
    size_t simdCount = v.length / 8;
    
    foreach (i; 0..simdCount)
    {
        __m256 vv = _mm256_loadu_ps(pv);
        __m256 result = _mm256_mul_ps(vv, invNormVec);
        _mm256_storeu_ps(pv, result);
        pv += 8;
    }
    
    size_t remainder = v.length - (simdCount * 8);
    if (remainder > 0)
    {
        for (size_t i = 0; i < remainder; i++)
        {
            *pv *= invNorm;
            pv++;
        }
    }
    
    return v;
}

unittest
{
    import std.math : isClose, sqrt;
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
    float expected = 1.0f / sqrt(2.0f);
    assert(isClose(cosineSimilarity(a, b), expected, 1e-5f));
}

unittest
{
    import std.math : isClose, sqrt;
    float[] a = [1.0f, 2.0f, 3.0f, 4.0f, 5.0f];
    float[] b = [2.0f, 3.0f, 4.0f, 5.0f, 6.0f];
    float dot = 1*2 + 2*3 + 3*4 + 4*5 + 5*6;
    float ma = 1*1 + 2*2 + 3*3 + 4*4 + 5*5;
    float mb = 2*2 + 3*3 + 4*4 + 5*5 + 6*6;
    float expected = dot / (sqrt(ma) * sqrt(mb));
    assert(isClose(cosineSimilarity(a, b), expected, 1e-5f));
}

unittest
{
    float[] a = [1.0f, 2.0f];
    float[] b = [1.0f, 2.0f, 3.0f];
    try
    {
        auto _ = cosineSimilarity(a, b);
        assert(false, "Should have thrown exception");
    }
    catch (Exception e) {}
}

unittest
{
    float[] a = [0.0f, 0.0f, 0.0f];
    float[] b = [1.0f, 0.0f, 0.0f];
    try
    {
        auto _ = cosineSimilarity(a, b);
        assert(false, "Should have thrown exception");
    }
    catch (Exception e) {}
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
    float[] a = [1.0f, 2.0f, 3.0f, 4.0f, 5.0f];
    float[] b = [2.0f, 3.0f, 4.0f, 5.0f, 6.0f];
    float expected = 1*2 + 2*3 + 3*4 + 4*5 + 5*6;
    assert(isClose(dotProduct(a, b), expected, 1e-5f));
}

unittest
{
    float[] a = [1.0f, 2.0f];
    float[] b = [1.0f, 2.0f, 3.0f];
    try
    {
        auto _ = dotProduct(a, b);
        assert(false, "Should have thrown exception");
    }
    catch (Exception e) {}
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
    import std.math : isClose, sqrt;
    float[] a = [1.0f, 2.0f, 3.0f];
    float[] b = [4.0f, 5.0f, 6.0f];
    float expected = sqrt(3.0f*3.0f + 3.0f*3.0f + 3.0f*3.0f);
    assert(isClose(euclideanDistance(a, b), expected, 1e-5f));
}

unittest
{
    float[] a = [1.0f, 2.0f];
    float[] b = [1.0f, 2.0f, 3.0f];
    try
    {
        auto _ = euclideanDistance(a, b);
        assert(false, "Should have thrown exception");
    }
    catch (Exception e) {}
}

unittest
{
    import std.math : isClose;
    float[] v = [3.0f, 4.0f, 0.0f];
    assert(isClose(l2Norm(v), 5.0f, 1e-5f));
}

unittest
{
    import std.math : isClose, sqrt;
    float[] v = [1.0f, 1.0f, 1.0f];
    assert(isClose(l2Norm(v), sqrt(3.0f), 1e-5f));
}

unittest
{
    import std.math : isClose;
    float[] v = [];
    assert(isClose(l2Norm(v), 0.0f, 1e-5f));
}

unittest
{
    import std.math : isClose;
    float[] v = [3.0f, 4.0f, 0.0f];
    ref float[] result = normalize(v);
    assert(result.ptr == v.ptr);
    assert(isClose(l2Norm(v), 1.0f, 1e-5f));
    assert(isClose(v[0], 0.6f, 1e-5f));
    assert(isClose(v[1], 0.8f, 1e-5f));
    assert(isClose(v[2], 0.0f, 1e-5f));
}

unittest
{
    float[] v = [0.0f, 0.0f, 0.0f];
    try
    {
        auto _ = normalize(v);
        assert(false, "Should have thrown exception");
    }
    catch (Exception e) {}
}

unittest
{
    import std.math : isClose, sqrt;
    float[][] embeddings = [
        [1.0f, 0.0f],
        [0.0f, 1.0f]
    ];
    float[] result = normMean(embeddings);
    assert(result.length == 2);
    float expected = 1.0f / sqrt(2.0f);
    assert(isClose(result[0], expected, 1e-5f));
    assert(isClose(result[1], expected, 1e-5f));
    assert(isClose(l2Norm(result), 1.0f, 1e-5f));
}

unittest
{
    import std.math : isClose, sqrt;
    float[][] embeddings = [
        [1.0f, 2.0f, 3.0f],
        [4.0f, 5.0f, 6.0f],
        [7.0f, 8.0f, 9.0f]
    ];
    float[] result = normMean(embeddings);
    assert(result.length == 3);
    float norm = sqrt(77.0f);
    assert(isClose(result[0], 4.0f / norm, 1e-5f));
    assert(isClose(result[1], 5.0f / norm, 1e-5f));
    assert(isClose(result[2], 6.0f / norm, 1e-5f));
    assert(isClose(l2Norm(result), 1.0f, 1e-5f));
}

unittest
{
    float[][] embeddings = [];
    float[] result = normMean(embeddings);
    assert(result.length == 0);
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
        assert(false, "Should have thrown exception");
    }
    catch (Exception e) {}
}

unittest
{
    import std.math : isClose;
    float[] a = new float[100];
    float[] b = new float[100];
    foreach (i; 0..100)
    {
        a[i] = cast(float)(i + 1);
        b[i] = cast(float)(i + 2);
    }
    float sim = cosineSimilarity(a, b);
    assert(sim >= -1.0f && sim <= 1.0f);
    assert(!isClose(sim, 0.0f, 0.1f));
}

unittest
{
    import std.math : isClose;
    float[] a = new float[100];
    float[] b = new float[100];
    foreach (i; 0..100)
    {
        a[i] = cast(float)(i + 1);
        b[i] = cast(float)(i + 2);
    }
    float dot = dotProduct(a, b);
    assert(dot > 0);
}

unittest
{
    import std.math : isClose;
    float[] a = [1.0f, 2.0f, 3.0f, 4.0f, 5.0f];
    float[] b = [1.0f, 2.0f, 3.0f, 4.0f, 5.0f];
    assert(isClose(euclideanDistance(a, b), 0.0f, 1e-5f));
}

unittest
{
    import std.math : isClose;
    float[] v = [5.0f];
    assert(isClose(l2Norm(v), 5.0f, 1e-5f));
}

unittest
{
    import std.math : isClose;
    float[] v = [5.0f];
    normalize(v);
    assert(isClose(v[0], 1.0f, 1e-5f));
    assert(isClose(l2Norm(v), 1.0f, 1e-5f));
}

unittest
{
    float[] a = [];
    float[] b = [];
    try
    {
        auto _ = cosineSimilarity(a, b);
        assert(false, "Should have thrown exception");
    }
    catch (Exception e) {}
}

