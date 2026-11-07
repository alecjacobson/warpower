#pragma once
#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include <cmath>

// Vector Addition
__host__ __device__ inline float3 operator+(const float3& A, const float3& B) { return { A.x + B.x, A.y + B.y, A.z + B.z }; }
__host__ __device__ inline double3 operator+(const double3& A, const double3& B) { return { A.x + B.x, A.y + B.y, A.z + B.z }; }
__host__ __device__ inline int3 operator+(const int3& A, const int3& B) { return { A.x + B.x, A.y + B.y, A.z + B.z }; }

// Vector Subtraction
__host__ __device__ inline float3 operator-(const float3& A, const float3& B) { return { A.x - B.x, A.y - B.y, A.z - B.z }; }
__host__ __device__ inline double3 operator-(const double3& A, const double3& B) { return { A.x - B.x, A.y - B.y, A.z - B.z }; }
__host__ __device__ inline int3 operator-(const int3& A, const int3& B) { return { A.x - B.x, A.y - B.y, A.z - B.z }; }

// Scalar Multiplication
__host__ __device__ inline REAL3 operator*(const REAL& a, const REAL3& B) {
    return { a * B.x, a * B.y, a * B.z };
}

// Scalar Division
__host__ __device__ inline REAL3 operator/(const REAL3& A, const REAL& b) {
    return { A.x / b, A.y / b, A.z / b };
}

// Cross Product
__host__ __device__ inline REAL3 cross3(REAL3 A, REAL3 B) {
    return { A.y * B.z - A.z * B.y, A.z * B.x - A.x * B.z, A.x * B.y - A.y * B.x };
}

// Dot Product
__host__ __device__ inline REAL dot3(REAL3 A, REAL3 B) {
    return A.x * B.x + A.y * B.y + A.z * B.z;
}

// 3x3 Determinant
template <typename T1, typename T2>
__host__ __device__ inline T1 det3x3(T2 p1, T2 p2, T2 p3) {
    return p1.x * (p2.y * p3.z - p2.z * p3.y) - p1.y * (p2.x * p3.z - p2.z * p3.x) + p1.z * (p2.x * p3.y - p2.y * p3.x);
}

// Norm Squared
template <typename T1, typename T2>
__host__ __device__ inline T1 normpow(T2 A) {
    return A.x * A.x + A.y * A.y + A.z * A.z;
}

// Norm
template <typename T1, typename T2>
__host__ __device__ inline T1 norm(T2 A) {
    return sqrt(normpow<T1, T2>(A));
}

// Ball Center
template <typename T1, typename T2>
__host__ __device__ inline T2 ballcenter(T2 p1, T2 p2, T2 p3, T2 p4) {
    T2 A1{ (T1)2 * (p2.x - p1.x), (T1)2 * (p3.x - p1.x), (T1)2 * (p4.x - p1.x) };
    T2 A2{ (T1)2 * (p2.y - p1.y), (T1)2 * (p3.y - p1.y), (T1)2 * (p4.y - p1.y) };
    T2 A3{ (T1)2 * (p2.z - p1.z), (T1)2 * (p3.z - p1.z), (T1)2 * (p4.z - p1.z) };
    T1 invdet = (T1)1.0 / det3x3<T1, T2>(A1, A2, A3);
    T1 b1 = normpow<T1, T2>(p2 - p1);
    T1 b2 = normpow<T1, T2>(p3 - p1);
    T1 b3 = normpow<T1, T2>(p4 - p1);
    T2 x{ invdet * ((A2.y * A3.z - A2.z * A3.y) * b1 + (A2.z * A3.x - A2.x * A3.z) * b2 + (A2.x * A3.y - A2.y * A3.x) * b3) + p1.x,
        invdet * ((A1.z * A3.y - A1.y * A3.z) * b1 + (A1.x * A3.z - A1.z * A3.x) * b2 + (A1.y * A3.x - A1.x * A3.y) * b3) + p1.y,
        invdet * ((A1.y * A2.z - A1.z * A2.y) * b1 + (A1.z * A2.x - A1.x * A2.z) * b2 + (A1.x * A2.y - A1.y * A2.x) * b3) + p1.z };
    return x;
}

// Infinite Normal (2 points)
__host__ __device__ inline REAL3 inf_normal(REAL3 p1, REAL3 p2) {
    REAL3 n = cross3(p1, p2);
    REAL3 A1{ 2 * p1.x, 2 * p2.x, n.x };
    REAL3 A2{ 2 * p1.y, 2 * p2.y, n.y };
    REAL3 A3{ 2 * p1.z, 2 * p2.z, n.z };
    REAL invdet = 1.0f / det3x3<REAL, REAL3>(A1, A2, A3);
    REAL b1 = normpow<REAL, REAL3>(p1);
    REAL b2 = normpow<REAL, REAL3>(p2);

    REAL3 n2{ invdet * ((A2.y * A3.z - A2.z * A3.y) * b1 + (A2.z * A3.x - A2.x * A3.z) * b2),
        invdet * ((A1.z * A3.y - A1.y * A3.z) * b1 + (A1.x * A3.z - A1.z * A3.x) * b2),
        invdet * ((A1.y * A2.z - A1.z * A2.y) * b1 + (A1.z * A2.x - A1.x * A2.z) * b2) };
    REAL m = abs(n2.x) + abs(n2.y) + abs(n2.z);
    n2 = { n2.x / m, n2.y / m, n2.z / m };
    if (dot3(n2, p1) < 0)
        n2 = { -n2.x, -n2.y, -n2.z };
    return cross3(n2, n);
}

// Infinite Normal (3 points)
__host__ __device__ inline REAL3 inf_normal2(REAL3 p1, REAL3 p2, REAL3 p3) {
    REAL3 n = ballcenter<REAL, REAL3>({ 0, 0, 0 }, p1, p2, p3);
    REAL m = abs(n.x) + abs(n.y) + abs(n.z);
    n = { n.x / m, n.y / m, n.z / m };
    if (dot3(n, p1) < 0)
        n = { -n.x, -n.y, -n.z };
    return n;
}

// Tetrahedra Volume
__host__ __device__ inline REAL tet_vol(REAL3 p1, REAL3 p2, REAL3 p3, REAL3 p4) {
    return det3x3<REAL, REAL3>(p2 - p1, p3 - p1, p4 - p1) / 6.0f;
}

// Cast to double3
template <typename T>
__host__ __device__ inline double3 cast_to_double3(const T A) {
    return { (double)A.x, (double)A.y, (double)A.z };
}

// Index Encoding/Decoding Helper
__host__ __device__ inline int decode_index(int v) {
    if (v >= 0) return v;
    return (v == -2147483648) ? 0 : -v; // -2147483648 is INT_MIN
}

// Common Math Utilities
__host__ __device__ inline int divide(int A, int B) {
    return (A - 1) / B + 1;
}

// Higher-order Geometry Functions
__device__ inline REAL weighted_bary(REAL3 p1, REAL3 p2, REAL3 p3, REAL3 p4, REAL3 p5, REAL3& bary) {
    REAL vol1 = tet_vol(p1, p4, p2, p5);
    REAL vol2 = tet_vol(p1, p3, p4, p5);
    REAL vol3 = tet_vol(p1, p2, p3, p5);
    bary = vol1 * (p1 + p4 + p2 + p5) + vol2 * (p1 + p3 + p4 + p5) + vol3 * (p1 + p2 + p3 + p5);
    return (vol1 + vol2 + vol3) * 4;
}

// Vector Max/Min
__host__ __device__ inline int3 vecmax(int3 A, int3 B) {
    return { (A.x < B.x) ? B.x : A.x, (A.y < B.y) ? B.y : A.y, (A.z < B.z) ? B.z : A.z };
}

__host__ __device__ inline int3 vecmin(int3 A, int3 B) {
    return { (A.x > B.x) ? B.x : A.x, (A.y > B.y) ? B.y : A.y, (A.z > B.z) ? B.z : A.z };
}

// Get Cell Index
template <typename T>
__host__ __device__ inline int3 getcellidx(T p, int M) {
    int3 idx = { (int)floorf(p.x * M), (int)floorf(p.y * M), (int)floorf(p.z * M) };
    idx = vecmin(idx, { M - 1, M - 1, M - 1 });
    idx = vecmax(idx, { 0, 0, 0 });
    return idx;
}

template <typename T>
__host__ __device__ inline int getcellidx_flat(T p, int M) {
    int3 idx = getcellidx(p, M);
    return idx.x + idx.y * M + idx.z * M * M;
}
