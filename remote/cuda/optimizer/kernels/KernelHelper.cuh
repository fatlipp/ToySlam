#pragma once

#include <cuda_runtime.h>

template<typename T>
__device__ inline static T MatToAngle(const T* R)
{  
    return atan2(R[3], R[0]);
}

template<typename T>
/// @brief Inverse transformation matrix (A^R=A^T)
/// @param mat 
/// @param result 
/// @return 
__device__ inline void inverse(const T* mat, T* result)
{
    const T& a = mat[0]; 
    const T& b = mat[1]; 
    const T& c = mat[3]; 
    const T& d = mat[4]; 
    const T& tx = mat[2];
    const T& ty = mat[5];

    result[0] = a;
    result[1] = c;
    result[3] = b;
    result[4] = d;

    result[2] = -(result[0] * tx + result[1] * ty);
    result[5] = -(result[3] * tx + result[4] * ty);

    result[6] = 0;
    result[7] = 0;
    result[8] = 1;
}

template<typename T, int M, int N, int K>
__device__ inline void mul(const T* A, const T* B, T* result, 
    const bool transA = false, const bool transB = false)
{
#pragma unroll
    for (int row = 0; row < M; ++row) 
    {
#pragma unroll
        for (int col = 0; col < N; ++col) 
        {
            T value = 0;

            for (int k = 0; k < K; ++k) 
            {
                const auto A_element = transA ? A[row + k * M] : A[row * K + k];
                const auto B_element = transB ? B[k + col * K] : B[k * N + col];
                value += A_element * B_element;
            }
    
            result[row * N + col] = value;
        }
    }
}

template<typename T, int ROWS, int COLS>
__device__ inline void transpose(const T* source, T* result)
{
#pragma unroll
    for (int row = 0; row < ROWS; ++row) 
    {
#pragma unroll
        for (int col = 0; col < COLS; ++col) 
        {
            result[row * COLS + col] = source[col * ROWS + row];
        }
    }
}

template<typename T, int DIMX, int DIMY>
__device__ inline void mulVec(const T* A, const T* vec, T* result, const bool transA = false)
{
#pragma unroll
    for (int row = 0; row < DIMY; ++row) 
    {
        T value = 0;

#pragma unroll
        for (int col = 0; col < DIMX; ++col) 
        {
            value += (transA ? A[col * DIMY + row] : A[row * DIMX + col]) * vec[col];
        }
    
        result[row] = value;
    }
}

template<typename T>
__device__ inline void Robustify(const T e, const T delta, const T deltaRobustSqr, T& errRobust, T& resultJ)
{
    if (e <= deltaRobustSqr)
    {
        errRobust = e;
        resultJ = 1;
        return;
    }

    const auto sqrte = sqrt(e);
    errRobust = 2 * sqrte * delta - deltaRobustSqr;
    resultJ = delta / sqrte;
};