#pragma once

#include <cusolverDn.h>

template<typename T>
struct SolverCudaOps
{
    using QrFactorizationBufferSizeType = cusolverStatus_t(*)(cusolverDnHandle_t, int, int, T*, int, int*);
    using QrFactorizationType = cusolverStatus_t(*)(cusolverDnHandle_t, int, int, T*, int, T*, T*, int, int *);
    using QbyVectorBType = cusolverStatus_t(*)(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int,
        int, const T *, int, const T *, T *, int, T *, int, int *);

    static QrFactorizationBufferSizeType QrFactorizationBufferSizeCalc;
    static QrFactorizationType QrFactorization;
    static QbyVectorBType QbyVectorB;
};