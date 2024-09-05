#pragma once

#include "cuda/graph/GpuMatrixDynamic.h"
#include "cuda/solver/SolverCudaOps.h"

#include "cuda/Helper.h"

template<typename T>
class SolverCudaQr
{
public:
    SolverCudaQr()
        : neededMemory{-1}
    {
        CUBLAS_CHECK(cublasCreate(&cublasH));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        CUBLAS_CHECK(cublasSetStream(cublasH, stream));
        CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

        CUDA_CHECK(cudaMalloc((void**)&devInfo, sizeof(int)));
    }

    ~SolverCudaQr()
    {
        CUBLAS_CHECK(cublasDestroy(cublasH));
        CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaFree(devInfo));

        if (neededMemory != -1)
        {
            CUDA_CHECK(cudaFree(tau));
            CUDA_CHECK(cudaFree(workerBuffer));
            neededMemory = -1;
        }
    }

    /// @brief 
    /// @param H 
    /// @param b - input b and output delta 
    void Solve(GpuMatrixDynamic<T>& H, GpuMatrixDynamic<T>& b, const unsigned dimension)
    {
        T* devH = H.getData();
        T* devB = b.getData();

        if (neededMemory == -1)
        {
            CUSOLVER_CHECK(SolverCudaOps<T>::QrFactorizationBufferSizeCalc(cusolverH, dimension, dimension, devH, 
                dimension, &neededMemory));

            CUDA_CHECK(cudaMalloc((void**)&workerBuffer, neededMemory * sizeof(T)));
            CUDA_CHECK(cudaMalloc((void**)&tau, dimension * sizeof(T)));
        }
        // upper_triangle(devH) = R 
        // lower_than_diagonal_tri(devH) = Householder vectors of Q
        // tau - factors of Q
        CUSOLVER_CHECK(SolverCudaOps<T>::QrFactorization(cusolverH, dimension, dimension, devH, 
            dimension, tau, workerBuffer, neededMemory, devInfo));
            
        const T alpha = 1.0;
        // CUBLAS_CHECK(cublasSscal(cublasH, dimension, &alpha, devB, 1));

        CUSOLVER_CHECK(SolverCudaOps<T>::QbyVectorB(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, dimension, 1, dimension,
                devH, dimension, tau, devB, dimension, workerBuffer, neededMemory, devInfo));

        // Solve
        if constexpr (std::is_same<T, float>::value)
        {
            CUBLAS_CHECK(cublasStrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, 
                CUBLAS_DIAG_NON_UNIT, dimension, 1, &alpha, devH, dimension, devB, dimension));
        }
        else if constexpr (std::is_same<T, double>::value)
        {
            CUBLAS_CHECK(cublasDtrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, 
                CUBLAS_DIAG_NON_UNIT, dimension, 1, &alpha, devH, dimension, devB, dimension));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

private:
    int neededMemory;
    int *devInfo;
    T *workerBuffer;
    T *tau;

    cusolverDnHandle_t cusolverH;
    cublasHandle_t cublasH;
    cudaStream_t stream;

};