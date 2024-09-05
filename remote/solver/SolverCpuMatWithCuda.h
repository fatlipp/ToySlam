#pragma once

#include "solver/ISolver.h"
#include "cuda/solver/SolverCudaQr.h"

#include "cuda/Helper.h"
#include <memory>

template<typename T>
class SolverCpuMatWithCuda : public ISolver<T>
{
public:
    SolverCpuMatWithCuda()
        : ISolver<T>()
    {
        solver = std::make_unique<SolverCudaQr<T>>();
    }

    void Solve(IMatrix<T>* H, IMatrix<T>* b, unsigned dims) override
    {
        if (devH.getCols() != dims)
        {
            devH = GpuMatrixDynamic<T>{dims, dims};
            devB = GpuMatrixDynamic<T>{1, dims};
        }

        CUDA_CHECK(cudaMemcpy(devH.getData(), H->getData(), 
            dims * dims * sizeof(T), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(devB.getData(), b->getData(), 
            dims * sizeof(T), cudaMemcpyHostToDevice));

        solver->Solve(devH, devB, dims);
        
        CUDA_CHECK(cudaMemcpy(b->getData(), devB.getData(), 
            dims * sizeof(T), cudaMemcpyDeviceToHost));
    }

private:
    GpuMatrixDynamic<T> devH;
    GpuMatrixDynamic<T> devB;
    std::unique_ptr<SolverCudaQr<T>> solver;
};