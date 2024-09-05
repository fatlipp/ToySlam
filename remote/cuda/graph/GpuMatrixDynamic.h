#pragma once

#include "graph/IMatrix.h"
#include "cuda/Helper.h"

#include <cuda_runtime.h>

template<typename T>
class GpuMatrixDynamic : public IMatrix<T>
{
public:
    GpuMatrixDynamic()
        : IMatrix<T>()
        , data{nullptr}
    {
    }

    GpuMatrixDynamic(const unsigned rows, const unsigned cols)
        : IMatrix<T>(rows, cols)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), sizeof(T) * cols * rows));
        setZero();
    }

    GpuMatrixDynamic(const unsigned rows, const unsigned cols, const T* inpData)
        : IMatrix<T>(rows, cols)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), sizeof(T) * cols * rows));

        if (inpData != nullptr)
        {
            CUDA_CHECK(cudaMemcpy(data, inpData, sizeof(T) * cols * rows, cudaMemcpyHostToDevice));
        }
        else
        {
            setZero();
        }
    }

    GpuMatrixDynamic& operator=(GpuMatrixDynamic&& other) noexcept
    {
        if (this != &other)
        {
            if (data != nullptr)
            {
                CUDA_CHECK(cudaFree(data));
            }

            this->cols = other.cols;
            this->rows = other.rows;
            data = other.data;

            other.cols = 0;
            other.rows = 0;
            other.data = nullptr;
        }
        return *this;
    }

    GpuMatrixDynamic(GpuMatrixDynamic&& other) noexcept = delete;
    GpuMatrixDynamic(GpuMatrixDynamic& other) = delete;
    GpuMatrixDynamic(const GpuMatrixDynamic& other) = delete;
    GpuMatrixDynamic& operator=(GpuMatrixDynamic& other) = delete;
    GpuMatrixDynamic& operator=(const GpuMatrixDynamic& other) = delete;

    ~GpuMatrixDynamic()
    {
        if (data != nullptr)
        {
            CUDA_CHECK(cudaFree(data));
        }
    }

    T const& operator[](const unsigned index) const
    {
        return data[index];
    }

    void setZero() override
    {
        CUDA_CHECK(cudaMemset(data, 0, this->cols * this->rows * sizeof(T)));
    }

    const T* getData() const override
    {
        return data;
    }

    T* getData() override
    {
        return data;
    }

private:
    T* data;
};
