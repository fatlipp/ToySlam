#include "cuda/optimizer/OptimizerGpu.h"
#include "cuda/optimizer/kernels/KernelCommon.cuh"

#include "cuda/graph/GraphGpu.h"
#include "cuda/Helper.h"
#include "cuda/graph/GpuMatrixDynamic.h"

// #include <thrust/device_vector.h>
// #include <thrust/reduce.h>
#include "tools/BlockTimer.h"

#include <iostream>
#include <iomanip>

template<typename T>
void OptimizerGpu<T>::Optimize(IGraph* graphIn)
{
    std::cout << "OptimizeCudaKernel\n";

    auto graph = dynamic_cast<GraphGpu<T>*>(graphIn);

    if (graph == nullptr)
    {
        std::cout << "Wrong GraphCpu\n";

        return;
    }

    BlockTimer timer {"OptimizeGPU"};
    {
        BlockTimer timer {"ToDevice", 2};
        graph->ToDevice();
    }


    auto [vertices, verticesCount] = graph->GetVerticesGpu();
    auto edges = graph->GetEdgesGpu();
    auto dimension = graph->GetHdimension();
    auto [fixedVertices, fixedVerticesCount] = graph->GetFixedVerticesGpu();

    GpuMatrixDynamic<T> H {dimension, dimension, nullptr};
    GpuMatrixDynamic<T> b {1, dimension, nullptr};

    T prevErr = -1;
    int penalty = 0;
    T* totalError;
    CUDA_CHECK(cudaMallocManaged(reinterpret_cast<void**>(&totalError), sizeof(T)));

    cudaStream_t streamSe2, streamSe2Point2;
    cudaStreamCreate(&streamSe2);
    cudaStreamCreate(&streamSe2Point2);

    const dim3 threads{128};
    for (int i = 0; i < this->iterations; ++i)
    {
        CUDA_CHECK(cudaMemsetAsync(H.getData(), 0, sizeof(T) * dimension * dimension));
        CUDA_CHECK(cudaMemsetAsync(b.getData(), 0, sizeof(T) * dimension));
        CUDA_CHECK(cudaMemsetAsync(totalError, 0, sizeof(T)));
        CUDA_CHECK(cudaDeviceSynchronize());

        for (const auto& [t, e] : edges)
        {
            const dim3 blocks{(e.second + threads.x - 1) / threads.x};

            // std::cout << "Edge dims: " << blocks.x << ": " << threads.x << std::endl;

            if (t == EdgeType::Se2)
            {
                ProcessSe2s<T><<<blocks, threads, 0, streamSe2>>>(vertices, static_cast<EdgeGpuSe2<T>*>(e.first), e.second,
                                    H.getData(), b.getData(), totalError, dimension);
            }
            else if (t == EdgeType::Se2Point2)
            {
                ProcessSe2Point2s<T><<<blocks, threads, 0, streamSe2>>>(vertices, static_cast<EdgeGpuSe2Point2<T>*>(e.first), e.second,
                                    H.getData(), b.getData(), totalError, dimension);
            }
        }
        const dim3 blocks{(verticesCount + threads.x - 1) / threads.x};
        FixVertices<T><<<blocks, threads>>>(vertices, fixedVertices, fixedVerticesCount,
                            H.getData(), b.getData(), dimension);
        
        Negativate<T><<<blocks, threads>>>(b.getData(), dimension);
        CUDA_CHECK(cudaDeviceSynchronize());

        // std::cout << i << ") error: " << std::setprecision(8) << *totalError << std::endl;

        // auto getSum = [](T* ptr, const unsigned size) -> T
        //     {
        //         thrust::device_ptr<T> d_ptr(ptr);
        //         thrust::device_vector<T> d_vector(d_ptr, d_ptr + size);
        //         return thrust::reduce(d_vector.begin(), d_vector.end(), 0.0f, thrust::plus<T>());
        //     };
        // std::cout << "sum(H): " << std::setprecision(8) << 
        //     getSum(H.getData(), dimension * dimension) << std::endl;
        // std::cout << "sum(b): " << std::setprecision(8) << 
        //     getSum(b.getData(), dimension) << std::endl;

        if (prevErr > 0 && *totalError > prevErr)
        {
            if (penalty++ > 1)
            {
                break;
            }
        }
        else
        {
            penalty = 0;
        }

        prevErr = *totalError;

        this->solver->Solve(&H, &b, dimension);

        Update<T><<<blocks, threads>>>(vertices, verticesCount, b.getData());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::cout << std::setw(2) << "" << "Summary() error: " << std::setprecision(8) << *totalError << std::endl;

    CUDA_CHECK(cudaFree(totalError));
    
    {
        BlockTimer timer {"ToHost", 2};
        graph->ToHost();
    }
}

template void OptimizerGpu<float>::Optimize(IGraph* graph);
template void OptimizerGpu<double>::Optimize(IGraph* graph);