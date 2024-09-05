#include "cuda/optimizer/kernels/KernelCommon.cuh"
#include "cuda/optimizer/kernels/KernelHelper.cuh"
#include "cuda/Helper.h"
#include "cuda/graph/BaseVertexGpu.h"

template<typename T>
__global__ void FixVertices(const VertexStorageGpu<T>* vertices, unsigned* fixedVertices, const unsigned fixedVerticesCount,
                        T* H, T* b, const unsigned dims)
{
    const int startId = threadIdx.x + blockDim.x * blockIdx.x;

    for (int i = startId; i < fixedVerticesCount; i += blockDim.x * gridDim.x)
    {
        const auto index1 = vertices->indexesInH[fixedVertices[i]];
        const unsigned vertexOffset = vertices->offsets[i];
        const auto vertexType = reinterpret_cast<BaseVertexGpu*>(vertices->layout + vertexOffset)->type;
        const unsigned vertexStateSize = vertexType == VertexType::Se2 ? 3 : 2;

        for (int j = 0; j < vertexStateSize; ++j) 
        {
            H[(index1 + j) * dims + (index1 + j)] += 1e6;
            b[index1 + j] = 0;
        }
    }
}

template<typename T>
__global__ void Update(const VertexStorageGpu<T>* vertices, const unsigned verticesCount, T* b)
{
    const int startId = threadIdx.x + blockDim.x * blockIdx.x;

    for (int i = startId; i < verticesCount; i += blockDim.x * gridDim.x)
    {
        const auto index1 = vertices->indexesInH[i];
        const unsigned vertexOffset = vertices->offsets[i];
        const auto vertexType = reinterpret_cast<BaseVertexGpu*>(vertices->layout + vertexOffset)->type;
        const T MUL = 0.2;

        // TODO: optimize it
        if (vertexType == VertexType::Point2)
        {
            VertexGpuSe2<T>* vertex = reinterpret_cast<VertexGpuSe2<T>*>(vertices->layout + vertexOffset);
            vertex->position[0] += b[index1 + 0] * MUL;
            vertex->position[1] += b[index1 + 1] * MUL;
        }
        else if (vertexType == VertexType::Se2)
        {
            VertexGpuPoint2<T>* vertex = reinterpret_cast<VertexGpuPoint2<T>*>(vertices->layout + vertexOffset);
            const auto theta = MatToAngle(vertex->position) + (b[index1 + 2] * MUL);
            const auto c = cos(theta);
            const auto s = sin(theta);
            vertex->position[0] = c;
            vertex->position[1] = -s;
            vertex->position[3] = s;
            vertex->position[4] = c;
            vertex->position[2] += b[index1 + 0] * MUL;
            vertex->position[5] += b[index1 + 1] * MUL;
        }
    }
}

template<typename T>
__global__ void Negativate(T* data, const unsigned count)
{
    const int startId = threadIdx.x + blockDim.x * blockIdx.x;

    for (int i = startId; i < count; i += blockDim.x * gridDim.x)
    {
        data[i] = -data[i];
    }
}


template __global__ void FixVertices(const VertexStorageGpu<float>* vertices, unsigned* fixedVertices, 
                    const unsigned fixedVerticesCount,
                        float* H, float* b, const unsigned dims);
template __global__ void Update(const VertexStorageGpu<float>* vertices, const unsigned verticesCount, float* b);
template __global__ void Negativate(float* data, const unsigned count);

template __global__ void FixVertices(const VertexStorageGpu<double>* vertices, unsigned* fixedVertices, 
                    const unsigned fixedVerticesCount,
                        double* H, double* b, const unsigned dims);
template __global__ void Update(const VertexStorageGpu<double>* vertices, const unsigned verticesCount, double* b);
template __global__ void Negativate(double* data, const unsigned count);