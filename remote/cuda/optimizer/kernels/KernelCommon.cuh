#pragma once

#include "cuda/graph/VertexStorageGpu.h"
#include "cuda/graph/BaseEdgeGpu.h"

template<typename T>
__global__ void ProcessSe2s(const VertexStorageGpu<T>* vertices, EdgeGpuSe2<T>* edges, const unsigned count,
                        T* H, T* b, T* totalError, const unsigned dims);
template<typename T>
__global__ void ProcessSe2Point2s(const VertexStorageGpu<T>* vertices, EdgeGpuSe2Point2<T>* edges, const unsigned count,
                        T* H, T* b, T* totalError, const unsigned dims);

template<typename T>
__global__ void FixVertices(const VertexStorageGpu<T>* vertices, unsigned* fixedVertices, const unsigned fixedVerticesCount,
                        T* H, T* b, const unsigned dims);

template<typename T>
__global__ void Update(const VertexStorageGpu<T>* vertices, 
    const unsigned verticesCount, T* b);

template<typename T>
__global__ void Negativate(T* data, const unsigned count);