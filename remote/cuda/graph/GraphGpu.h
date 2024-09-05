#pragma once

#include "graph/IGraph.h"

#include "cuda/graph/BaseEdgeGpu.h"
#include "cuda/graph/BaseVertexGpu.h"
#include "cuda/graph/VertexStorageGpu.h"
#include "cuda/Helper.h"

#include <vector>
#include <unordered_map>
#include <cstring>

template<typename T>
class GraphGpu : public IGraph
{
public:
    GraphGpu()
        : verticesCount{0}
        , dimension{0}
        , memSize{0}
    {
    }

    ~GraphGpu()
    {
        for (const auto& [t, e] : edgesStorage)
        {
            CUDA_CHECK(cudaFree(e.first));
        }

        VertexStorageGpu<T> storage;
        CUDA_CHECK(cudaMemcpy(&storage, vertexStorage,
                sizeof(VertexStorageGpu<T>), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(storage.layout));
        CUDA_CHECK(cudaFree(storage.offsets));
        CUDA_CHECK(cudaFree(storage.indexesInH));
        CUDA_CHECK(cudaFree(vertexStorage));

        CUDA_CHECK(cudaFree(fixedVerticesPtr));
    }

public:
    void AddVertex(const unsigned id, std::unique_ptr<BaseVertexGpu>&& v)
    {
        memSize += GetVertexSize(v->type);
        verticesCpu[id] = std::move(v);
    }

    void AddEdge(const EdgeType type, std::unique_ptr<BaseEdgeGpu>&& edge)
    {
        edgesCpu[type].push_back(std::move(edge));
    }

    void FixVertex(const unsigned id)
    {
        fixedVerticesCpu.push_back(id);
    }

    std::pair<VertexStorageGpu<T>*, unsigned> GetVerticesGpu()
    {
        return {vertexStorage, verticesCount};
    }

    const std::unordered_map<EdgeType, std::pair<void*, unsigned>>& GetEdgesGpu() const
    {
        return edgesStorage;
    }

    std::pair<unsigned*, unsigned> GetFixedVerticesGpu() const
    {
        return {fixedVerticesPtr, fixedVerticesCount};
    }
    unsigned GetHdimension() const
    {
        return dimension;
    }


    void ToDevice()
    {
        // 1. vertices
        verticesCount = verticesCpu.size();

        // std::cout << "vertices: " << verticesCount << ", mem: " << memSize << "\n"; 

        char* ptr = static_cast<char*>(std::malloc(memSize));

        if (ptr == nullptr)
        {
            throw std::bad_alloc();
        }

        char* currentPtr = ptr;
        std::vector<unsigned> indexesInH(verticesCount, 0);
        std::vector<unsigned> offsets(verticesCount, 0);
        int offset = 0;

        for (const auto& [id, v] : verticesCpu)
        {
            indexesInH[id] = dimension;
            dimension += (v->type == VertexType::Point2 ? 2 : 3);

            const auto size = GetVertexSize(v->type);

            std::memcpy(currentPtr, &(*v), size);
            currentPtr += size;

            offsets[id] = offset;
            offset += size;
        }

        VertexStorageGpu<T> vertexStorage1;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&vertexStorage1.layout), memSize));
        CUDA_CHECK(cudaMemcpy(vertexStorage1.layout, ptr,
            memSize, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&vertexStorage1.offsets), sizeof(unsigned) * verticesCount));
        CUDA_CHECK(cudaMemcpyAsync(vertexStorage1.offsets, offsets.data(),
            sizeof(unsigned) * verticesCount, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&vertexStorage1.indexesInH), sizeof(unsigned) * verticesCount));
        CUDA_CHECK(cudaMemcpyAsync(vertexStorage1.indexesInH, indexesInH.data(),
            sizeof(unsigned) * verticesCount, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&vertexStorage), sizeof(VertexStorageGpu<T>)));
        CUDA_CHECK(cudaMemcpy(vertexStorage, &vertexStorage1,
            sizeof(VertexStorageGpu<T>), cudaMemcpyHostToDevice));

        // 2. edges
        for (const auto& [t, es] : edgesCpu)
        {
            const auto edgesMemSize = GetEdgeSize(t) * es.size();
            // std::cout << "edge: " << static_cast<int>(t) << ", count: " << es.size() << ", mem: " << edgesMemSize << "\n"; 
            if (t == EdgeType::Se2)
            {
                EdgeGpuSe2<T>* ptr = static_cast<EdgeGpuSe2<T>*>(std::malloc(edgesMemSize));
                if (ptr == nullptr)
                {
                    throw std::bad_alloc();
                }

                EdgeGpuSe2<T>* currentPtr = ptr;
                for (const auto& e : es)
                {
                    std::memcpy(currentPtr, &(*e), sizeof(EdgeGpuSe2<T>));
                    ++currentPtr;
                }

                EdgeGpuSe2<T>* edgeStorage;
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&edgeStorage), edgesMemSize));
                CUDA_CHECK(cudaMemcpy(edgeStorage, ptr,
                    edgesMemSize, cudaMemcpyHostToDevice));
                edgesStorage[t] = {edgeStorage, es.size()};
            }
            else if (t == EdgeType::Se2Point2)
            {
                EdgeGpuSe2Point2<T>* ptr = static_cast<EdgeGpuSe2Point2<T>*>(std::malloc(edgesMemSize));
                if (ptr == nullptr)
                {
                    throw std::bad_alloc();
                }

                EdgeGpuSe2Point2<T>* currentPtr = ptr;
                for (const auto& e : es)
                {
                    std::memcpy(currentPtr, &(*e), sizeof(EdgeGpuSe2Point2<T>));
                    ++currentPtr;
                }

                EdgeGpuSe2Point2<T>* edgeStorage;
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&edgeStorage), edgesMemSize));
                CUDA_CHECK(cudaMemcpyAsync(edgeStorage, ptr,
                    edgesMemSize, cudaMemcpyHostToDevice));
                edgesStorage[t] = {edgeStorage, es.size()};
            }
        }

        // 3. Fixed
        fixedVerticesCount = fixedVerticesCpu.size();
        // std::cout << "fixedVerticesCount: " << fixedVerticesCount << "\n"; 

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&fixedVerticesPtr), sizeof(unsigned) * fixedVerticesCount));
        CUDA_CHECK(cudaMemcpyAsync(fixedVerticesPtr, fixedVerticesCpu.data(),
            sizeof(unsigned) * fixedVerticesCount, cudaMemcpyHostToDevice));
    }

    // only vertices
    void ToHost()
    {
        // 1. vertices
        VertexStorageGpu<T> vertexStorage1;
        CUDA_CHECK(cudaMemcpy(&vertexStorage1, vertexStorage, sizeof(VertexStorageGpu<T>), cudaMemcpyDeviceToHost));

        std::vector<unsigned> offsets(verticesCount, 0);
        CUDA_CHECK(cudaMemcpyAsync(offsets.data(), vertexStorage1.offsets,
            sizeof(unsigned) * verticesCount, cudaMemcpyDeviceToHost));

        std::vector<char> layout(memSize, 0);
        CUDA_CHECK(cudaMemcpyAsync(layout.data(), vertexStorage1.layout,
            memSize, cudaMemcpyDeviceToHost));

        for (int i = 0; i < verticesCount; ++i)
        {
            const unsigned offset = offsets[i];
            const auto type = reinterpret_cast<BaseVertexGpu*>(&layout[offset])->type;

            // std::cout << offset<< ", type: " << static_cast<int>(type) << "\n";

            if (type == VertexType::Se2)
            {
                auto vertex = std::make_unique<VertexGpuSe2<T>>();
                std::memcpy(vertex.get(), &layout[offset], GetVertexSize(type));
                verticesCpu[i] = std::move(vertex);
            }
            else if (type == VertexType::Point2)
            {
                auto vertex = std::make_unique<VertexGpuPoint2<T>>();
                std::memcpy(vertex.get(), &layout[offset], GetVertexSize(type));
                verticesCpu[i] = std::move(vertex);
            }
        }
    }

    const std::unordered_map<unsigned, std::unique_ptr<BaseVertexGpu>>& GetVertices() const
    {
        return verticesCpu;
    }
    
    const std::unordered_map<EdgeType, std::vector<std::unique_ptr<BaseEdgeGpu>>>& GetEdges() const
    {
        return edgesCpu;
    }

    const std::vector<unsigned> GetFixedVertices() const
    {
        return fixedVerticesCpu;
    }

private:
    int GetVertexSize(const VertexType type) const
    {
        switch (type)
        {
        case VertexType::Se2:
            return sizeof(VertexGpuSe2<T>);

        default:
            return sizeof(VertexGpuPoint2<T>);
        }

        return 0;
    }

    int GetEdgeSize(const EdgeType type) const
    {
        switch (type)
        {
        case EdgeType::Se2:
            return sizeof(EdgeGpuSe2<T>);

        default:
            return sizeof(EdgeGpuSe2Point2<T>);
        }

        return 0;
    }


private:
    std::unordered_map<unsigned, std::unique_ptr<BaseVertexGpu>> verticesCpu;
    std::unordered_map<EdgeType, std::vector<std::unique_ptr<BaseEdgeGpu>>> edgesCpu;
    std::vector<unsigned> fixedVerticesCpu;

private:
    VertexStorageGpu<T>* vertexStorage;
    unsigned verticesCount;
    std::unordered_map<EdgeType, std::pair<void*, unsigned>> edgesStorage;

    unsigned* fixedVerticesPtr;
    unsigned fixedVerticesCount;

    unsigned dimension;
    unsigned memSize;
};