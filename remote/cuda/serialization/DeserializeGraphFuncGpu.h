#pragma once

#include "cuda/graph/GraphGpu.h"
#include "cuda/graph/BaseVertexGpu.h"
#include "cuda/graph/BaseEdgeGpu.h"

template<typename T>
struct DeserializeGraphFuncGpu 
{
    static std::unique_ptr<BaseVertexGpu> CreateVertex(const VertexType type, const DynamicMatrix<T>& data) 
    {
        switch (type)
        {
        case VertexType::Se2:
            auto v = std::make_unique<VertexGpuSe2<T>>();
            v->type = type;
            std::memcpy(v->position, data.data(), sizeof(T) * 9);
            return v;
        }

        auto v = std::make_unique<VertexGpuPoint2<T>>();
        v->type = type;
        std::memcpy(v->position, data.data(), sizeof(T) * 2);
        return v;
    }

    static std::unique_ptr<BaseEdgeGpu> CreateEdge(const EdgeType edgeType, const unsigned id1, const unsigned id2, 
        const DynamicMatrix<T>& measurement, const DynamicMatrix<T>& information) 
    {
        if (edgeType == EdgeType::Se2)
        {
            auto e = std::make_unique<EdgeGpuSe2<T>>();
            e->type = edgeType;
            e->id1 = id1;
            e->id2 = id2;
            std::memcpy(e->meas, measurement.data(), sizeof(T) * 9);
            std::memcpy(e->inf, information.data(), sizeof(T) * 9);
            return e;
        }
        else if (edgeType == EdgeType::Se2Point2)
        {
            auto e = std::make_unique<EdgeGpuSe2Point2<T>>();
            e->type = edgeType;
            e->id1 = id1;
            e->id2 = id2;
            std::memcpy(e->meas, measurement.data(), sizeof(T) * 2);
            std::memcpy(e->inf, information.data(), sizeof(T) * 4);
            return e;
        }

        return {};
    }
};