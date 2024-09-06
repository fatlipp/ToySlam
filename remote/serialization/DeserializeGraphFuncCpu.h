#pragma once

#include "graph/IGraph.h"
#include "graph/edge/EdgeSe2.h"
#include "graph/edge/EdgeSe2Point2d.h"
#include "graph/vertex/VertexSe2.h"
#include "graph/vertex/Vertex2d.h"

#include <memory>

template<typename T>
struct DeserializeGraphFuncCpu 
{
    static std::unique_ptr<BaseVertexCpu<T>> CreateVertex(const VertexType type, const DynamicMatrix<T>& data) 
    {
        switch (type)
        {
        case VertexType::Se2:
            return std::make_unique<VertexSe2<T>>(data);
        }

        return std::make_unique<Vertex2d<T>>(data);
    }

    static std::unique_ptr<BaseEdgeCpu<T>> CreateEdge(const EdgeType edgeType, const unsigned id1, const unsigned id2, 
        const DynamicMatrix<T>& measurement, const DynamicMatrix<T>& information) 
    {
        if (edgeType == EdgeType::Se2)
        {
            return std::make_unique<EdgeSE2<T>>(id1, id2, measurement, information);
        }
        else if (edgeType == EdgeType::Se2Point2)
        {
            return std::make_unique<EdgeSE2Point2d<T>>(id1, id2, measurement, information);
        }

        return {};
    }
};