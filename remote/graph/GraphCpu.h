#pragma once

#include "graph/IGraph.h"
#include "graph/vertex/BaseVertexCpu.h"
#include "graph/edge/BaseEdgeCpu.h"

#include <memory>
#include <vector>
#include <unordered_map>

template<typename T>
class GraphCpu : public IGraph
{
public:
    void AddVertex(const unsigned id, std::unique_ptr<BaseVertexCpu<T>>&& vertex)
    {
        vertices[id] = std::move(vertex);
    }

    void AddEdge(const EdgeType type, std::unique_ptr<BaseEdgeCpu<T>>&& edge)
    {
        edges[type].push_back(std::move(edge));
    }

    void FixVertex(const unsigned id)
    {
        fixedVertices.push_back(id);
    }

    const std::unordered_map<unsigned, std::unique_ptr<BaseVertexCpu<T>>>& GetVertices() const
    {
        return vertices;
    }
    
    const std::unordered_map<EdgeType, std::vector<std::unique_ptr<BaseEdgeCpu<T>>>>& GetEdges() const
    {
        return edges;
    }

    const std::vector<unsigned> GetFixedVertices() const
    {
        return fixedVertices;
    }

    const BaseVertexCpu<T>* GetVertex(const unsigned id) const
    {
        return vertices.at(id).get();
    }

    BaseVertexCpu<T>* GetVertex(const unsigned id)
    {
        return vertices.at(id).get();
    }

private:
    std::unordered_map<unsigned, std::unique_ptr<BaseVertexCpu<T>>> vertices;
    std::unordered_map<EdgeType, std::vector<std::unique_ptr<BaseEdgeCpu<T>>>> edges;
    std::vector<unsigned> fixedVertices;
};