#pragma once

#include "graph/BaseEdge.h"
#include "graph/BaseVertex.h"
#include "graph/Types.h"

#include <set>
#include <memory>
#include <vector>
#include <unordered_map>

template<typename T>
class Graph
{
public:
    void AddVertex(const unsigned id, std::unique_ptr<BaseVertex<T>>&& vertex)
    {
        vertices[id] = std::move(vertex);
    }

    void AddEdge(std::unique_ptr<BaseEdge<T>>&& edge)
    {
        edges.push_back(std::move(edge));
    }

    void FixVertex(const unsigned id)
    {
        fixedVertices.insert(id);
    }

    const std::unordered_map<unsigned, std::unique_ptr<BaseVertex<T>>>& GetVertices() const
    {
        return vertices;
    }
    
    const std::vector<std::unique_ptr<BaseEdge<T>>>& GetEdges() const
    {
        return edges;
    }

    const std::set<unsigned> GetFixedVertices() const
    {
        return fixedVertices;
    }

    const BaseVertex<T>* GetVertex(const unsigned id) const
    {
        return vertices.at(id).get();
    }

private:
    std::unordered_map<unsigned, std::unique_ptr<BaseVertex<T>>> vertices;
    std::vector<std::unique_ptr<BaseEdge<T>>> edges;
    std::set<unsigned> fixedVertices;
};