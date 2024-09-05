#pragma once

#include "serialization/SerializeHelper.h"
#include "tools/BlockTimer.h"

#include <vector>

template<typename T, typename GraphT, typename Functions>
[[nodiscard]] std::vector<uint8_t> SerializeGraph(GraphT* graph)
{
    if (graph == nullptr)
    {
        return {};
    }
    BlockTimer timer{"SerializeGraph"};

    std::vector<uint8_t> total;

    // 1. Vertices
    const auto& vertices = graph->GetVertices();

    const auto verticesCountBytes = UintToBytes(static_cast<uint32_t>(vertices.size()));
    total.insert(total.end(), verticesCountBytes.begin(), verticesCountBytes.end());
        
    for (const auto& vertex_pair : vertices) 
    {
        const auto id = vertex_pair.first;
        const auto* v = vertex_pair.second.get();
        const auto type = v->type;

        const auto bytes = Functions::SerializeVertex(id, v);
        total.insert(total.end(), bytes.begin(), bytes.end());
    }

    // 2. Edges
    const auto& edges = graph->GetEdges();

    int count = 0;
    for (const auto& [type, edges] : edges)
    {
        count += edges.size();
    }

    auto edgesCountBytes = UintToBytes(static_cast<uint32_t>(count));
    total.insert(total.end(), edgesCountBytes.begin(), edgesCountBytes.end());

    for (const auto& [type, edges] : edges)
    {
        for (const auto& edge : edges) 
        {
            const auto bytes = Functions::SerializeEdge(edge.get());
            total.insert(total.end(), bytes.begin(), bytes.end());
        }
    }

    // 3.Fixes
    const auto& fixedVertices = graph->GetFixedVertices();

    const auto fixedCountBytes = UintToBytes(static_cast<uint32_t>(fixedVertices.size()));
    total.insert(total.end(), fixedCountBytes.begin(), fixedCountBytes.end());

    for (const auto& id : fixedVertices) 
    {
        std::vector<uint8_t> idBytes = UintToBytes(id);
        total.insert(total.end(), idBytes.begin(), idBytes.end());
    }

    const auto sizeBytes = UintToBytes(static_cast<uint32_t>(total.size()));
    total.insert(total.begin(), sizeBytes.begin(), sizeBytes.end());

    return total;
}