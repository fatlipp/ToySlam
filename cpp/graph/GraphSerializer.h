#pragma once

#include "graph/BaseEdge.h"
#include "graph/BaseVertex.h"
#include "graph/Helper.h"

#include <Eigen/Eigen>

#include <vector>
#include <unordered_map>
#include <set>
#include <string>
#include <cstdint>


static std::vector<uint8_t> UintToBytes(uint32_t value)
{
    std::vector<uint8_t> bytes(4);
    std::memcpy(bytes.data(), &value, sizeof(uint32_t));
    return bytes;
}

static std::vector<uint8_t> FloatToBytes(float value)
{
    std::vector<uint8_t> bytes(4);
    std::memcpy(bytes.data(), &value, sizeof(float));
    return bytes;
}

static std::vector<uint8_t> StringToBytes(const std::string& value)
{
    std::vector<uint8_t> bytes = UintToBytes(static_cast<uint32_t>(value.length()));
    bytes.insert(bytes.end(), value.begin(), value.end());
    return bytes;
}

static std::vector<uint8_t> MatrixToByteArray(const Eigen::MatrixXf& matrix, bool is_diag = false)
{
    std::vector<uint8_t> bytes;
    
    if (is_diag) 
    {
        const auto bytesRows = UintToBytes(0);
        bytes.insert(bytes.end(), bytesRows.begin(), bytesRows.end());
        const auto bytesCols = UintToBytes(matrix.rows());
        bytes.insert(bytes.end(), bytesCols.begin(), bytesCols.end());
        for (int i = 0; i < matrix.rows(); ++i) 
        {
            std::vector<uint8_t> diagBytes = FloatToBytes(matrix(i, i));
            bytes.insert(bytes.end(), diagBytes.begin(), diagBytes.end());
        }
    } 
    else 
    {
        const auto bytesRows = UintToBytes(matrix.rows());
        bytes.insert(bytes.end(), bytesRows.begin(), bytesRows.end());
        const auto bytesCols = UintToBytes(matrix.cols());
        bytes.insert(bytes.end(), bytesCols.begin(), bytesCols.end());
        for (int i = 0; i < matrix.rows(); ++i)
        {
            for (int j = 0; j < matrix.cols(); ++j)
            {
                std::vector<uint8_t> elemBytes = FloatToBytes(matrix(i, j));
                bytes.insert(bytes.end(), elemBytes.begin(), elemBytes.end());
            }
        }
    }

    return bytes;
}

template<typename T>
static std::vector<uint8_t> SerializeGraph(const Graph<T>* graph)
{
    std::vector<uint8_t> total;

    // 1. Vertices
    const auto& vertices = graph->GetVertices();

    const auto verticesCountBytes = UintToBytes(static_cast<uint32_t>(vertices.size()));
    total.insert(total.end(), verticesCountBytes.begin(), verticesCountBytes.end());
        
    for (const auto& vertex_pair : vertices) 
    {
        const auto id = vertex_pair.first;
        const auto* v = vertex_pair.second.get();

        std::vector<uint8_t> idBytes = UintToBytes(id);
        total.insert(total.end(), idBytes.begin(), idBytes.end());

        std::vector<uint8_t> typeBytes = StringToBytes(v->GetType());
        total.insert(total.end(), typeBytes.begin(), typeBytes.end());

        // Serialize vertex position
        std::vector<uint8_t> posBytes;
        if (v->GetType() == "pose2d") 
        {
            const auto xBytes = FloatToBytes(v->position(0, 2));
            posBytes.insert(posBytes.end(), xBytes.begin(), xBytes.end());
            const auto yBytes = FloatToBytes(v->position(1, 2));
            posBytes.insert(posBytes.end(), yBytes.begin(), yBytes.end());
            const auto theta = std::atan2(v->position(1, 0), v->position(0, 0));
            const auto thetaBytes = FloatToBytes(theta);
            posBytes.insert(posBytes.end(), thetaBytes.begin(), thetaBytes.end());
        } 
        else if (v->GetType() == "lm2d") 
        {
            posBytes = FloatToBytes(v->position(0));
            auto yBytes = FloatToBytes(v->position(1));
            posBytes.insert(posBytes.end(), yBytes.begin(), yBytes.end());
        }
        total.insert(total.end(), posBytes.begin(), posBytes.end());
    }

    // 2. Edges
    const auto& edges = graph->GetEdges();

    auto edgesCountBytes = UintToBytes(static_cast<uint32_t>(edges.size()));
    total.insert(total.end(), edgesCountBytes.begin(), edgesCountBytes.end());

    for (const auto& edge : edges) 
    {
        const auto typeBytes = StringToBytes(edge->GetType());
        total.insert(total.end(), typeBytes.begin(), typeBytes.end());

        const auto id1Bytes = UintToBytes(edge->GetId1());
        total.insert(total.end(), id1Bytes.begin(), id1Bytes.end());

        const auto id2Bytes = UintToBytes(edge->GetId2());
        total.insert(total.end(), id2Bytes.begin(), id2Bytes.end());

        const auto measurementBytes = MatrixToByteArray(edge->meas);
        total.insert(total.end(), measurementBytes.begin(), measurementBytes.end());

        const auto informationBytes = MatrixToByteArray(edge->inf, true);
        total.insert(total.end(), informationBytes.begin(), informationBytes.end());
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