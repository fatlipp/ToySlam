#pragma once

#include "cuda/graph/BaseVertexGpu.h"
#include "cuda/graph/BaseEdgeGpu.h"
#include "cuda/graph/GraphGpu.h"
#include "serialization/SerializeHelper.h"

template<typename T>
struct SerializeGraphFuncGpu 
{
    static std::vector<uint8_t> SerializeVertex(const unsigned id, const BaseVertexGpu* v) 
    {
        std::vector<uint8_t> total;

        std::vector<uint8_t> idBytes = UintToBytes(id);
        total.insert(total.end(), idBytes.begin(), idBytes.end());

        const auto typeBytes = UintToBytes(static_cast<unsigned>(v->type));
        total.insert(total.end(), typeBytes.begin(), typeBytes.end());

        // Serialize vertex position
        std::vector<uint8_t> posBytes;
        if (v->type == VertexType::Se2)
        {
            const auto se2 = static_cast<const VertexGpuSe2<T>*>(v);
            const auto xBytes = FloatToBytes<T>(se2->position[2]);
            posBytes.insert(posBytes.end(), xBytes.begin(), xBytes.end());
            const auto yBytes = FloatToBytes<T>(se2->position[5]);
            posBytes.insert(posBytes.end(), yBytes.begin(), yBytes.end());
            const auto theta = std::atan2(se2->position[3], se2->position[0]);
            const auto thetaBytes = FloatToBytes<T>(theta);
            posBytes.insert(posBytes.end(), thetaBytes.begin(), thetaBytes.end());
        } 
        else if (v->type == VertexType::Point2)
        {
            const auto p2 = static_cast<const VertexGpuPoint2<T>*>(v);
            posBytes = FloatToBytes<T>(p2->position[0]);
            auto yBytes = FloatToBytes<T>(p2->position[1]);
            posBytes.insert(posBytes.end(), yBytes.begin(), yBytes.end());
        }
        total.insert(total.end(), posBytes.begin(), posBytes.end());
        
        return total;
    }

    static std::vector<uint8_t> SerializeEdge(const BaseEdgeGpu* edge) 
    {
        std::vector<uint8_t> total;

        const auto typeBytes = UintToBytes(static_cast<unsigned>(edge->type));
        total.insert(total.end(), typeBytes.begin(), typeBytes.end());

        const auto id1Bytes = UintToBytes(edge->id1);
        total.insert(total.end(), id1Bytes.begin(), id1Bytes.end());

        const auto id2Bytes = UintToBytes(edge->id2);
        total.insert(total.end(), id2Bytes.begin(), id2Bytes.end());

        if (edge->type == EdgeType::Se2)
        {
            const auto p2 = static_cast<const EdgeGpuSe2<T>*>(edge);
            const auto measurementBytes = MatrixToByteArray<T>(3, 3, p2->meas, false);
            total.insert(total.end(), measurementBytes.begin(), measurementBytes.end());

            const auto informationBytes = MatrixToByteArray<T>(3, 3, p2->inf, true);
            total.insert(total.end(), informationBytes.begin(), informationBytes.end());
        }
        else if (edge->type == EdgeType::Se2Point2)
        {
            const auto p2 = static_cast<const EdgeGpuSe2Point2<T>*>(edge);
            const auto measurementBytes = MatrixToByteArray<T>(1, 2, p2->meas, false);
            total.insert(total.end(), measurementBytes.begin(), measurementBytes.end());

            const auto informationBytes = MatrixToByteArray<T>(2, 2, p2->inf, true);
            total.insert(total.end(), informationBytes.begin(), informationBytes.end());
        }

        return total;
    }
};

template std::vector<uint8_t> SerializeGraph<float, GraphGpu<float>, SerializeGraphFuncGpu<float>>(GraphGpu<float>* graphIn);
template std::vector<uint8_t> SerializeGraph<double, GraphGpu<double>, SerializeGraphFuncGpu<double>>(GraphGpu<double>* graphIn);