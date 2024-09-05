#pragma once

#include "graph/vertex/BaseVertexCpu.h"
#include "graph/edge/BaseEdgeCpu.h"
#include "serialization/SerializeHelper.h"

template<typename T>
struct SerializeGraphFuncCpu 
{
    static std::vector<uint8_t> SerializeVertex(const unsigned id, const BaseVertexCpu<T>* v) 
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
            const auto xBytes = FloatToBytes<T>(v->GetPosition()(0, 2));
            posBytes.insert(posBytes.end(), xBytes.begin(), xBytes.end());
            const auto yBytes = FloatToBytes<T>(v->GetPosition()(1, 2));
            posBytes.insert(posBytes.end(), yBytes.begin(), yBytes.end());
            const auto theta = std::atan2(v->GetPosition()(1, 0), v->GetPosition()(0, 0));
            const auto thetaBytes = FloatToBytes<T>(theta);
            posBytes.insert(posBytes.end(), thetaBytes.begin(), thetaBytes.end());
        } 
        else if (v->type == VertexType::Point2)
        {
            posBytes = FloatToBytes<T>(v->GetPosition()(0));
            auto yBytes = FloatToBytes<T>(v->GetPosition()(1));
            posBytes.insert(posBytes.end(), yBytes.begin(), yBytes.end());
        }
        total.insert(total.end(), posBytes.begin(), posBytes.end());
        
        return total;
    }

    static std::vector<uint8_t> SerializeEdge(const BaseEdgeCpu<T>* edge) 
    {
        std::vector<uint8_t> total;

        const auto typeBytes = UintToBytes(static_cast<unsigned>(edge->type));
        total.insert(total.end(), typeBytes.begin(), typeBytes.end());

        const auto id1Bytes = UintToBytes(edge->id1);
        total.insert(total.end(), id1Bytes.begin(), id1Bytes.end());

        const auto id2Bytes = UintToBytes(edge->id2);
        total.insert(total.end(), id2Bytes.begin(), id2Bytes.end());

        const auto m = edge->GetMeas();
        const auto measurementBytes = MatrixToByteArray<T>(m.rows(), m.cols(), m.data(), false);
        total.insert(total.end(), measurementBytes.begin(), measurementBytes.end());

        const auto inf = edge->GetInf();
        const auto informationBytes = MatrixToByteArray<T>(inf.rows(), inf.cols(), inf.data(), true);
        total.insert(total.end(), informationBytes.begin(), informationBytes.end());

        return total;
    }
};