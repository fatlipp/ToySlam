#pragma once

#include "graph/BaseVertex.h"

#include <Eigen/Eigen>

class VertexSe2 : public BaseVertex<float>
{
public:
    VertexSe2(const DynamicMatrix<float>& position)
        : BaseVertex(position)
    {
    }

    unsigned GetDims() const override
    {
        return 3;
    }
        
    std::string GetType() const override
    {
        return "pose2d";
    }

    void Update(const DynamicMatrix<float>& delta) override
    {
        const auto theta = std::atan2(position(1, 0), position(0, 0)) + delta(2);
        const auto c = std::cos(theta);
        const auto s = std::sin(theta);
        position(0, 0) = c;
        position(0, 1) = -s;
        position(1, 0) = s;
        position(1, 1) = c;
        position(0, 2) += delta(0);
        position(1, 2) += delta(1);
    }
};