#pragma once

#include "graph/vertex/BaseVertexCpu.h"

#include <Eigen/Eigen>

template<typename T>
class VertexSe2 : public BaseVertexCpu<T>
{
public:
    VertexSe2(const DynamicMatrix<T>& position)
        : BaseVertexCpu<T>(VertexType::Se2, position)
    {
    }

    void Update(const DynamicMatrix<T>& delta) override
    {
        const auto theta = std::atan2(this->position(1, 0), this->position(0, 0)) + delta(2);
        const auto c = std::cos(theta);
        const auto s = std::sin(theta);
        this->position(0, 0) = c;
        this->position(0, 1) = -s;
        this->position(1, 0) = s;
        this->position(1, 1) = c;
        this->position(0, 2) += delta(0);
        this->position(1, 2) += delta(1);
    }
};