#pragma once

#include "graph/vertex/BaseVertexCpu.h"

#include <Eigen/Eigen>

template<typename T>
class Vertex2d : public BaseVertexCpu<T>
{
public:
    Vertex2d(const DynamicMatrix<T>& position)
        : BaseVertexCpu<T>(VertexType::Point2, position)
    {
    }

    void Update(const DynamicMatrix<T>& delta) override
    {
        this->position += delta;
    }

};
