#pragma once

#include "graph/BaseVertex.h"

#include <Eigen/Eigen>

class Vertex2d : public BaseVertex<float>
{
public:
    Vertex2d(const DynamicMatrix<float>& position)
        : BaseVertex(position)
    {
    }

    unsigned GetDims() const override
    {
        return 2;
    }

    std::string GetType() const override
    {
        return "lm2d";
    }

    void Update(const DynamicMatrix<float>& delta) override
    {
        position += delta;
    }

};
