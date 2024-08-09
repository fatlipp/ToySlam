#pragma once

#include "graph/BaseEdge.h"

#include <Eigen/Eigen>

class EdgeSE2 : public BaseEdge<float>
{
public:
    EdgeSE2(const unsigned id1, const unsigned id2, 
        const DynamicMatrix<float>& meas, 
        const DynamicMatrix<float>& inf)
        : BaseEdge(id1, id2, meas, inf)
        {
        }
public:
    std::tuple<DynamicVector<float>, DynamicMatrix<float>, DynamicMatrix<float>> 
        CalcErrorAndJ(const Graph<float>* graph) override;

    std::string GetType() const override
    {
        return "se2";
    }
};
