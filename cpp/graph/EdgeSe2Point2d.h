#pragma once

#include "graph/BaseEdge.h"

#include <Eigen/Eigen>

class EdgeSE2Point2d : public BaseEdge<float>
{
public:
    using JacobianMat = Eigen::Matrix<float, 2, 3>;
    using JacobianMatLm = Eigen::Matrix<float, 2, 2>;

public:
    EdgeSE2Point2d(const unsigned id1, const unsigned id2, 
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
        return "se2point2";
    }

};
