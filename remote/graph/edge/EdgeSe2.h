#pragma once

#include "graph/edge/BaseEdgeCpu.h"
#include "graph/Helper.h"

#include <Eigen/Eigen>

template<typename T>
class EdgeSE2 : public BaseEdgeCpu<T>
{
public:
    EdgeSE2(const unsigned id1, const unsigned id2, 
        const DynamicMatrix<T>& meas, 
        const DynamicMatrix<T>& inf)
        : BaseEdgeCpu<T>(EdgeType::Se2, id1, id2, meas, inf)
        {
        }
public:
    std::tuple<DynamicVector<T>, DynamicMatrix<T>, DynamicMatrix<T>> 
        CalcErrorAndJ(const GraphCpu<T>* graph) override;
};

template<typename T>
std::tuple<DynamicVector<T>, DynamicMatrix<T>, DynamicMatrix<T>> 
            EdgeSE2<T>::CalcErrorAndJ(const GraphCpu<T>* graph)
{
    const auto& pos_1 = graph->GetVertex(this->id1)->GetPosition();
    const auto& pos_2 = graph->GetVertex(this->id2)->GetPosition();

    const auto posInv = InverseTransform2d<T>(pos_1);
    const auto pp = posInv * pos_2;
    const auto delta = this->meas.inverse() * pp;
    const auto err = Eigen::Vector3<T>{ delta(0, 2), delta(1, 2), ConvertMatToAngle<T>(delta) };

    const auto J = Eigen::Matrix3<T>::Identity();

    return {err, -J, J};
}