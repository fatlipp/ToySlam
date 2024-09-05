#pragma once

#include "graph/edge/BaseEdgeCpu.h"
#include "graph/Helper.h"

#include <Eigen/Eigen>

template<typename T>
class EdgeSE2Point2d : public BaseEdgeCpu<T>
{
public:
    using JacobianMat = Eigen::Matrix<T, 2, 3>;
    using JacobianMatLm = Eigen::Matrix<T, 2, 2>;

public:
    EdgeSE2Point2d(const unsigned id1, const unsigned id2, 
        const DynamicMatrix<T>& meas, 
        const DynamicMatrix<T>& inf)
        : BaseEdgeCpu<T>(EdgeType::Se2Point2, id1, id2, meas, inf)
        {
        }
public:
    std::tuple<DynamicVector<T>, DynamicMatrix<T>, DynamicMatrix<T>> 
            CalcErrorAndJ(const GraphCpu<T>* graph) override;
};

template<typename T>
std::tuple<DynamicVector<T>, DynamicMatrix<T>, DynamicMatrix<T>> 
            EdgeSE2Point2d<T>::CalcErrorAndJ(const GraphCpu<T>* graph)
{
    const auto& pos = graph->GetVertex(this->id1)->GetPosition();
    const auto& lm = graph->GetVertex(this->id2)->GetPosition();

    auto lm_local = Eigen::Vector2<T>{this->meas(0) * std::cos(this->meas(1)), 
            this->meas(0) * std::sin(this->meas(1))};
    const auto posInv = InverseTransform2d<T>(pos);
    const Eigen::Vector3<T> pp = posInv * Eigen::Vector3<T>{lm(0), lm(1), 1};
    auto err = Eigen::Vector2<T>{ pp[0] - lm_local[0], pp[1] - lm_local[1] };


    // printf("%d - %d, pp: %f, %f, lm_local: %f, %f\n ",  
    //     id1, id2, pp[0], pp[1], lm_local[0], lm_local[1]);
    // printf("%d - %d, pp: %f, %f, lm_local: %f, %f, pos: %f, %f, %f, lm: %f, %f\n ",  
    //     id1, id2, pp[0], pp[1], lm_local[0], lm_local[1], pos(0, 0), pos(0, 1), pos(0, 2), lm(0), lm(1));

    const auto x1 = pos(0, 2);
    const auto y1 = pos(1, 2);
    const auto th1 = ConvertMatToAngle<T>(pos);
    const auto cosA = std::cos(th1);
    const auto sinA = std::sin(th1);

    // printf("%d - %d, pp: %f, %f, lm_local: %f, %f, pos: %f, %f, lm: %f, %f\n ",  
    //     id1, id2, pp[0], pp[1], lm_local[0], lm_local[1], x1, y1, lm(0), lm(1));

    EdgeSE2Point2d::JacobianMat A;
    A(0, 0) = -cosA;
    A(0, 1) = -sinA;
    A(0, 2) = cosA * lm(1, 0) - sinA * lm(0, 0) - cosA * y1 + sinA * x1;
    A(1, 0) = sinA;
    A(1, 1) = -cosA;
    A(1, 2) = -sinA * lm(1, 0) - cosA * lm(0, 0) + sinA * y1 + cosA * x1;

    EdgeSE2Point2d::JacobianMatLm B;
    B(0, 0) = cosA;
    B(0, 1) = sinA;
    B(1, 0) = -sinA;
    B(1, 1) = cosA;

    return {err, A, B};
}