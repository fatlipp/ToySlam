#include "graph/EdgeSe2Point2d.h"
#include "graph/Graph.h"
#include "graph/Helper.h"

#include <Eigen/Eigen>

std::tuple<DynamicVector<float>, DynamicMatrix<float>, DynamicMatrix<float>> 
            EdgeSE2Point2d::CalcErrorAndJ(const Graph<float>* graph)
{
    const auto& pos = graph->GetVertex(id1)->position;
    const auto& lm = graph->GetVertex(id2)->position;

    auto lm_local = Eigen::Vector2f{meas(0, 0) * std::cos(meas(1, 0)), 
            meas(0, 0) * std::sin(meas(1, 0))};
    const Eigen::Vector3f pp = pos.inverse() * Eigen::Vector3f{lm(0, 0), lm(1, 0), 1};
    auto err = Eigen::Vector2f{ pp[0] - lm_local[0], pp[1] - lm_local[1] };

    const auto x1 = pos(0, 2);
    const auto y1 = pos(1, 2);
    const auto th1 = ConvertMatToAngle(pos);
    const auto cosA = std::cos(th1);
    const auto sinA = std::sin(th1);

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